from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import time

import algorithm.tarp_info as TI
import soft_renderer as sr

from algorithm.pd_cpp import *


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

IMAGE_SIZE=128
VIEW_ANGLE=4  
VIEW_SCALE=0.25
BALANCE_COF=0.001
LEARNING_RATE=0.01
STEP_SIZE=100
DECAY_GAMMA=0.98
MAX_ITER=20000
numerical_error=1e-4
gradient_error=1e-7

figure_x=[]
figure_barrierloss=[]
figure_shadowloss=[]
figure_loss=[]
plt.ion()

class ExternalForce(nn.Module):
    def __init__(self,vertices,tarp_info,boundary_index):
        super(ExternalForce,self).__init__()
        batch_size=vertices.shape[0]
        f=torch.zeros(batch_size,tarp_info.C.size(0)-1,3).cuda().double()
        if False:
            C0_dir=vertices[:,tarp_info.C0,:]-vertices[:,tarp_info.CI,:]
            C0_dir=C0_dir/torch.sqrt(torch.sum(C0_dir**2,dim=2)).t().repeat(1,3)
            C1_dir=vertices[:,tarp_info.C1,:]-vertices[:,tarp_info.CI,:]
            C1_dir=C1_dir/torch.sqrt(torch.sum(C1_dir**2,dim=2)).t().repeat(1,3)
        else:
            f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ -17.0,   17.50,  -19.0 ],[  17.0,   17.50,  -19.0 ],
                             [   0.0,  30.0,  -19.0 ],[   0.0, -30.0,  -19.0 ],
                             [ 20.0,    0.0,   79.05],[-20.0,    0.0,   79.05]]]).cuda().double()
        

        """ self.boundary_index=torch.tensor(boundary_index).cuda()
        f=self.get_init_force(vertices,tarp_info,boundary_index) """
        self.register_buffer("force",nn.Parameter(f))
        self.register_parameter("force_displace",nn.Parameter(torch.zeros_like(f)))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        #self.register_buffer("force_last2_displace",nn.Parameter(torch.zeros_like(f)))

        self.tarp_info=tarp_info
        self.elp=tarp_info.elp
        self.theta2=0.04
        self.dtheta=0.02
        self.dF=2.0
    
    def get_init_force(self,vertices,tarp_info,boundary_index):
        batch_size=vertices.shape[0]
        f=torch.zeros(batch_size,boundary_index.size,3).cuda().double()
        average_weight=tarp_info.mass*tarp_info.g/boundary_index.size
        f[:,:,2]=average_weight
        return f
        

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FConstraint(self,force):
        f2=(force**2).sum(dim=2)
        fz2=force[:,:,2]**2
        """ print(force)
        print('F2Fz2')
        print(F2)
        print(Fz2)
        print(self.logelp(self.tarp_info.Fmax**2-F2, self.tarp_info.Fmax*2-1.0).sum(),
              +self.logelp(Fz2/F2-self.theta2, 0.01).sum()) """
        return self.logelp(self.tarp_info.Fmax**2-f2, self.tarp_info.Fmax*2-self.dF).sum()\
              +self.logelp(fz2/f2-self.theta2, self.dtheta).sum()
    
    def equivalence_projection(self):
        resultant_force=self.force_displace[0].t().sum(dim=1)/self.force_displace.shape[1]
        self.force_displace.data=self.force_displace-resultant_force.unsqueeze(dim=0).repeat(self.force_displace.shape[1],1)

    def linesearch(self):
        deltaforce=self.force_displace-self.force_last_displace
        itertimes=0
        while 1:
            itertimes=itertimes+1
            if itertimes>500:
                if deltaforce.norm()<1e-9:
                    deltaforce=0.0*deltaforce
                    break

            #F2=torch.sum((self.force[0]+self.force_last_displace[0]+deltaforce[0])**2,dim=1)
            f2=((self.force+self.force_last_displace+deltaforce)**2).sum(dim=2)
            ForceMaxCondition=((self.tarp_info.Fmax**2)<(f2*(1.0+numerical_error)))
            if ForceMaxCondition.sum()!=0:
                deltaforce=0.5*deltaforce
                continue

            #Fz2=torch.tensor((self.force[0,:,2]+self.force_last_displace[0,:,2]+deltaforce[0,:,2])**2)
            fz2=(self.force+self.force_last_displace+deltaforce)[:,:,2]**2
            ForceDirCondition=((fz2/f2)<(self.theta2*(1.0+numerical_error)))
            if ForceDirCondition.sum()!=0:
                deltaforce=0.5*deltaforce
                continue
            
            break

        self.force_displace.data=self.force_last_displace+deltaforce


    def forward(self):
        #self.force_last2_displace=self.force_last_displace.clone().detach()
        self.force_last_displace=self.force_displace.clone().detach()
        return self.FConstraint(self.force+self.force_displace)*1e-3

class py_simulatin(torch.autograd.Function):
    @staticmethod
    def forward(ctx,forces,diff_simulator,flag,newton_flag):
        diff_simulator.set_forces(forces[0].clone().detach().cpu().numpy().flatten())
        if flag:
            if newton_flag:
                diff_simulator.newton()
            else:
                diff_simulator.Opt()
        else:
            diff_simulator.compute_jacobi()
        ctx.jacobi=torch.from_numpy(diff_simulator.jacobi.astype(np.float32)).unsqueeze(dim=0).cuda()
        v_size=int(diff_simulator.v.size/3)
        vertices=diff_simulator.v.reshape(v_size,3).astype(np.float32)
        return torch.from_numpy(vertices).unsqueeze(dim=0).cuda()
    
    @staticmethod
    def backward(ctx,grad_vertices):
        grad_vertices=grad_vertices.reshape(1,1,grad_vertices.size(1)*3)
        force_num=int(ctx.jacobi.size(2)/3)
        grad_forces=torch.bmm(grad_vertices,ctx.jacobi).reshape(1,force_num,3)
        return grad_forces,None,None,None
    
class Tarp():
    def __init__(self,args):
        template_mesh=sr.Mesh.from_obj(args.template_mesh)
        self.batch_size=args.batch_size
        self.vertices=template_mesh.vertices
        self.faces=template_mesh.faces

        data=np.loadtxt(args.info_path,dtype=np.float64)
        self.tarp_info=TI.tarp_info(self.vertices,data)

    def get_render_mesh(self):
        return sr.Mesh(self.vertices.repeat(self.batch_size,1,1),self.faces.repeat(self.batch_size,1,1))

def shadow_area(image):
    return torch.sum(image)

class deform():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(data_dir, 'obj/hex06.obj'))
        parser.add_argument('-i', '--image',     type=str,default=os.path.join(data_dir, 'image/pentagram.png'))
        parser.add_argument('-p', '--info-path', type=str, default=os.path.join(data_dir,'info/hex_info06.txt'))
        parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(data_dir, 'results'))
        parser.add_argument('-b', '--batch-size', type=int, default=1)
        self.args = parser.parse_args()

        os.makedirs(self.args.output_dir, exist_ok=True)

        #此处的投影需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.transform=sr.LookAt(perspective=False,viewing_scale=VIEW_SCALE,eye=[0,0,-1.0])
        #self.transform=sr.LookAt(viewing_angle=VIEW_ANGLE,eye=[0,0,-50])
        self.lighting=sr.Lighting(directions=[0,0,1])
        self.rasterizer=sr.SoftRasterizer(image_size=IMAGE_SIZE,sigma_val=1e-4,aggr_func_rgb='hard')

        self.tarp = Tarp(self.args)
        self.simu_pos=self.tarp.vertices[0].clone().detach().cpu().numpy()
        self.diffsimulator=DiffSimulation()
        self.diffsimulator.set_info(
                        self.tarp.faces[0].clone().detach().cpu().numpy(),
                        self.simu_pos.flatten(),
                        self.tarp.tarp_info.C.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.k.clone().detach().cpu().numpy(),
                        0.1,
                        self.tarp.tarp_info.mass.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.CI.clone().detach().cpu().numpy()
                        )
        self.pd_step=0
        self.small_gradient=False

        boundary_index=TI.get_mesh_boundary(self.args.template_mesh)
        
        self.external_force=ExternalForce(self.tarp.vertices,self.tarp.tarp_info,boundary_index).cuda()
        self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=STEP_SIZE,gamma=DECAY_GAMMA)

        self.writer = imageio.get_writer(os.path.join(self.args.output_dir, 'deform.gif'), mode='I')

        self.simu_index0=self.tarp.tarp_info.C0.clone().detach().cpu().numpy()
        self.simu_index1=self.tarp.tarp_info.C1.clone().detach().cpu().numpy()
        self.simu_jacobi=0
        self.itertimes=0
        self.simu_force=0
        self.set_all_forces()

        self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).cuda().unsqueeze(dim=0)
        """ self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        save_image=np.zeros([1,128,128,4])
        save_image[0,:,:,0]=np.where(self.target_image[:,:]>0.0000001,0.5,0.0)
        save_image[0,:,:,1]=np.where(self.target_image[:,:]>0.0000001,0.5,0.0)
        save_image[0,:,:,2]=np.where(self.target_image[:,:]>0.0000001,0.5,0.0)
        save_image[0,:,:,3]=np.where(self.target_image[:,:]>0.0000001,1.0,0.0)
        imageio.imsave(os.path.join(self.args.output_dir,'pentagram.png'),(255*save_image[0]).astype(np.uint8)) """

    def set_all_forces(self):
        resultant_force_displace=-self.external_force.force_displace[0].t().sum(dim=1)
        resultant_force_displace=torch.cat([self.external_force.force_displace[0],resultant_force_displace.unsqueeze(dim=0)],dim=0)
        #print(resultant_force_displace)


        resultant_force=(self.external_force.force[0]+self.external_force.force_displace[0]).t().sum(dim=1)
        resultant_force[0]=-resultant_force[0]
        resultant_force[1]=-resultant_force[1]
        resultant_force[2]=self.tarp.tarp_info.mass*self.tarp.tarp_info.g-resultant_force[2]
        resultant_force=torch.cat([self.external_force.force[0]+self.external_force.force_displace[0],resultant_force.unsqueeze(dim=0)],dim=0)
        #print(resultant_force)
        self.simu_force=resultant_force.clone().detach().cpu().numpy()
       
    def one_iterate(self):

        ps=py_simulatin.apply
        self.pd_step=self.pd_step+1
        vertices=0
        """ if(self.pd_step>15):
            vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,True)
        else:
            vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,False) """
        vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,False)

            
        self.simu_pos=vertices[0].clone().detach().cpu().numpy()
        if self.diffsimulator.print_balance_info(BALANCE_COF)==False:
            return
        
        self.pd_step=0
        vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,False,False)

        """ print(self.external_force.force_displace)
        print(-self.external_force.force_displace[0].t().sum(dim=1)) """
        self.tarp.vertices=vertices
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        shadow_image=self.rasterizer(mesh)




        image=shadow_image.detach().cpu().numpy()[0].transpose((1,2,0))
        """ print('image',image)
        print(image[64][64]) """

        if self.itertimes==0:
            imageio.imsave(os.path.join(self.args.output_dir,'init.png'),(255*image).astype(np.uint8))
        if self.itertimes%100==0:
            imageio.imsave(os.path.join(self.args.output_dir,'deform_%05d.png'%self.itertimes),(255*image).astype(np.uint8))
        #for i in range(10):
        #self.writer.append_data((255*image).astype(np.uint8))
        self.itertimes=self.itertimes+1


        barrier_loss=self.external_force()
        #shadow_loss=-shadow_area(shadow_image)/(IMAGE_SIZE*IMAGE_SIZE)
        shadow_loss=((shadow_image-self.target_image)**2).sum()/(IMAGE_SIZE*IMAGE_SIZE)
        loss=barrier_loss+shadow_loss

        figure_x.append(self.itertimes)
        barrier_loss_cpu=barrier_loss.clone().detach().cpu().numpy()
        shadow_loss_cpu=shadow_loss.clone().detach().cpu().numpy()
        #figure_barrierloss.append(barrier_loss_cpu)
        figure_shadowloss.append(shadow_loss_cpu)
        figure_loss.append(barrier_loss_cpu+shadow_loss_cpu)

        plt.clf()
        #plt.plot(figure_x, figure_barrierloss)
        plt.plot(figure_x, figure_shadowloss,label="-A: negative shadow area",color="blue")#,linestyle=':',linewidth=5,alpha=0.8)
        plt.plot(figure_x, figure_loss,label="-A + B: negative shadow area plus barrier loss",color="red",linestyle=':',linewidth=3)
        plt.legend(loc="upper right")
        if self.itertimes%100==0:
            plt.savefig(os.path.join(self.args.output_dir, 'loss.png'))



        loss.backward()

        self.simu_jacobi=self.diffsimulator.jacobi.astype(np.float32)
        
        self.optimizer.step()
        self.scheduler.step()
        self.external_force.equivalence_projection()

        #self.set_all_forces()
        self.external_force.linesearch()
        self.simu_force=(self.external_force.force+self.external_force.force_displace)[0].clone().detach().cpu().numpy()

        delta=(self.external_force.force_displace-self.external_force.force_last_displace).clone().detach().cpu().numpy()
        if (delta**2).sum()<gradient_error:
            self.small_gradient=True
        


