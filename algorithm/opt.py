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
import math

import algorithm.tarp_info as TI
import soft_renderer as sr
import algorithm.tool as tool
from algorithm.pd_cpp import *


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

IMAGE_SIZE=128
VIEW_ANGLE=4  
VIEW_SCALE=0.25
BALANCE_COF=0.0000001
STEP_SIZE=100
DECAY_GAMMA=0.98
LEARNING_RATE=0.01
MAX_ITER=20000
SIGMA_VAL=1e-4
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
            """ f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ -17.0,   17.50,  -19.0 ],[  17.0,   17.50,  -19.0 ],
                             [   0.0,  30.0,  -19.0 ],[   0.0, -30.0,  -19.0 ],
                             [ 20.0,    0.0,   79.05],[-20.0,    0.0,   79.05]]]).cuda().double() """
            f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ 17.0,   17.50,  -19.0 ],[  -17.0,   17.50,  -19.0 ],
                             [ -20.0,    0.0,   60.05],[20.0,    0.0,   60.05]]]).cuda().double()
        
        #f[:,:,0:2]=0
        #f[:,:,2]=44.1/6
        #f=self.get_init_force(vertices,tarp_info,boundary_index)
        self.register_buffer("force",nn.Parameter(f))
        self.register_parameter("force_displace",nn.Parameter(torch.zeros_like(f)))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        #self.register_buffer("force_last2_displace",nn.Parameter(torch.zeros_like(f)))

        self.tarp_info=tarp_info
        self.elp=tarp_info.elp
        self.theta2=0.04
        self.dtheta=0.02
        self.dF=2.0

        self.xi=1.0
        self.eta=0
        self.rho=0.5
        self.epsilon=1.0e-6
    
    def get_init_force(self,vertices,tarp_info,boundary_index):
        val=tool.force_SOCP(vertices[0],boundary_index,tarp_info.CI,
                            (tarp_info.mass*tarp_info.g*0.1/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        #f=torch.zeros(batch_size,boundary_index.shape[0],3).cuda()
        f=np.zeros((batch_size,boundary_index.shape[0],3))
        average_weight=tarp_info.mass*tarp_info.g/boundary_index.shape[0]
        f[:,:,2]=average_weight.cpu().numpy()
        #f[0,:,0]=val[:,0]
        #f[0,:,1]=val[:,1]
        return torch.from_numpy(f).cuda()
        

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FConstraint(self,force):
        return torch.tensor([0]).cuda()
        f2=(force**2).sum(dim=2)
        fz2=force[:,:,2]**2
        """ print(force)
        print('F2Fz2')
        print(F2)
        print(Fz2)"""
        """ print(self.tarp_info.Fmax**2-f2, self.tarp_info.Fmax*2-1.0,
              +self.logelp(fz2/f2-self.theta2, 0.01).sum())  """
        return self.logelp(self.tarp_info.Fmax**2-f2, self.tarp_info.Fmax*2-self.dF).sum()\
              +self.logelp(fz2/f2-self.theta2, self.dtheta).sum()
    
    def L1Norm(self,force):
        return torch.tensor([0]).cuda()
        force_magnitude=torch.sqrt((force[0]**2).sum(dim=1)+self.epsilon)
        weight=torch.pow(force_magnitude.clone().detach()+self.xi,self.eta-1.0)
        #self.xi=self.xi*self.rho
        print('xi:',self.xi)
        return (force_magnitude*weight).sum()
        force_sqrt_magnitude=torch.sqrt(force_magnitude).clone().detach()
        return (force_magnitude/force_sqrt_magnitude).sum()

    def equivalence_projection(self):
        resultant_force=self.force_displace[0].t().sum(dim=1)/self.force_displace.shape[1]
        self.force_displace.data=self.force_displace-resultant_force.unsqueeze(dim=0).repeat(self.force_displace.shape[1],1)

    def linesearch(self):
        return
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
        self.force_last_displace=self.force_displace.clone().detach()
        return self.FConstraint(self.force+self.force_displace)*1e-3\
              +self.L1Norm(self.force+self.force_displace)*1e-4

class py_simulatin(torch.autograd.Function):
    @staticmethod
    def forward(ctx,forces,diff_simulator,flag,newton_flag):
        diff_simulator.set_forces(forces[0].clone().detach().cpu().numpy().flatten())
        if newton_flag==True:
            diff_simulator.newton()
        else:
            diff_simulator.Opt()
        if flag==False:
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
        self.begin_time=time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(data_dir, 'obj/hex400.obj'))
        parser.add_argument('-i', '--image',     type=str,default=os.path.join(data_dir, 'image/square40.png'))
        parser.add_argument('-p', '--info-path', type=str, default=os.path.join(data_dir,'info/hex_info400.txt'))
        parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(data_dir, 'results'))
        parser.add_argument('-b', '--batch-size', type=int, default=1)
        self.args = parser.parse_args()

        os.makedirs(self.args.output_dir, exist_ok=True)

        #此处的投影需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.transform=sr.LookAt(perspective=False,viewing_scale=VIEW_SCALE,eye=[0,0,-1.0])
        #self.transform=sr.LookAt(viewing_angle=VIEW_ANGLE,eye=[0,0,-50])
        self.lighting=sr.Lighting(directions=[0,0,1])
        self.rasterizer=sr.SoftRasterizer(image_size=IMAGE_SIZE,sigma_val=SIGMA_VAL,gamma_val=5e-3,aggr_func_rgb='hard')

        boundary_index=TI.get_mesh_boundary(self.args.template_mesh)
        self.tarp = Tarp(self.args)
        #self.tarp.tarp_info.C=boundary_index
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

        
        self.external_force=ExternalForce(self.tarp.vertices,self.tarp.tarp_info,boundary_index).cuda()
        self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=STEP_SIZE,gamma=DECAY_GAMMA)

        self.writer = imageio.get_writer(os.path.join(self.args.output_dir, 'deform.gif'), mode='I')

        #self.simu_index0=self.tarp.tarp_info.C0.clone().detach().cpu().numpy()
        #self.simu_index1=self.tarp.tarp_info.C1.clone().detach().cpu().numpy()
        self.simu_index=self.tarp.tarp_info.C.clone().detach().cpu().numpy()
        self.simu_jacobi=0
        self.itertimes=0
        self.simu_force=0
        self.simu_force_grad=0
        self.simu_equa_force_grad=0
        self.simu_vertices_grad=0
        self.set_all_forces()

        self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).cuda().unsqueeze(dim=0)

        para_dir=str(math.floor(time.time()))+'+'+str(self.tarp.tarp_info.C.clone().detach().cpu().numpy().shape[0])+'+'+str(BALANCE_COF)\
                 +'+'+str(LEARNING_RATE)+'+'+os.path.splitext(os.path.split(self.args.image)[1])[0]
        self.para_result=os.path.join(self.args.output_dir,para_dir)
        if os.path.exists(self.para_result)==False:
            os.mkdir(self.para_result)
        """ self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        save_image=np.zeros([1,128,128,4])
        save_image[0,46:82,46:82,0]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,1]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,2]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,3]=1.0#np.where(self.target_image[44:84,44:84]>0.0000001,1.0,0.0)
        imageio.imsave(os.path.join(self.args.output_dir,'square36.png'),(255*save_image[0]).astype(np.uint8)) """

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
        self.simu_force_grad=self.simu_force
        self.simu_equa_force_grad=self.simu_force
        self.simu_vertices_grad=torch.zeros_like(self.tarp.vertices).clone().detach().cpu().numpy()[0]
       
    def one_iterate(self):
        if self.itertimes==400:
            resfs=0

        ps=py_simulatin.apply
        self.pd_step=self.pd_step+1
        vertices=0
        """ if(self.pd_step>50):
            vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,True)
        else:
            vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,False) """
        vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,True,False)

            
        self.simu_pos=vertices[0].clone().detach().cpu().numpy()
        if self.diffsimulator.print_balance_info(BALANCE_COF)==False:
            return
        
        self.pd_step=0
        ssize=6
        ooff=0.0001

        offset_f=torch.zeros([1,ssize,3]).cuda().double()
        offset_f[0,0,0]=ooff
        offset_f[0,1,0]=-ooff
        offset_vertices=ps(self.external_force.force+offset_f+self.external_force.force_displace,self.diffsimulator,False,False)
        numex_jacobi=(offset_vertices-vertices)[0].flatten()/ooff

        #diffx_jacobi=torch.from_numpy(self.diffsimulator.jacobi)
        offset_f[0,0,1]=ooff
        offset_f[0,1,1]=-ooff
        offset_f=torch.zeros([1,ssize,3]).cuda().double()
        offset_vertices=ps(self.external_force.force+offset_f+self.external_force.force_displace,self.diffsimulator,False,False)
        numey_jacobi=(offset_vertices-vertices)[0].flatten()/ooff
        #diffy_jacobi=torch.from_numpy(self.diffsimulator.jacobi)

        offset_f=torch.zeros([1,ssize,3]).cuda().double()
        offset_f[0,0,2]=ooff
        offset_f[0,1,2]=-ooff
        offset_vertices=ps(self.external_force.force+offset_f+self.external_force.force_displace,self.diffsimulator,False,False)
        numez_jacobi=(offset_vertices-vertices)[0].flatten()/ooff
        #diffz_jacobi=torch.from_numpy(self.diffsimulator.jacobi)

        vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator,False,False)
        diff_jacobi=torch.from_numpy(self.diffsimulator.jacobi)

        print(numex_jacobi[0]-numex_jacobi[3],numex_jacobi[1]-numex_jacobi[4],numex_jacobi[2]-numex_jacobi[5])

        leftmat=self.diffsimulator.leftmat

        
        sumfun=torch.zeros([3,3]).cuda()
        sumfun[0,0]=1.0
        sumfun[1,1]=1.0
        sumfun[2,2]=1.0
        x10=(vertices[0][1]-vertices[0][0]).reshape(3,1)
        x40=(vertices[0][4]-vertices[0][0]).reshape(3,1)
        x60=(vertices[0][6]-vertices[0][0]).reshape(3,1)
        norm01=x10.norm()
        norm04=x40.norm()
        norm06=x60.norm()

        mat01=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            mat01[i,i]=1.0-2.7/norm01
        mat01=mat01+2.7*x10*x10.transpose(0,1)/(norm01*norm01*norm01)
        mat04=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            mat04[i,i]=1.0-1.684399/norm04
        mat04=mat04+1.684399*x40*x40.transpose(0,1)/(norm04*norm04*norm04)
        mat06=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            mat06[i,i]=1.0-1.9885/norm06
        mat06=mat06+1.9885*x60*x60.transpose(0,1)/(norm06*norm06*norm06)
        
        difmat01=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                difmat01[i,j]=diff_jacobi[1*3+i,0*3+j]-diff_jacobi[0*3+i,0*3+j]
        difmat04=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                difmat04[i,j]=diff_jacobi[4*3+i,0*3+j]-diff_jacobi[0*3+i,0*3+j]
        difmat06=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                difmat06[i,j]=diff_jacobi[6*3+i,0*3+j]-diff_jacobi[0*3+i,0*3+j]
 
        print(torch.mm(mat01,difmat01)+torch.mm(mat04,difmat04)+torch.mm(mat06,difmat06))

        nume_jacobi=[numex_jacobi,numey_jacobi,numez_jacobi]
        numemat01=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                numemat01[i,j]=nume_jacobi[j][1*3+i]-nume_jacobi[j][0*3+i]
        numemat04=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                numemat04[i,j]=nume_jacobi[j][4*3+i]-nume_jacobi[j][0*3+i]
        numemat06=torch.zeros([3,3]).cuda().double()
        for i in range(3):
            for j in range(3):
                numemat06[i,j]=nume_jacobi[j][6*3+i]-nume_jacobi[j][0*3+i]

        print(torch.mm(mat01,numemat01)+torch.mm(mat04,numemat04)+torch.mm(mat06,numemat06))
        
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
            imageio.imsave(os.path.join(self.para_result,'init.png'),(255*image).astype(np.uint8))
        if self.itertimes%20==0:
            imageio.imsave(os.path.join(self.para_result,'deform_%05d.png'%self.itertimes),(255*image).astype(np.uint8))
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
            plt.savefig(os.path.join(self.para_result, 'loss.png'))

        vertices.retain_grad()
        loss.backward()

        #self.simu_jacobi=self.diffsimulator.jacobi.astype(np.float32)
        
        self.optimizer.step()
        #self.simu_force_grad=(self.external_force.force_displace-self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
        self.simu_force_grad=-self.external_force.force_displace.grad[0].clone().detach().cpu().numpy()
        
        self.scheduler.step()
        self.external_force.equivalence_projection()

        #self.set_all_forces()
        self.external_force.linesearch()
        self.simu_force=(self.external_force.force+self.external_force.force_displace)[0].clone().detach().cpu().numpy()
        self.simu_equa_force_grad=(self.external_force.force_displace-self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
        #self.simu_equa_force_grad=-vertices[0,self.tarp.tarp_info.C,:].clone().detach().cpu().numpy()
        self.simu_vertices_grad=-vertices.grad[0].clone().detach().cpu().numpy()

        delta=(self.external_force.force_displace-self.external_force.force_last_displace).clone().detach().cpu().numpy()

        if self.itertimes%200==0:
            self.write_results()
        if (delta**2).sum()<gradient_error:
            self.small_gradient=True
            self.write_results()
            print('small gradient')
            run_time=np.floor(time.time()-self.begin_time)
            hour_time=run_time//3600
            minute_time=(run_time-3600*hour_time)//60
            second_time=run_time-3600*hour_time-60*minute_time
            print (f'run time:{hour_time}hour{minute_time}minute{second_time}second')


    def write_results(self):
        self.tarp.get_render_mesh().save_obj(os.path.join(self.para_result,'result.obj'))

        np.set_printoptions(threshold=self.tarp.tarp_info.C.shape[0]*3)
        file=0
        if os.path.exists(os.path.join(self.para_result,'force.txt'))==False:
            os.mknod(os.path.join(self.para_result,'force.txt'))
        file=open(os.path.join(self.para_result,'force.txt'),'w')
        file_force=self.external_force.force[0].flatten().clone().detach().cpu().numpy()
        for i in range(file_force.shape[0]):
            print(file_force[i],file=file)
        file.close()

        if os.path.exists(os.path.join(self.para_result,'force_displace.txt'))==False:
            os.mknod(os.path.join(self.para_result,'force_displace.txt'))
        file=open(os.path.join(self.para_result,'force_displace.txt'),'w')
        file_force=self.external_force.force_displace[0].flatten().clone().detach().cpu().numpy()
        for i in range(file_force.shape[0]):
            print(file_force[i],file=file)
        file.close()


