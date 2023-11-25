from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import time
import math

import algorithm.tarp_info as TI
import soft_renderer as sr
import algorithm.tool as tool
from algorithm.pd_cpp import *


figure_x=[]
figure_barrierloss=[]
figure_shadowloss=[]
figure_loss=[]
plt.ion()

params=TI.tarp_params()

class ExternalForce(nn.Module):
    def __init__(self,vertices,tarp_info,boundary_index):
        super(ExternalForce,self).__init__()
        batch_size=vertices.shape[0]
        f=torch.zeros(batch_size,tarp_info.C.size(0)-1,3).cuda().double()
        """ f=torch.tensor([[[   9.5,  13.09,   -10.0],[   -9.5,    13.09,  -10.0 ],
                             [  15.4,    -5.0,  -10.0],[   -15.4,     -5.0,   -10.0],
                             [   0.0,    -16.18,   -10.0],
                             [ 0.0,  20.0,  18.82 ],
                             [  19.0,  6.18,    18.82],[  -19.0,     6.18,   18.82 ],
                             [  11.8,  -16.18,  18.82],[  -11.8,   -16.18,  18.82 ]
                             ]]).cuda().double()  """
        f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ -17.0,   17.50,  -19.0 ],[  17.0,   17.50,  -19.0 ],
                             [   0.0,  30.0,  -19.0 ],[   0.0, -30.0,  -19.0 ],
                             [ 20.0,    0.0,   79.05],[-20.0,    0.0,   79.05]]]).cuda().double()
        """ f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ 17.0,   17.50,  -19.0 ],[  -17.0,   17.50,  -19.0 ],
                             [ -20.0,    0.0,   60.05],[20.0,    0.0,   60.05]]]).cuda().double() """
        """ f=torch.tensor([[[ -17.32,  -10.0,  -20.0 ],[  17.32,  -10.0,  -20.0 ],
                             [ 0.0,    20.0,   84.1]]]).cuda().double() """
        
        
        #f[:,:,0:2]=0
        #f[:,:,2]=44.1/8
        f=self.get_init_force(vertices,tarp_info,boundary_index)
        """ f[:,:,2]=(60-44.1)/(boundary_index.shape[0]-2)
        f[:,2,2]=3
        f[:,3,2]=3 """
        #f[0,0,0]=f[0,0,0]+0.0001
        #f[0,-1,0]=f[0,-1,0]-0.0001
        """ rrr=np.array([0,1,4,5,56,156,2,205])
        fo=torch.zeros((batch_size,boundary_index.shape[0],3)).cuda().double()
        fo[0,rrr,:]=f[0,0:8,:]
        f=fo """
        #data=np.loadtxt(args.info_path,dtype=np.float64)


        

        if params.use_denseInfo:
            f=torch.from_numpy(np.loadtxt(params.force_file,
                                          dtype=np.float64)).reshape(batch_size,f.shape[1],3).cuda().double()
        self.register_buffer("force",nn.Parameter(f))
        uniquesize=self.force.shape[1]-1
        f=torch.zeros((batch_size,uniquesize,3)).double()
        if params.use_denseInfo:
            f=torch.from_numpy(np.loadtxt(params.forcedis_file,
                                          dtype=np.float64)).reshape(batch_size,f.shape[1],3).cuda().double()
        self.register_parameter("force_displace",nn.Parameter(f))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        
        transform=torch.zeros((batch_size,self.force.shape[1],uniquesize)).double()
        for i in range(uniquesize):
            transform[:,i,i]=1.0
            #transform[:,i+1,i]=-1.0
        transform[:,uniquesize,:]=-1.0
        self.register_buffer("transform",nn.Parameter(transform))

        self.tarp_info=tarp_info
        self.elp=tarp_info.elp
        """ self.theta2=0.04
        self.dtheta=0.02
        self.dF=2.0

        self.alpha=0.2
        self.lamda=1e-3
        self.beta=1.0 """
        self.xi=params.xi
        self.eta=params.eta
        self.rho=params.rho
        self.epsilon=params.epsilon
    
    def get_init_force(self,vertices,tarp_info,boundary_index):
        val=tool.force_SOCP(vertices[0],boundary_index,tarp_info.CI,
                            (tarp_info.mass*tarp_info.g/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        f=np.zeros((batch_size,boundary_index.shape[0],3))
        #print(val[:,0].sum(),val[:,1].sum())
        f[0,:,0]=val[:,0]
        f[0,:,1]=val[:,1]

        """ #cut the boundary to several segments and set upward and downward initial force separately
        segment_num=8 
        upward=np.array([]).astype(int)
        downward=np.array([]).astype(int)
        for i in range(segment_num):
            start=np.floor(boundary_index.shape[0]*i/segment_num).astype(int)
            end=np.floor(boundary_index.shape[0]*(i+1)/segment_num).astype(int)
            if i%2==0:
                upward=np.append(upward,boundary_index[start:end].cpu().numpy())
            else:
                downward=np.append(downward,boundary_index[start:end].cpu().numpy())
        
        average_weight=tarp_info.mass*tarp_info.g/boundary_index.shape[0]
        f[:,downward,2]=-average_weight.cpu().numpy()
        f[:,upward,2]=(tarp_info.mass*tarp_info.g+average_weight*downward.shape[0]).cpu().numpy()/upward.shape[0]
        #print(f[:,:,0].sum(),f[:,:,1].sum(),f[:,:,2].sum())
        return torch.from_numpy(f).cuda() """

        average_weight=tarp_info.mass*tarp_info.g/boundary_index.shape[0]
        f[:,:,2]=average_weight.cpu().numpy()
        return torch.from_numpy(f).cuda()
        
    def now_force(self):
        return self.force+torch.bmm(self.transform,self.force_displace)

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FmaxConstraint(self,forces):
        if params.fmax_cons:
            f2=(forces**2).sum(dim=2)
            return self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*2-params.force_delay).sum()
        else:
            return torch.tensor([0]).cuda()

    def FConstraint(self,forces):
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
    
    def L1Norm(self,forces):
        #return torch.tensor([0]).cuda()
        force_magnitude=torch.sqrt((forces[0]**2).sum(dim=1)+self.epsilon)
        weight=torch.pow(force_magnitude.clone().detach()+self.xi,self.eta-1.0)
        self.xi=self.xi*self.rho
        #print('xi:',self.xi)
        return (force_magnitude*weight).sum()

    def equivalence_projection(self):
        return
        resultant_force=self.force_displace[0].t().sum(dim=1)/self.force_displace.shape[1]
        self.force_displace.data=self.force_displace-resultant_force.unsqueeze(dim=0).repeat(self.force_displace.shape[1],1)

    def linesearch(self):
        if params.fmax_cons+params.fdir_cons==0:
            return
        
        deltaforce=self.force_displace-self.force_last_displace
        itertimes=0
        while 1:
            itertimes=itertimes+1
            if itertimes>500:
                if deltaforce.norm()<1e-9:
                    deltaforce=0.0*deltaforce
                    break
            
            forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
            f2=(forces**2).sum(dim=2)
            ForceMaxCondition=((self.tarp_info.Fmax**2)<(f2*(1.0+params.nume_error)))
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
    
    def record_last_displace(self):
        self.force_last_displace=self.force_displace.clone().detach()

    def forward(self):
        forces=self.force+torch.bmm(self.transform,self.force_displace)
        return self.FmaxConstraint(forces)*params.fmax_weight
        return self.FConstraint(forces)*1e-3\
              +self.L1Norm(forces)*1e-4

def shadow_image_loss(shadow_image,target_image):
    if params.loss_type=='image':
        return ((shadow_image-target_image)**2).sum()/(params.image_size*params.image_size)
    elif params.loss_type=='area':
        return -torch.sum(shadow_image)/(params.image_size*params.image_size)
    else:
        return torch.tensor([0]).cuda()

class deform():
    def __init__(self):
        self.begin_time=time.time()
        
        os.makedirs(params.output_dir, exist_ok=True)

        if params.use_denseInfo:
            params.template_mesh=params.result_mesh


        self.transform=sr.LookAt(perspective=False,viewing_scale=params.view_scale,eye=[0,0,-1.0])
        #self.transform=sr.LookAt(viewing_angle=VIEW_ANGLE,eye=[0,0,-50])
        self.lighting=sr.Lighting(directions=[0,0,1])
        self.rasterizer=sr.SoftRasterizer(image_size=params.image_size,sigma_val=params.sigmal_value,aggr_func_rgb='hard')

        boundary_index=tool.get_mesh_boundary(params.template_mesh)
        self.tarp = TI.Tarp(params)
        self.tarp.tarp_info.C=boundary_index
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
        self.newton_flag=False
        self.small_gradient=False

        self.external_force=ExternalForce(self.tarp.vertices,self.tarp.tarp_info,boundary_index).cuda()
        self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)

        self.simu_index=self.tarp.tarp_info.C.clone().detach().cpu().numpy()
        self.simu_jacobi=0
        self.itertimes=0
        self.simu_force=0
        self.simu_force_grad=0
        self.simu_equa_force_grad=0
        self.simu_vertices_grad=0
        self.set_all_forces()

        self.target_image = imageio.imread(params.image).astype('float32') / 255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).cuda().unsqueeze(dim=0)
        
        para_dir=str(tool.get_datetime())+' '+str(self.tarp.tarp_info.C.shape[0])+' '+str(params.balance_cof)\
                 +' '+str(params.learning_rate)+' '+os.path.splitext(os.path.split(params.image)[1])[0]
        self.result_folder=os.path.join(params.output_dir,para_dir)
        if os.path.exists(self.result_folder)==False:
            os.mkdir(self.result_folder)
        self.write_params()
        """ self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        save_image=np.zeros([1,128,128,4])
        save_image[0,46:82,46:82,0]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,1]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,2]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,46:82,46:82,3]=1.0#np.where(self.target_image[44:84,44:84]>0.0000001,1.0,0.0)
        imageio.imsave(os.path.join(self.args.output_dir,'square36.png'),(255*save_image[0]).astype(np.uint8)) """

    def set_all_forces(self):
        forces=self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace)
        self.simu_force=forces.clone().detach().cpu().numpy()[0]
        self.diffsimulator.set_forces(self.simu_force.flatten())
        self.simu_force_grad=self.simu_force
        self.simu_equa_force_grad=self.simu_force
        self.simu_vertices_grad=torch.zeros_like(self.tarp.vertices).clone().detach().cpu().numpy()[0]
       
    def one_iterate(self):
        #compute balance
        if self.newton_flag==True:
            self.diffsimulator.newton()
            print('newton')
        else:
            self.diffsimulator.Opt()
            print('pd')
        if self.itertimes%params.updategl_hz==0:
            self.simu_pos=self.diffsimulator.v.reshape(int(self.diffsimulator.v.size/3),3)
        self.diffsimulator.print_balance_info(params.balance_cof)
        rate=self.diffsimulator.balance_rate
        if rate<params.newton_rate:
            self.newton_flag=True
        else:
            self.newton_flag=False
        if rate>params.balance_cof:
            return
        
        self.newton_flag=False
        self.diffsimulator.compute_jacobi()

        #get shadow image from mesh
        vertices=tool.py_simulation.apply(self.external_force.force_displace,self.diffsimulator)
        self.tarp.vertices=vertices
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        shadow_image=self.rasterizer(mesh)
        image=shadow_image.detach().cpu().numpy()[0].transpose((1,2,0))


        #compute loss using force and shadow image
        self.external_force.record_last_displace()
        barrier_loss=self.external_force()
        shadow_loss=shadow_image_loss(shadow_image,self.target_image)
        loss=barrier_loss+shadow_loss
        
        #update loss figure
        figure_x.append(self.itertimes)
        barrier_loss_cpu=barrier_loss.clone().detach().cpu().numpy()
        shadow_loss_cpu=shadow_loss.clone().detach().cpu().numpy()
        figure_shadowloss.append(shadow_loss_cpu)
        figure_loss.append(barrier_loss_cpu+shadow_loss_cpu)
        plt.clf()
        plt.plot(figure_x, figure_shadowloss,label="shadow image loss",color="blue")#,linestyle=':',linewidth=5,alpha=0.8)
        plt.plot(figure_x, figure_loss,label="total loss",color="red",linestyle=':',linewidth=3)
        plt.legend(loc="upper right")

        #save some results
        if self.itertimes==0:
            imageio.imsave(os.path.join(self.result_folder,'init.png'),(255*image).astype(np.uint8))
        if self.itertimes%params.saveshadow_hz==0:
            imageio.imsave(os.path.join(self.result_folder,'deform_%05d.png'%self.itertimes),(255*image).astype(np.uint8))
        if self.itertimes%params.saveloss_hz==0:
            plt.savefig(os.path.join(self.result_folder, 'loss.png'))
        if self.itertimes%params.saveresult_hz==0:
            self.write_results()
        self.itertimes=self.itertimes+1

        #backward
        if params.use_vertgrad:
            vertices.retain_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        #gui info
        if params.use_forcegrad:
            self.simu_force_grad=-torch.bmm(self.external_force.transform,self.external_force.force_displace.grad)[0].clone().detach().cpu().numpy()
        self.external_force.equivalence_projection()
        self.external_force.linesearch()
        if params.use_adamgrad:
            self.simu_equa_force_grad=torch.bmm(self.external_force.transform,self.external_force.force_displace\
                                            -self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
        if params.use_vertgrad:
            self.simu_vertices_grad=-vertices.grad[0].clone().detach().cpu().numpy()

        self.simu_force=(self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace))[0].clone().detach().cpu().numpy()
        self.diffsimulator.set_forces(self.simu_force.flatten())
        
        #terminate condition
        delta=(self.external_force.force_displace-self.external_force.force_last_displace).clone().detach().cpu().numpy()
        if (delta**2).sum()<params.grad_error or self.itertimes > params.max_iter-params.updategl_hz:
            self.small_gradient=True
            self.write_results()
            print('small gradient or max iter')
            run_time=np.floor(time.time()-self.begin_time)
            hour_time=run_time//3600
            minute_time=(run_time-3600*hour_time)//60
            second_time=run_time-3600*hour_time-60*minute_time
            print (f'run time:{hour_time}hour{minute_time}minute{second_time}second')

    def write_params(self):
        tool.copy_file(os.path.join(params.current_dir,'params.yaml'),self.result_folder)

    def write_results(self):
        self.tarp.get_render_mesh().save_obj(os.path.join(self.result_folder,'result.obj'))
        tool.write_data(self.external_force.force.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'force.txt'))
        #here wu use force_last_displace rather than force_displace because force_displace has been updated
        tool.write_data(torch.bmm(self.external_force.transform,self.external_force.force_last_displace).clone().detach().cpu().numpy(),
                        os.path.join(self.result_folder,'force_displace.txt'))

    def write_jacobi(self,jacobi):
        tool.write_data(jacobi.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'jacobi.txt'))

    def write_leftmat(self,leftmat):
        tool.write_data(leftmat.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'leftmat.txt'))

