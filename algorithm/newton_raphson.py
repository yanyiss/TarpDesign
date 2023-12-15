from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import time

import algorithm.tarp_info as TI
import algorithm.tool as tool
import algorithm.meshrender as meshrender
import algorithm.ropeforce as ropeforce



params=TI.tarp_params()

class newton_raphson(nn.Module):
    def __init__(self):
        super(newton_raphson,self).__init__()

        os.makedirs(params.output_dir, exist_ok=True)
        self.tarp=TI.Tarp(params)
        self.meshrender=meshrender.MeshRender(self.tarp,params)
        self.boundary_index=tool.get_mesh_boundary(params.template_mesh)
        self.ropeforce=ropeforce.RopeForce(self.tarp.vertices,self.tarp.tarp_info,self.boundary_index,params)

        vf=torch.cat((self.tarp.vertices,self.ropeforce.force),dim=1)
        """ self.register_parameter("vf_displace",nn.Parameter(torch.zeros(vf.shape[0],vf.shape[1]-1,vf.shape[2]).cuda()))
        self.register_buffer("vf_last_displace",nn.Parameter(torch.zeros_like(self.vf_displace)).cuda()) """
        self.register_parameter("vf_displace",nn.Parameter(torch.zeros_like(vf).cuda()))
        self.register_buffer("vf_last_displace",nn.Parameter(torch.zeros_like(vf)).cuda())



        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)

        self.simu_index=self.tarp.tarp_info.C.clone().detach().cpu().numpy()
        self.simu_jacobi=0
        self.itertimes=0
        self.simu_force=0
        self.simu_force_grad=0
        self.simu_equa_force_grad=0
        self.simu_vertices_grad=0
        self.set_all_forces()
        
        para_dir=str(tool.get_datetime())+' '+str(self.tarp.tarp_info.C.shape[0])+' '+str(params.balance_cof)\
                 +' '+str(params.learning_rate)+' '+os.path.splitext(os.path.split(params.image)[1])[0]
        self.result_folder=os.path.join(params.output_dir,para_dir)
        if os.path.exists(self.result_folder)==False:
            os.mkdir(self.result_folder)
        self.write_params()
        tool.write_readme(' ',os.path.join(self.result_folder,'readme.txt'))

        self.adj,self.len=tool.get_adj(params.template_mesh)
        self.v2f_id=tool.get_v2f_id(self.meshrender.nv,self.boundary_index)
        self.loss_drawer=tool.LossDrawer()


    def set_all_forces(self):
        return
        """ forces=self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace)
        self.simu_force=forces.clone().detach().cpu().numpy()[0]
        self.balance_solver.set_forces(self.simu_force.flatten())
        self.simu_force_grad=self.simu_force
        self.simu_equa_force_grad=self.simu_force
        self.simu_vertices_grad=torch.zeros_like(self.tarp.vertices).clone().detach().cpu().numpy()[0] """

    def one_iterate(self):
        vf=torch.zeros_like(self.vf_displace).cuda()
        vf[:,0:self.meshrender.nv,:]=self.meshrender.vertices+self.vf_displace[:,0:self.meshrender.nv,:]
        vf[:,-self.ropeforce.nf:,:]=self.ropeforce.force+self.vf_displace[:,-self.ropeforce.nf:,:]

        shadow_loss=self.meshrender.loss_evaluation(vf)
        force_loss=self.ropeforce.loss_evaluation(vf)
        balance_loss=tool.balance_energy.apply(vf,self.meshrender.nv,self.ropeforce.nf,self.adj,self.len,self.v2f_id)
        total_loss=shadow_loss+force_loss+balance_loss
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.loss_drawer.update(self.itertimes,self.ropeforce,self.meshrender)
        if self.itertimes%params.saveloss_hz==0:
            self.loss_drawer.save_loss(self.result_folder)
        if self.itertimes%params.saveshadow_hz==0:
            self.meshrender.save_image(self.result_folder,self.itertimes)


        """ #loss=parallel_energy.energy_forward(self.vf_displace,self.nv)
        vf=torch.zeros_like(self.vf_displace).cuda()
        vf[:,0:self.meshrender.nv,:]=self.meshrender.vertices+self.vf_displace[:,0:self.meshrender.nv,:]
        vf[:,-self.ropeforce.nf:,:]=self.ropeforce.force+self.vf_displace[:,-self.ropeforce.nf:,:]
        adj,len=tool.get_adj(params.template_mesh)

        v2f_id=torch.IntTensor(self.meshrender.nv).cuda()
        for i in range(self.meshrender.nv):
            v2f_id[i]=-1
        it=0
        for i in range(self.boundary_index.shape[0]):
            v2f_id[self.boundary_index[i]]=it
            it=it+1
        
        start=time.perf_counter()
        balance_value=torch.zeros(self.meshrender.nv*3).cuda()
        parallel_energy.energy_forward(vf,self.meshrender.nv,self.ropeforce.nf,adj,len,v2f_id,balance_value)
        print('loss',torch.sum(balance_value**2))

        balance_gradient=torch.zeros((self.meshrender.nv+self.ropeforce.nf)*3).cuda()
        parallel_energy.energy_backward(vf,self.meshrender.nv,self.ropeforce.nf,adj,len,v2f_id,balance_value,balance_gradient)
        print('gradient',balance_gradient)
        print('time',time.perf_counter()-start)

        exit(0) """


        """ start=time.perf_counter()
        total_loss=self.compute_loss(self.vf_displace)
        grad=torch.autograd.grad(outputs=total_loss,inputs=self.vf_displace,grad_outputs=torch.ones_like(total_loss),retain_graph=True,create_graph=True)[0]
        hessian=torch.autograd.functional.hessian(self.compute_loss,self.vf_displace,vectorize=True)
        grad2=torch.zeros((grad.shape[1],grad.shape[2],1,grad.shape[1],grad.shape[2]))
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                grad2[i,j,:,:,:]=torch.autograd.grad(outputs=grad[:,i,j],inputs=self.vf_displace,grad_outputs=torch.ones_like(grad[:,i,j]),retain_graph=True)[0]
        print(time.perf_counter()-start) """

        #if self.itertimes%params.updategl_hz==0:
        #self.simu_pos=self.balance_solver.v.reshape(int(self.balance_solver.v.size/3),3)

        
        if self.itertimes%params.update_w_hz==0:
            #self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=params.learning_rate)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)
            self.ropeforce.update_weight(self.vf_displace)
        
        self.itertimes=self.itertimes+1
        #terminate condition
        """ delta=(self.external_force.force_displace-self.external_force.force_last_displace).clone().detach().cpu().numpy()
        if (delta**2).sum()<params.grad_error*0 or self.itertimes > params.max_iter-params.updategl_hz:
            self.small_gradient=True
            self.write_results()
            print('small gradient or max iter')
            run_time=np.floor(time.time()-self.begin_time)
            hour_time=run_time//3600
            minute_time=(run_time-3600*hour_time)//60
            second_time=run_time-3600*hour_time-60*minute_time
            print (f'run time:{hour_time}hour{minute_time}minute{second_time}second')
            self.stop=True
            #tool.write_data(np.array([hour_time,])) """

    def write_params(self):
        tool.copy_file(os.path.join(params.current_dir,'params.yaml'),self.result_folder)

    def write_results(self):
        self.tarp.get_render_mesh().save_obj(os.path.join(self.result_folder,'result.obj'))
        primal_force=self.external_force.force[0].clone().detach().cpu().numpy()
        tool.write_data(primal_force,os.path.join(self.result_folder,'force.txt'))
        #here I use force_last_displace rather than force_displace because force_displace has been updated
        delta_force=torch.bmm(self.external_force.transform,self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
        tool.write_data(delta_force,os.path.join(self.result_folder,'force_displace.txt'))
        tool.write_data(primal_force+delta_force,os.path.join(self.result_folder,'last_force.txt'))
        
        run_time=np.floor(time.time()-self.begin_time)
        hour_time=run_time//3600
        minute_time=(run_time-3600*hour_time)//60
        second_time=run_time-3600*hour_time-60*minute_time
        tool.write_data(np.array([self.itertimes,hour_time,minute_time,second_time]),os.path.join(self.result_folder,'time.txt'),x3=False)

    def write_jacobi(self,jacobi):
        tool.write_data(jacobi.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'jacobi.txt'))

    def write_leftmat(self,leftmat):
        tool.write_data(leftmat.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'leftmat.txt'))

