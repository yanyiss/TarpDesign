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
from algorithm.balance_solver import *



params=TI.tarp_params()

class newton_raphson(nn.Module):
    def __init__(self):
        super(newton_raphson,self).__init__()

        os.makedirs(params.output_dir, exist_ok=True)
        self.tarp=TI.Tarp(params)
        self.meshrender=meshrender.MeshRender(self.tarp,params)
        boundary_index=tool.get_mesh_boundary(params.template_mesh)
        self.ropeforce=ropeforce.RopeForce(self.tarp.vertices,self.tarp.tarp_info,boundary_index,params)

        vf=torch.cat((self.tarp.vertices,self.ropeforce.force),dim=1)
        self.register_parameter("vf_displace",nn.Parameter(torch.zeros(vf.shape[0],vf.shape[1]-1,vf.shape[2]).cuda()))
        self.register_buffer("vf_last_displace",nn.Parameter(torch.zeros_like(self.vf_displace)).cuda())



        #self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=params.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)

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

        self.prev_time=time.perf_counter()


    def set_all_forces(self):
        return
        """ forces=self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace)
        self.simu_force=forces.clone().detach().cpu().numpy()[0]
        self.balance_solver.set_forces(self.simu_force.flatten())
        self.simu_force_grad=self.simu_force
        self.simu_equa_force_grad=self.simu_force
        self.simu_vertices_grad=torch.zeros_like(self.tarp.vertices).clone().detach().cpu().numpy()[0] """

    def compute_loss(self,x):
        shadow_loss=self.meshrender.loss_evaluation(x)
        force_loss=self.ropeforce.loss_evaluation(x)
        total_loss=shadow_loss+force_loss
        return total_loss

    def one_iterate(self):
        
        start=time.perf_counter()
        total_loss=self.compute_loss(self.vf_displace)
        grad=torch.autograd.grad(outputs=total_loss,inputs=self.vf_displace,grad_outputs=torch.ones_like(total_loss),retain_graph=True,create_graph=True)[0]
        hessian=torch.autograd.functional.hessian(self.compute_loss,self.vf_displace,vectorize=True)
        grad2=torch.zeros((grad.shape[1],grad.shape[2],1,grad.shape[1],grad.shape[2]))
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                grad2[i,j,:,:,:]=torch.autograd.grad(outputs=grad[:,i,j],inputs=self.vf_displace,grad_outputs=torch.ones_like(grad[:,i,j]),retain_graph=True)[0]
        print(time.perf_counter()-start)

        #if self.itertimes%params.updategl_hz==0:
        self.simu_pos=self.balance_solver.v.reshape(int(self.balance_solver.v.size/3),3)

        
        start=time.perf_counter()
        self.balance_solver.compute_balance()
        print('balance',time.perf_counter()-start)
        if self.balance_solver.balance_result>params.balance_cof:
            return
        print('compute balance done')
        ptt=time.perf_counter()
        start=time.perf_counter()
        if self.itertimes%params.update_w_hz==0:
            #self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=params.learning_rate)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)
            self.external_force.update_weight()
        print('vertice',time.perf_counter()-start)
        start=time.perf_counter()
        self.balance_solver.compute_jacobi()

        #get shadow image from mesh
        vertices=tool.py_simulation.apply(self.external_force.force_displace,self.balance_solver)
        print('vertice',time.perf_counter()-start)
        start=time.perf_counter()
        self.tarp.vertices=vertices
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        shadow_image=self.rasterizer(mesh)
        image=shadow_image.detach().cpu().numpy()[0].transpose((1,2,0))

        print('image',time.perf_counter()-start)
        start=time.perf_counter()


        #compute loss using force and shadow image
        self.external_force.record_last_displace()
        barrier_loss=self.external_force()
        print('barrier',time.perf_counter()-start)
        start=time.perf_counter()
        loss=barrier_loss+shadow_loss
        print('vertice',time.perf_counter()-start)
        print('vesfs',time.perf_counter()-ptt)
        print('compute loss done')
        
        #update loss figure
        
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
        #exit(0)

        #backward
        if params.use_vertgrad:
            vertices.retain_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        print('backward done')

        #proximal processing
        if params.enable_prox:
            self.external_force.prox_processing()

        #gui info
        if params.use_forcegrad:
            self.simu_force_grad=-torch.bmm(self.external_force.transform,self.external_force.force_displace.grad)[0].clone().detach().cpu().numpy()
        self.external_force.linesearch(self.tarp.vertices)
        if params.use_adamgrad:
            self.simu_equa_force_grad=torch.bmm(self.external_force.transform,self.external_force.force_displace\
                                            -self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
        if params.use_vertgrad:
            self.simu_vertices_grad=-vertices.grad[0].clone().detach().cpu().numpy()

        self.simu_force=(self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace))[0].clone().detach().cpu().numpy()
        self.balance_solver.set_forces(self.simu_force.flatten())
        
        #terminate condition
        delta=(self.external_force.force_displace-self.external_force.force_last_displace).clone().detach().cpu().numpy()
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
            #tool.write_data(np.array([hour_time,]))

        print('one iteration done')
        print('\n\n')
        print(self.itertimes)
        print(time.perf_counter()-self.prev_time)
        self.prev_time=time.perf_counter()

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

