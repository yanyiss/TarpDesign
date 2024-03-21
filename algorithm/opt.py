from typing import Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import time

import algorithm.tarp_info as TI
import algorithm.tool as tool
from algorithm.balance_solver import *

import algorithm.force as force
import algorithm.geometry as geometry
import algorithm.shadow as shadow

figure_x=[]
figure_fmax_loss=[]
figure_fdir_loss=[]
figure_fnorm1_loss=[]
figure_shadow_loss=[]
figure_total_loss=[]
plt.ion()

params=TI.tarp_params()

class deform():
    def __init__(self):
        self.begin_time=time.time()
        #init folder and params notes
        os.makedirs(params.output_dir, exist_ok=True)
        para_dir=str(tool.get_datetime())
        self.result_folder=os.path.join(params.output_dir,para_dir)
        if os.path.exists(self.result_folder)==False:
            os.mkdir(self.result_folder)
            os.mkdir(os.path.join(self.result_folder,'shadow'))
        self.write_params()
        tool.write_readme(' ',os.path.join(self.result_folder,'readme.txt'))
        
        self.tarp = TI.Tarp(params)

        #init force class
        self.force=force.Force(self.tarp.vertices,self.tarp.tarp_info).cuda()

        #init balance_solver
        spring,stiffness=tool.get_spring(params.template_mesh,self.tarp.tarp_info.k.cpu(),params.use_bending)
        self.balance_solver=balance_solver()
        self.balance_solver.set_info(
            spring,
            stiffness,
            self.tarp.vertices[0].clone().detach().cpu().numpy().flatten(),
            self.tarp.tarp_info.rope_index_in_mesh.clone().detach().cpu().numpy(),
            self.tarp.tarp_info.stick_index_in_mesh.clone().detach().cpu().numpy(),
            0.1,
            self.tarp.tarp_info.mass.clone().detach().cpu().numpy(),
            self.tarp.tarp_info.CI.clone().detach().cpu().numpy()
        )
        self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_cof,params.newton_rate)
        self.force.compute_now_forces()
        self.balance_solver.set_forces(self.force.now_force.clone().detach().cpu().numpy().flatten())
        self.balance_solver.compute_csr_right()
        self.jacobi_solver=tool.linear_solver()
        self.jacobi_solver.compute_right(self.balance_solver.jacobiright)
        self.middleright=0
        #self.jacobiright=tool.compute_jacobiright(self.balance_solver.jacobiright)

        #calc cubic sampling
        self.balance_solver.set_sampling_parameter(params.loc_rad,params.dir_rad,params.start_rad,params.end_rad,params.sample_num)
        self.balance_solver.cubic_sampling()
        self.tarp.set_sampling(torch.from_numpy(self.balance_solver.sampling_lambda).cuda())

        #init other class
        self.geometry=geometry.Geometry(self.tarp,self.force)
        self.shadow=shadow.Shadow(self.tarp)
        #self.optimizer=torch.optim.SGD(self.force.parameters(),lr=params.learning_rate)
        self.optimizer = torch.optim.Adam(self.force.parameters(), lr=params.learning_rate,betas=(params.beta1,params.beta2))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)
        self.loss_drawer=tool.LossDrawer(params)

        #others
        self.loss=torch.tensor([1.0e8],device=torch.device("cuda"))
        self.small_gradient=False
        self.stop=False
        self.itertimes=0
        self.simu_index=self.tarp.tarp_info.rope_index_in_mesh.clone().detach().cpu().numpy()
        self.damping_rate=np.power(params.balance_min/params.balance_cof,params.accurate_hz/params.max_iter)
        self.now_balance=params.balance_cof
        self.prev_time=time.perf_counter()


        self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_min,params.newton_rate)

    def get_loss(self,compute_jacobi=True,for_backward=True):
        #compute balance
        if self.itertimes%params.accurate_hz==0:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_min,params.newton_rate)
            #self.now_balance=self.now_balance*self.damping_rate
            self.now_balance=params.balance_cof
        else:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,self.now_balance,params.newton_rate)
            self.now_balance=params.balance_cof

        while True:
            self.balance_solver.compute_balance()
            if self.balance_solver.balance_result<self.now_balance:
                break
        
        #forward
        if self.itertimes%params.update_w_hz==0 and self.itertimes > params.update_start-2:
            self.force.update_weight()
            #self.force.restart_sparse()

        if self.balance_solver.x_invariant==False and compute_jacobi:
        #if compute_jacobi:
            self.balance_solver.compute_csr()
            self.jacobi_solver.lu(self.balance_solver.jacobirow,self.balance_solver.jacobicol,self.balance_solver.jacobival)
            self.middlejacobi=self.jacobi_solver.solve().unsqueeze(dim=0)
            
        self.optimizer.zero_grad()
        self.tarp.vertices=tool.py_simulation.apply(self.force.force_displace,self.balance_solver,self.middlejacobi)
        shadow_loss=self.shadow.loss_evaluation()
        geometry_loss=self.geometry.loss_evaluation()
        if for_backward:
            barrier_loss=self.force()
        else:
            barrier_loss=self.force.loss_evaluation()
        return barrier_loss+geometry_loss+shadow_loss

    def one_iterate(self):
        print('\n\nitertimes:',self.itertimes)

        
        self.force.compute_now_forces()
        self.balance_solver.set_forces(self.force.now_force.clone().detach().cpu().numpy().flatten())
        #loss=barrier_loss
        self.force.record_last_displace()
        self.loss=self.get_loss()
        
        #update loss figure
        self.loss_drawer.update(self.itertimes,self.force,self.geometry,self.shadow)

        #save some results
        self.shadow.save_image(self.result_folder,self.itertimes)
        self.loss_drawer.save_loss(self.result_folder,self.itertimes)
        if self.itertimes%params.saveresult_hz==0:
            self.write_results()
        
        #backward
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if params.use_proximal:
            self.proximal_processing()
        if params.rope_cons or params.fmax_cons or params.fdir_cons:
            self.geometry.linesearch()
        
        #forward
        if self.itertimes%params.update_w_hz==0 and self.itertimes > params.update_start-2:
            #self.force.update_weight()
            self.force.restart_sparse()
            
        #terminate condition
        delta=(self.force.force_displace-self.force.force_last_displace).clone().detach().cpu().numpy()
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
        print('one iteration time',time.perf_counter()-self.prev_time)
        self.prev_time=time.perf_counter()
        self.itertimes=self.itertimes+1

    def prox(self,value):
        mag=torch.sqrt((value**2).sum(dim=2,keepdim=True)).repeat(1,1,3)
        alpha_lambda=self.force.l1_alpha*params.fnorm1_weight
        return torch.where(mag>alpha_lambda,(1.0-alpha_lambda/mag)*value,0.0)
    
    def proximal_processing(self):
        if params.fnorm1_cons==0 or params.use_proximal==0:
            return
        force=self.force
        df=force.force+force.force_last_displace-\
            self.prox(force.force+force.force_last_displace-force.l1_alpha*(force.force_last_displace-force.force_displace))
        force.force_displace.data=force.force_last_displace-df
        return
        # force.compute_now_forces()
        # compare_loss=self.get_loss(False,False)
        # itr=0
        # while True:
        #     force.force_displace.data=force.force_last_displace-df
        #     force.compute_now_forces()
        #     search_loss=self.get_loss(False,False)
        #     print(search_loss,compare_loss,search_loss-compare_loss,df.norm())
        #     if search_loss<compare_loss:
        #         break
        #     df=df*params.l1_beta
        #     itr=itr+1
        #     if itr>20:
        #         df=df*0.0
        #         break


    def write_params(self):
        tool.copy_file(os.path.join(params.current_dir,'params.yaml'),self.result_folder)

    def write_results(self):
        self.tarp.get_mesh().save_obj(os.path.join(self.result_folder,'result.obj'))
        tool.write_data(self.force.force_last_displace[0].clone().detach().cpu().numpy(),os.path.join(self.result_folder,'force_displace.txt'))
        tool.write_data(self.force.force[0].clone().detach().cpu().numpy(),os.path.join(self.result_folder,'force.txt'))
        tool.write_data(self.force.now_force[0].clone().detach().cpu().numpy(),os.path.join(self.result_folder,'last_force.txt'))
        tool.write_data(self.simu_index,os.path.join(self.result_folder,'index.txt'),x3=False)
        
        run_time=np.floor(time.time()-self.begin_time)
        hour_time=run_time//3600
        minute_time=(run_time-3600*hour_time)//60
        second_time=run_time-3600*hour_time-60*minute_time
        tool.write_data(np.array([self.itertimes,hour_time,minute_time,second_time]),os.path.join(self.result_folder,'time.txt'),x3=False)
        tool.write_data(torch.tensor([self.force.fmax_loss,self.force.fdir_loss,self.force.fnorm1_loss,self.geometry.geometry_loss,\
                                  self.shadow.shadow_loss]).clone().detach().cpu().numpy(),os.path.join(self.result_folder,'loss.txt'),x3=False)


