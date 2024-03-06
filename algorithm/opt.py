from typing import Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import time

import algorithm.tarp_info as TI
import soft_renderer as sr
import algorithm.tool as tool
from algorithm.balance_solver import *

import algorithm.force as force
import algorithm.geometry as geometry
import algorithm.shadow as shadow
from line_profiler import line_profiler
import random

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
        
        #calc boundary information
        boundary_index,boundary_weight=tool.get_mesh_boundary(params.template_mesh)
        self.tarp = TI.Tarp(params)
        stick_index=self.tarp.tarp_info.C
        for i,value in enumerate(self.tarp.tarp_info.C):
            stick_index[i]=torch.nonzero(torch.eq(boundary_index,value))

        #init force class
        self.force=force.Force(self.tarp.vertices,self.tarp.tarp_info,boundary_index,boundary_weight,stick_index).cuda()

        #init balance_solver
        self.balance_solver=balance_solver()
        self.balance_solver.set_info(
                        self.tarp.faces[0].clone().detach().cpu().numpy(),
                        self.tarp.vertices[0].clone().detach().cpu().numpy().flatten(),
                        boundary_index.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.k.clone().detach().cpu().numpy(),
                        0.1,
                        self.tarp.tarp_info.mass.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.CI.clone().detach().cpu().numpy()
                        )
        self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_cof,params.newton_rate)
        self.balance_solver.set_forces(self.force.now_force().clone().detach().cpu().numpy().flatten())
        self.balance_solver.compute_csr_right()
        self.jacobi_solver=tool.jacobi_solver()
        self.jacobi_solver.compute_right(self.balance_solver.jacobiright)
        self.middleright=0
        #self.jacobiright=tool.compute_jacobiright(self.balance_solver.jacobiright)

        #calc cubic sampling
        self.balance_solver.set_sampling_parameter(params.loc_rad,params.dir_rad,params.start_rad,params.end_rad,params.sample_num)
        self.balance_solver.cubic_sampling()
        self.tarp.set_sampling(torch.from_numpy(self.balance_solver.sampling_lambda).cuda())

        #init other class
        self.geometry=geometry.Geometry(self.tarp)
        self.shadow=shadow.Shadow(self.tarp)
        self.optimizer = torch.optim.Adam(self.force.parameters(), lr=params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)
        self.loss_drawer=tool.LossDrawer(params)

        #others
        self.small_gradient=False
        self.openFnorm1Cons=params.fnorm1_cons==2
        self.stop=False
        self.itertimes=0
        self.simu_index=boundary_index.clone().detach().cpu().numpy()
        self.damping_rate=np.power(params.balance_min/params.balance_cof,params.accurate_hz/params.max_iter)
        self.now_balance=params.balance_cof
        self.prev_time=time.perf_counter()


    def one_iterate(self):
        print('\n\nitertimes:',self.itertimes)

        #compute balance
        if self.itertimes%params.accurate_hz==0:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_min,params.newton_rate)
            self.now_balance=self.now_balance*self.damping_rate
        else:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,self.now_balance,params.newton_rate)
        self.balance_solver.compute_balance()
        if self.balance_solver.balance_result>self.now_balance:
            return
        
        from_forward=time.perf_counter()
        #forward
        if self.itertimes%params.update_w_hz==0 and self.itertimes > params.update_start-2:
            if params.fnorm1_cons==2:
                self.force.update_weight()
            elif params.fnorm1_cons==1:
                if self.openFnorm1Cons==False:
                    self.force.weight=torch.tensor([1.0],device=torch.device("cuda"))
                    self.openFnorm1Cons=True
                self.force.update_weight()

        start=time.perf_counter()
        if self.balance_solver.x_invariant==False:
            self.balance_solver.compute_csr()
            self.jacobi_solver.lu(self.balance_solver.jacobirow,self.balance_solver.jacobicol,self.balance_solver.jacobival)
            self.middlejacobi=self.jacobi_solver.solve()
        print('solve jacobi',time.perf_counter()-start)
        start=time.perf_counter()
            

        #start=time.perf_counter()
        #self.balance_solver.compute_csr()
        #print('compute_csr',time.perf_counter()-start)
        #start=time.perf_counter()
        #self.balance_solver.compute_csr_right()
        #print('compute_csr_right',time.perf_counter()-start)
        #start=time.perf_counter()
        #middlejacobi=tool.compute_jacobi(self.balance_solver.jacobirow,self.balance_solver.jacobicol,self.balance_solver.jacobival,
        #                    self.jacobiright)
        #print('compute_middlejacobi',time.perf_counter()-start)
        #start=time.perf_counter()
        vertices=tool.py_simulation.apply(self.force.force_displace,self.balance_solver,self.middlejacobi)
        print('compute_vertices',time.perf_counter()-start)
        start=time.perf_counter()

        self.tarp.vertices=vertices
        shadow_loss=self.shadow.loss_evaluation()
        print('compute_shadow',time.perf_counter()-start)
        start=time.perf_counter()
        geometry_loss=self.geometry.loss_evaluation()
        print('compute_geometry',time.perf_counter()-start)
        start=time.perf_counter()
        self.force.record_last_displace()
        barrier_loss=self.force()
        print('compute_barrier',time.perf_counter()-start)
        start=time.perf_counter()
        loss=barrier_loss+geometry_loss+shadow_loss
        print('compute_loss',time.perf_counter()-start)
        start=time.perf_counter()
        
        #update loss figure
        self.loss_drawer.update(self.itertimes,self.force,self.geometry,self.shadow)
        print('draw',time.perf_counter()-start)
        start=time.perf_counter()

        #save some results
        self.shadow.save_image(self.result_folder,self.itertimes)
        self.loss_drawer.save_loss(self.result_folder,self.itertimes)
        if self.itertimes%params.saveresult_hz==0:
            self.write_results()
        print('save results',time.perf_counter()-start)
        start=time.perf_counter()
        
        #backward
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        print('compute_grad',time.perf_counter()-start)
        start=time.perf_counter()

        #proximal processing
        if params.enable_prox:
            self.external_force.prox_processing()

        self.force.linesearch(self.tarp.vertices)
        print('compute_linesearch',time.perf_counter()-start)
        start=time.perf_counter()
        self.balance_solver.set_forces(self.force.now_force().clone().detach().cpu().numpy().flatten())
        
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

        print('end',time.perf_counter()-start)
        print('one iteration done')
        print('fff',time.perf_counter()-from_forward)
        print('one iteration time',time.perf_counter()-self.prev_time)
        self.prev_time=time.perf_counter()
        self.itertimes=self.itertimes+1

    def write_params(self):
        tool.copy_file(os.path.join(params.current_dir,'params.yaml'),self.result_folder)

    def write_results(self):
        self.tarp.get_mesh().save_obj(os.path.join(self.result_folder,'result.obj'))
        primal_force=self.force.force[0].clone().detach().cpu().numpy()
        tool.write_data(primal_force,os.path.join(self.result_folder,'force.txt'))
        #here I use force_last_displace rather than force_displace because force_displace has been updated
        delta_force=torch.bmm(self.force.transform,self.force.force_last_displace)[0].clone().detach().cpu().numpy()
        tool.write_data(delta_force,os.path.join(self.result_folder,'force_displace.txt'))
        tool.write_data(primal_force+delta_force,os.path.join(self.result_folder,'last_force.txt'))
        tool.write_data(self.simu_index,os.path.join(self.result_folder,'index.txt'),x3=False)
        
        run_time=np.floor(time.time()-self.begin_time)
        hour_time=run_time//3600
        minute_time=(run_time-3600*hour_time)//60
        second_time=run_time-3600*hour_time-60*minute_time
        tool.write_data(np.array([self.itertimes,hour_time,minute_time,second_time]),os.path.join(self.result_folder,'time.txt'),x3=False)

    def write_jacobi(self,jacobi):
        tool.write_data(jacobi.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'jacobi.txt'))

    def write_leftmat(self,leftmat):
        tool.write_data(leftmat.clone().detach().cpu().numpy(),os.path.join(self.result_folder,'leftmat.txt'))

