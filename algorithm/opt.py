from typing import Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio
import time

import algorithm.tarp_info as TI
import soft_renderer as sr
import algorithm.tool as tool
import algorithm.external_force as external_force
from algorithm.balance_solver import *


figure_x=[]
figure_fmax_loss=[]
figure_fdir_loss=[]
figure_fnorm1_loss=[]
figure_shadow_loss=[]
figure_total_loss=[]
plt.ion()

params=TI.tarp_params()
eye_value=[0.0,0.0,-1.0]

def shadow_image_loss(shadow_image,target_image):
    if params.loss_type=='image':
        return ((shadow_image-target_image)**2).sum()/(params.image_size*params.image_size)
    elif params.loss_type=='area':
        return -torch.sum(shadow_image)/(params.image_size*params.image_size)
    else:
        print('shadow image loss error')
        exit(0)
        return torch.tensor([0]).cuda()

class deform():
    def __init__(self):
        self.begin_time=time.time()
        
        os.makedirs(params.output_dir, exist_ok=True)
        para_dir=str(tool.get_datetime())#+' '+str(self.tarp.tarp_info.C.shape[0])+' '+str(params.balance_cof)\
                 #+' '+str(params.learning_rate)+' '+os.path.splitext(os.path.split(params.image)[1])[0]
        #para_dir='temp'
        self.result_folder=os.path.join(params.output_dir,para_dir)
        if os.path.exists(self.result_folder)==False:
            os.mkdir(self.result_folder)
            os.mkdir(os.path.join(self.result_folder,'shadow'))
        self.write_params()
        tool.write_readme(' ',os.path.join(self.result_folder,'readme.txt'))

        if params.use_denseInfo:
            params.template_mesh=params.result_mesh


        self.transform=sr.LookAt(perspective=False,viewing_scale=params.view_scale,eye=eye_value)
        self.lighting=sr.Lighting(directions=[0,0,1.0])
        self.rasterizer=sr.SoftRasterizer(image_size=params.image_size,sigma_val=params.sigma_value,gamma_val=params.gamma_value,aggr_func_rgb='hard')

        boundary_index,boundary_weight=tool.get_mesh_boundary(params.template_mesh)
        self.tarp = TI.Tarp(params)

        """ mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        shadow_image=self.rasterizer(mesh)
        image=shadow_image.detach().cpu().numpy()[0].transpose((1,2,0))
        para_dir=str(tool.get_datetime())+' '+str(self.tarp.tarp_info.C.shape[0])+' '+str(params.balance_cof)\
                 +' '+str(params.learning_rate)+' '+os.path.splitext(os.path.split(params.image)[1])[0]
        #para_dir='temp'
        self.result_folder=os.path.join(params.output_dir,para_dir)
        if os.path.exists(self.result_folder)==False:
            os.mkdir(self.result_folder)
            os.mkdir(os.path.join(self.result_folder,'shadow'))
        imageio.imsave(os.path.join(self.result_folder,'init.png'),(255*image).astype(np.uint8))
        exit(0) """

        self.tarp.tarp_info.C=boundary_index
        self.simu_pos=self.tarp.vertices[0].clone().detach().cpu().numpy()
        self.balance_solver=balance_solver()
        self.balance_solver.set_info(
                        self.tarp.faces[0].clone().detach().cpu().numpy(),
                        self.simu_pos.flatten(),
                        self.tarp.tarp_info.C.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.k.clone().detach().cpu().numpy(),
                        0.1,
                        self.tarp.tarp_info.mass.clone().detach().cpu().numpy(),
                        self.tarp.tarp_info.CI.clone().detach().cpu().numpy()
                        )
        self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_cof,params.newton_rate)
        self.balance_solver.set_sampling_parameter(params.loc_rad,params.dir_rad,params.start_rad,params.end_rad,params.sample_num)
        self.balance_solver.cubic_sampling()
        self.sampling_lambda=torch.from_numpy(self.balance_solver.sampling_lambda).cuda()

        #self.newton_flag=False
        self.small_gradient=False
        self.stop=False

        self.external_force=external_force.ExternalForce(self.tarp.vertices,self.tarp.tarp_info,boundary_index,boundary_weight).cuda()
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

        self.balance_solver.compute_csr_right()
        self.jacobiright=tool.compute_jacobiright(self.balance_solver.jacobiright)

        self.target_image = imageio.imread(params.image).astype('float32') / 255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).cuda().unsqueeze(dim=0)
        
        self.pd_time=0
        self.newton_time=0
        self.prev_time=time.perf_counter()

        #self.tarp.vertices[:,:,0]=self.tarp.vertices[:,:,0]-self.tarp.vertices[:,326,0]
        #self.tarp.vertices[:,:,1]=self.tarp.vertices[:,:,1]-self.tarp.vertices[:,326,1]
        """ tr=self.tarp.faces[:,:,0].clone()
        self.tarp.faces[:,:,0]=self.tarp.faces[:,:,1]
        self.tarp.faces[:,:,1]=tr
        self.tarp.get_render_mesh().save_obj(os.path.join(self.result_folder,'result.obj')) """

        #self.target_image = imageio.imread(self.args.image).astype('float32') / 255.
        """ save_image=np.zeros([1,128,128,4])
        save_image[0,40:88,40:88,0]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,40:88,40:88,1]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,40:88,40:88,2]=0.5#np.where(self.target_image[44:84,44:84]>0.0000001,0.5,0.0)
        save_image[0,40:88,40:88,3]=1.0#np.where(self.target_image[44:84,44:84]>0.0000001,1.0,0.0)
        for i in range(40,88):
            si=np.floor(0.25*(i-40)).astype(np.int64)
            for j in range(40,40+si):
                for k in range(0,4):
                    save_image[0,i,j,k]=0.0
        imageio.imsave(os.path.join(self.result_folder,'trapezoid.png'),(255*save_image[0]).astype(np.uint8)) """

    def set_all_forces(self):
        forces=self.external_force.force+torch.bmm(self.external_force.transform,self.external_force.force_displace)
        self.simu_force=forces.clone().detach().cpu().numpy()[0]
        self.balance_solver.set_forces(self.simu_force.flatten())
        self.simu_force_grad=self.simu_force
        self.simu_equa_force_grad=self.simu_force
        self.simu_vertices_grad=torch.zeros_like(self.tarp.vertices).clone().detach().cpu().numpy()[0]

    def one_iterate(self):
        print('\n\nitertimes:',self.itertimes)
        if self.itertimes%100==0:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_cof*1e-9,params.newton_rate)
        else:
            self.balance_solver.set_compute_balance_parameter(params.updategl_hz,params.balance_cof,params.newton_rate)

        #compute balance
        """ if self.newton_flag==True:
            self.balance_solver.newton()
            print('newton')
        else:
            self.balance_solver.Opt()
            print('pd')
        if self.itertimes%params.updategl_hz==0:
            self.simu_pos=self.balance_solver.v.reshape(int(self.balance_solver.v.size/3),3)
        self.balance_solver.print_balance_info(params.balance_cof)
        rate=self.balance_solver.balance_rate
        if rate<params.newton_rate:
            self.newton_flag=True
        else:
            self.newton_flag=False
        if rate>params.balance_cof:
            return
        
        self.newton_flag=False """
        #if self.itertimes%params.updategl_hz==0:
        self.simu_pos=self.balance_solver.v.reshape(int(self.balance_solver.v.size/3),3)

        """ self.balance_solver.print_balance_info(params.balance_cof)
        rate=self.balance_solver.balance_rate

        if rate>params.balance_cof:
            if rate<params.newton_rate:
                self.balance_solver.newton()
            else:
                self.balance_solver.Opt()
            return """
        
        start=time.perf_counter()
        self.balance_solver.compute_balance()
        print('compute balance',time.perf_counter()-start)
        if self.balance_solver.balance_result>params.balance_cof:
            return
        print('compute balance done')
        ptt=time.perf_counter()
        if self.itertimes%params.update_w_hz==0 and self.itertimes > params.update_start-2:
            #self.optimizer = torch.optim.Adam(self.external_force.parameters(), lr=params.learning_rate)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=params.step_size,gamma=params.decay_gamma)
            self.external_force.update_weight()
        start=time.perf_counter()




        """ self.balance_solver.compute_jacobi()
        exit(0) """

        self.balance_solver.compute_csr()
        self.balance_solver.compute_csr_right()
        #self.balance_solver.jacobi=tool.compute_jacobi(self.balance_solver.jacobirow,self.balance_solver.jacobicol,self.balance_solver.jacobival,
        #                   self.jacobiright)
        middlejacobi=tool.compute_jacobi(self.balance_solver.jacobirow,self.balance_solver.jacobicol,self.balance_solver.jacobival,
                            self.jacobiright)

        print('compute F2X jacobi',time.perf_counter()-start)
        start=time.perf_counter()

        #get shadow image from mesh
        vertices=tool.py_simulation.apply(self.external_force.force_displace,self.balance_solver,middlejacobi)
        print('vertices',time.perf_counter()-start)

        start=time.perf_counter()
        self.tarp.vertices=vertices
        mesh=self.tarp.get_render_mesh(self.sampling_lambda)
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        all_shadow_image=self.rasterizer(mesh)
        shadow_image=tool.shadow_intersection(all_shadow_image)
        #print(shadow_image)
        image=shadow_image.detach().cpu().numpy().transpose((1,2,0))

        print('image loss',time.perf_counter()-start)
        start=time.perf_counter()


        #compute loss using force and shadow image
        self.external_force.record_last_displace()
        barrier_loss=self.external_force()
        print('barrier loss',time.perf_counter()-start)
        start=time.perf_counter()
        shadow_loss=shadow_image_loss(shadow_image,self.target_image)*params.image_weight
        loss=barrier_loss+shadow_loss
        print('shadow loss',time.perf_counter()-start)
        print('total',time.perf_counter()-ptt)
        print('compute loss done')
        
        start=time.perf_counter()
        #update loss figure
        figure_x.append(self.itertimes)
        fmax_loss_cpu=self.external_force.fmax_loss.clone().detach().cpu().numpy()
        fdir_loss_cpu=self.external_force.fdir_loss.clone().detach().cpu().numpy()
        fnorm1_loss_cpu=self.external_force.fnorm1_loss.clone().detach().cpu().numpy()
        shadow_loss_cpu=shadow_loss.clone().detach().cpu().numpy()
        total_loss_cpu=fmax_loss_cpu+fdir_loss_cpu+fnorm1_loss_cpu+shadow_loss_cpu
        if params.enable_prox and not params.fnorm1_cons:
            fnorm1_loss_cpu=self.external_force.FNorm1(self.external_force.force+
                            torch.bmm(self.external_force.transform,self.external_force.force_displace)).clone().detach().cpu().numpy()*params.fnorm1_weight
        figure_fmax_loss.append(fmax_loss_cpu)
        figure_fdir_loss.append(fdir_loss_cpu)
        figure_fnorm1_loss.append(fnorm1_loss_cpu)
        figure_shadow_loss.append(shadow_loss_cpu)
        figure_total_loss.append(total_loss_cpu)
        if self.itertimes%params.updateplt_hz==0:
           plt.clf()
           plt.plot(figure_x,figure_total_loss,label="total loss",color='red')
           plt.plot(figure_x,figure_shadow_loss,label="shadow loss",color='lightgreen')
           plt.plot(figure_x,figure_fnorm1_loss,label="force l1-norm loss",color='peru')
           plt.plot(figure_x,figure_fdir_loss,label="force direction barrier loss",color="cyan")
           plt.plot(figure_x,figure_fmax_loss,label="force maximum barrier loss",color="magenta")
           plt.legend(loc="upper right")

        print('draw figure',time.perf_counter()-start)
        start=time.perf_counter()
        #save some results
        if self.itertimes==0:
            imageio.imsave(os.path.join(self.result_folder,'init.png'),(255*image).astype(np.uint8))
        if self.itertimes%params.saveshadow_hz==0:
            imageio.imsave(os.path.join(self.result_folder,'shadow/deform_%05d.png'%self.itertimes),(255*image).astype(np.uint8))
        if self.itertimes%params.saveloss_hz==0:
            plt.savefig(os.path.join(self.result_folder, 'loss.png'))
        if self.itertimes%params.saveresult_hz==0:
            self.write_results()
        #exit(0)

        print('save results',time.perf_counter()-start)
        start=time.perf_counter()
        #backward
        if params.use_vertgrad:
            vertices.retain_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        print('backward done')

        print('step time',time.perf_counter()-start)
        start=time.perf_counter()
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
        
        print('gui info',time.perf_counter()-start)
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
        print('one iteration time',time.perf_counter()-self.prev_time)
        self.prev_time=time.perf_counter()
        self.itertimes=self.itertimes+1

    def write_params(self):
        tool.copy_file(os.path.join(params.current_dir,'params.yaml'),self.result_folder)

    def write_results(self):
        self.tarp.get_mesh().save_obj(os.path.join(self.result_folder,'result.obj'))
        primal_force=self.external_force.force[0].clone().detach().cpu().numpy()
        tool.write_data(primal_force,os.path.join(self.result_folder,'force.txt'))
        #here I use force_last_displace rather than force_displace because force_displace has been updated
        delta_force=torch.bmm(self.external_force.transform,self.external_force.force_last_displace)[0].clone().detach().cpu().numpy()
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

