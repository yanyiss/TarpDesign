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
VIEW_ANGLE=4  
BALANCE_COF=0.01
LEARNING_RATE=0.1
numerical_error=0

simu_grad_vertices=0

class ExternalForce(nn.Module):
    def __init__(self,vertices,tarp_info):
        super(ExternalForce,self).__init__()
        batch_size=vertices.shape[0]
        f=torch.zeros(batch_size,tarp_info.C.size(0)-1,3).cuda()
        if False:
            C0_dir=vertices[:,tarp_info.C0,:]-vertices[:,tarp_info.CI,:]
            C0_dir=C0_dir/torch.sqrt(torch.sum(C0_dir**2,dim=2)).t().repeat(1,3)
            C1_dir=vertices[:,tarp_info.C1,:]-vertices[:,tarp_info.CI,:]
            C1_dir=C1_dir/torch.sqrt(torch.sum(C1_dir**2,dim=2)).t().repeat(1,3)
        else:
            """ f=torch.tensor([[[ -7.0,  -7.50,  -9.0 ],[  7.0,  -7.10,  -9.0 ],
                             [ -7.0,   7.50,  -9.0 ],[  7.0,   7.10,  -9.0 ],
                             [   0.0,  10.0,  -9.0 ],[   0.0, -10.0,  -9.0 ],
                             [ 10.0,    0.0,   49.05],[-10.0,    0.0,   49.05]]]).cuda() """
            f=torch.tensor([[[ -7.0,  -7.5,  -9.0  ],[   7.0,  -7.1,  -9.0 ],
                             [ -7.0,   7.5,  -9.0  ],[   7.0,   7.1,  -9.0 ],
                             [   0.0,  10.0,  -9.0 ],[   0.0, -10.0,  -9.0 ],
                             [ 10.0,    0.0,  49.05]]]).cuda()
            """ f=torch.tensor([[[ -7.0,  -7.5,  -9.0  ],[   7.0,  -7.1,  -9.0 ],
                             [ 7.0,   7.5,  -9.0   ],[   -7.0,   7.1,  -9.0 ],
                             [ -10.0,    0.0,  40.05]]]).cuda() """
            """ f=torch.tensor([[[ -7.00,  -7.00,  -9.0  ],[   7.0,  7.0,  -9.0 ],
                             [ -7.0,   7.0,  31.05   ]]]).cuda() """
        
        self.register_buffer("force",nn.Parameter(f))
        self.register_parameter("force_displace",nn.Parameter(torch.zeros_like(f)))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))

        self.tarp_info=tarp_info
        self.elp=tarp_info.elp
    
    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)

    #return force constraint energy
    #reference: Incremental Potential Contact: Intersection- and Inversion-free, Large-Deformation Dynamics
    def forward(self):
        resultant_force=(self.force[0]+self.force_displace[0]).t().sum(dim=1)
        resultant_force[0]=-resultant_force[0]
        resultant_force[1]=-resultant_force[1]
        resultant_force[2]=self.tarp_info.mass*self.tarp_info.g-resultant_force[2]
        resultant_force=torch.cat([self.force[0]+self.force_displace[0],resultant_force.unsqueeze(dim=0)],dim=0)
        print((resultant_force**2).sum(dim=1))
        return torch.exp((resultant_force**2).sum(dim=1)-self.tarp_info.Fmax**2).sum()*1e2

        """ current_force=self.force+self.force_displace
        (current_force**2).sum(dim=2) """
        """ print('force num:',(self.force_displace[0].t().sum(dim=1)**2).sum())
        return 1e4*(self.force_displace[0].t().sum(dim=1)**2).sum() """
        """ resultant_force=self.force+self.force_displace#+torch.tensor([[[0.0,0.0,-self.tarp_info.mass*self.tarp_info.g]]]).cuda()
        resultant_force[0,0,2]=resultant_force[0,0,2]-self.tarp_info.mass*self.tarp_info.g
        resultant_force_mag=resultant_force[0].t().sum(dim=1)
        print('resultant force magnitude: ',resultant_force_mag)
        return 1e4*(resultant_force_mag**2).sum() """
        """ resultant_force=self.force+self.force_displace+torch.tensor([0.0,0.0,-self.tarp_info.mass*self.tarp_info.g])
        force_magnitude=torch.sum(force**2,dim=2)
        force_n_projection=force[:,:,2]**2
        return 1e3*torch.sum(force_magnitude) """
        """ +self.logelp(self.tarp_info.Fmax**2-force_magnitude,1.0)
        +self.logelp(0.9*force_magnitude-force_n_projection,1.0)
        +self.logelp(force_n_projection-0.1*force_magnitude,1.0) """

class py_simulatin(torch.autograd.Function):
    @staticmethod
    def forward(ctx,forces,diff_simulator):
        diff_simulator.set_forces(forces[0].clone().detach().cpu().numpy().flatten())
        diff_simulator.Opt()
        diff_simulator.compute_jacobi()
        ctx.jacobi=torch.from_numpy(diff_simulator.jacobi.astype(np.float32)).unsqueeze(dim=0).cuda()
        v_size=int(diff_simulator.v.size/3)
        vertices=diff_simulator.v.reshape(v_size,3).astype(np.float32)
        return torch.from_numpy(vertices).unsqueeze(dim=0).cuda()
    
    @staticmethod
    def backward(ctx,grad_vertices):
        """ print('grad_vertices')
        print(grad_vertices[0][0])
        print(grad_vertices[0][1])
        print(grad_vertices[0][4])
        print(grad_vertices[0][5])
        print(grad_vertices[0][56])
        print(grad_vertices[0][156])
        print(grad_vertices[0][2])
        print(grad_vertices[0][3]) """
        """ print('grad_vertices')
        print(grad_vertices[0][0])
        print(grad_vertices[0][1])
        print(grad_vertices[0][2])
        print(grad_vertices[0][3])
        print(grad_vertices[0][4])
        print(grad_vertices[0][5])
        print(grad_vertices[0][6]) """
        simu_grad_vertices=grad_vertices[0].clone().detach().cpu().numpy()
        grad_vertices=grad_vertices.reshape(1,1,grad_vertices.size(1)*3)
        force_num=int(ctx.jacobi.size(2)/3)
        grad_forces=torch.bmm(grad_vertices,ctx.jacobi).reshape(1,force_num,3)
        """ print('grad_forces')
        print(grad_forces) """
        """ print('jacobi')
        for i in range(9,12):
            for j in range(0,3):
                print(i,j,ctx.jacobi[0][i][j]) """
        return grad_forces,None
    
class Tarp():
    def __init__(self,args):
        template_mesh=sr.Mesh.from_obj(args.template_mesh)
        self.batch_size=args.batch_size
        self.vertices=template_mesh.vertices
        self.faces=template_mesh.faces

        data=np.loadtxt(args.info_path,dtype=np.float32)
        self.tarp_info=TI.tarp_info(self.vertices,data)

    def get_render_mesh(self):
        return sr.Mesh(self.vertices.repeat(self.batch_size,1,1),self.faces.repeat(self.batch_size,1,1))

def shadow_area(image):
    return torch.sum(image)

class deform():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(data_dir, 'obj/hex06.obj'))
        parser.add_argument('-p', '--info-path', type=str, default=os.path.join(data_dir,'info/hex_info06.txt'))
        parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(data_dir, 'results'))
        parser.add_argument('-b', '--batch-size', type=int, default=1)
        self.args = parser.parse_args()

        os.makedirs(self.args.output_dir, exist_ok=True)

        #此处的投影需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #transform=sr.Look(camera_direction=np.array([0,0,1]),perspective=False, eye=np.array([0,0,0]))
        self.transform=sr.LookAt(viewing_angle=VIEW_ANGLE,eye=[0,0,-50])
        self.lighting=sr.Lighting()
        self.rasterizer=sr.SoftRasterizer(image_size=128,sigma_val=1e-4,aggr_func_rgb='hard')

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
        
        self.external_force=ExternalForce(self.tarp.vertices,self.tarp.tarp_info).cuda()
        self.optimizer = torch.optim.Adamax(self.external_force.parameters(), LEARNING_RATE)
        self.writer = imageio.get_writer(os.path.join(self.args.output_dir, 'deform.gif'), mode='I')

        self.simu_index0=self.tarp.tarp_info.C0.clone().detach().cpu().numpy()
        self.simu_index1=self.tarp.tarp_info.C1.clone().detach().cpu().numpy()
        self.simu_jacobi=0
        self.itertimes=0
        self.simu_force=0
        self.set_all_forces()

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
        
    def iterate(self):
        for i in range(0,1000):
            ps=py_simulatin.apply
            vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator)
            self.tarp.vertices=vertices
            mesh=self.tarp.get_render_mesh()
            mesh=self.lighting(mesh)
            mesh=self.transform(mesh)
            shadow_image=self.rasterizer(mesh)

            loss=-shadow_area(shadow_image)
            loss.backward()

            self.optimizer.step()

    def one_iterate(self):
        ps=py_simulatin.apply
        vertices=ps(self.external_force.force+self.external_force.force_displace,self.diffsimulator)
        self.simu_pos=vertices[0].clone().detach().cpu().numpy()
        if self.diffsimulator.print_balance_info(BALANCE_COF)==False:
            return

        """ print(self.external_force.force_displace)
        print(-self.external_force.force_displace[0].t().sum(dim=1)) """
        self.tarp.vertices=vertices
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        shadow_image=self.rasterizer(mesh)

        image=shadow_image.detach().cpu().numpy()[0].transpose((1,2,0))
        #print('image',image)
        #print(image[64][64])
        if self.itertimes%10==0:
            imageio.imsave(os.path.join(self.args.output_dir,'deform_%05d.png'%self.itertimes),(255*image[...,-1]).astype(np.uint8))
        self.itertimes=self.itertimes+1

        loss=-shadow_area(shadow_image)+self.external_force()
        print(loss)
        loss.backward()

        fe=self.diffsimulator.leftmat

        self.simu_jacobi=self.diffsimulator.jacobi.astype(np.float32)
        
        self.optimizer.step()
        self.set_all_forces()
        
        #self.simu_force=(self.external_force.force+self.external_force.force_displace)[0].clone().detach().cpu().numpy()


