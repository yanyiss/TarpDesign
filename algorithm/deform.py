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
numerical_error=0

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        data = np.loadtxt(args.info_path,dtype=np.float32)
        self.exit=False

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(args.template_mesh)
        self.register_buffer('textures', self.template_mesh.textures)
        

        self.register_buffer('position', self.template_mesh.vertices)
        self.register_parameter('pos_displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_buffer('pos_last_displace', nn.Parameter(torch.zeros_like(self.pos_displace)))
        self.register_parameter('pos_center', nn.Parameter(torch.zeros(args.batch_size, 1, 3)))
        
        self.register_buffer('faces',self.template_mesh.faces)
        
        f=torch.zeros(args.batch_size,int(data[9])+int(data[10]),3)
        data[11:11+int(data[9])].astype(int)
        '''f=torch.tensor([[[ 0.0000, -0.0500, -0.0375],
         [ 0.0000, -0.0500, -0.0375],
         [ 0.0000,  0.0500, -0.0375],
         [ 0.0000,  0.0500, -0.0375],
         [-0.0500,  0.0000, 22.1251],
         [ 0.0500,  0.0000, 22.1251]]])'''
        C0_dir=self.position[:,data[11:11+int(data[9])].astype(int),:]-self.position[:,int(data[8]),:]
        C0_dir=C0_dir/torch.sqrt(torch.sum(C0_dir**2,dim=2)).t().repeat(1,3)
        C1_dir=self.position[:,data[-int(data[10]):].astype(int),:]-self.position[:,int(data[8]),:]
        C1_dir=C1_dir/torch.sqrt(torch.sum(C1_dir**2,dim=2)).t().repeat(1,3)
        f[:,0:int(data[9]),:]=C0_dir*100.0
        f[:,-int(data[10]):,:]=C1_dir*100.0
        f[:,0:int(data[9]),2]=-50.0
        f[:,-int(data[10]):,2]=50.0
        f=torch.tensor([[[ -70.0,  -75.0,  -19.0],
         [  70.0,  -71.0,  -19.0],
         [ -70.0,   75.0,  -19.0],
         [  70.0,   71.0,  -19.0],
         [   0.0,  100.0,  -19.0],
         [   0.0, -100.0,  -19.0],
         [ 100.0,    0.0,   79.05],
         [-100.0,    0.0,   79.05]]])
        '''f[:,0:2,1]=-0.05
        f[:,2:4,1]=0.05
        f[:,4,0]=-0.05
        f[:,5,0]=0.05'''
        '''f[:,0:3,1]=-50.0
        f[:,3:6,1]=50.0
        f[:,6,0]=-50.0
        f[:,7,0]=50.0'''
        self.register_buffer('force', nn.Parameter(f))
        self.register_parameter('for_displace',nn.Parameter(torch.zeros_like(f)))
        self.register_buffer('for_last_displace',nn.Parameter(torch.zeros_like(f)))

        self.tarp_info = TI.tarp_info(self.position,data)

    def force_constraint(self,vertices,forces):
        def logelp(self,x, elp=0.0):
            if elp>0.0:
                return torch.where(x<elp,-(x-elp)**2*torch.log(x/elp)/(elp**2),0.0)
            else:
                return torch.where(x<self.elp,-(x-self.elp)**2*torch.log(x/self.elp)/(self.elp**2),0.0)  
        return 1e3*(forces**2).sum()

    def forward(self, batch_size,dt):
        
        rad=np.zeros([batch_size,self.position.size(1),3]).astype(np.float32)
        rad[:,:,0:2]=self.tarp_info.Rmax.cpu()
        rad[:,:,2]=self.tarp_info.H.cpu()*1.05
        rad=torch.from_numpy(rad).cuda()
        #vertices_base = torch.log((self.simu_position+rad) / (rad - self.simu_position.abs()))
        vertice_centroid = torch.tanh(self.pos_center)
        #vertices = (torch.sigmoid(vertices_base + self.simu_pos_displace) * rad) * torch.sign(self.simu_position)
        #vertices = F.relu(vertices) * (1 - vertice_centroid) - F.relu(-vertices) * (vertice_centroid + 1)
        #vertices = vertices + vertice_centroid
        #forces_base=torch.log(self.force.abs()/(self.tarp_info.Fmax-self.force.abs()))
        #forces=(torch.sigmoid(forces_base+self.for_displace)*self.tarp_info.Fmax)*torch.sign(self.force)

        vertices=self.position+self.pos_displace
        forces=self.force+self.for_displace
        self.pos_last_displace=self.pos_displace.clone().detach()
        self.for_last_displace=self.for_displace.clone().detach()
        #laplacian_loss = self.laplacian_loss(vertices).mean()
        # apply Laplacian and flatten geometry constraints
        #flatten_loss = self.flatten_loss(vertices).mean()
        """ tarp_info=self.tarp_info(vertices,forces,self.simu_position+self.simu_pos_last_displace,
                                 self.simu_position+self.simu_pos_last2_displace,self.faces[0].cpu(),dt) """
        

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1))#, laplacian_loss#,tarp_info#,flatten_loss
        
    def line_search(self, if_LBFGS):
        if if_LBFGS == False:
            return
        rad=np.zeros([self.position.size(0),self.position.size(1),3]).astype(np.float32)
        rad[:,:,0:2]=self.tarp_info.Rmax.cpu()
        rad[:,:,2]=self.tarp_info.H.cpu()*1.1
        rad=torch.from_numpy(rad).cuda()
        #vertices_base = torch.log(self.simu_position.abs() / (rad - self.simu_position.abs()))
        #f_base=torch.log(self.force.abs()/(self.tarp_info.Fmax-self.force.abs()))
        deltaposition=self.pos_displace-self.pos_last_displace
        #self.for_displace.data=0.0*self.for_displace
        deltaforce=self.for_displace-self.for_last_displace
        '''with open ('/home/yanyisheshou/SoftRas/data/ref/hex20_force.txt','w') as file_object:
            file_object.write(str(deltaforce.cpu().tolist()))
        with open ('/home/yanyisheshou/SoftRas/data/ref/hex20_simu_position.txt','w') as file_object:
            file_object.write(str(deltasimu_position.cpu().tolist())) '''  
        
        
        '''print('xxx',self.simu_pos_displace[:,326,:],deltasimu_position[:,326,:])
        print('xxyy',self.simu_pos_displace[:,212,:],deltasimu_position[:,212,:])
        print('yyy',self.simu_pos_displace[:,0,:],deltasimu_position[:,0,:])'''

        C0=self.tarp_info.C0.clone()
        C1=self.tarp_info.C1.clone()
        CI=self.tarp_info.CI.clone()
        Fmax=self.tarp_info.Fmax.clone()
        Lmax=self.tarp_info.Lmax.clone()
        Hmin=self.tarp_info.Hmin.clone()
        n=self.tarp_info.n.clone()
        itertimes=0
        while 1:
            vertices=self.position+self.pos_last_displace+deltaposition
            C0_dir=vertices[:,C0,:]-vertices[:,CI,:]
            C1_dir=vertices[:,C1,:]-vertices[:,CI,:]
            #vertices=torch.sigmoid(vertices_base + self.simu_pos_last_displace + deltasimu_position) * rad * torch.sign(self.simu_position)
            #f=torch.sigmoid(f_base+self.for_last_displace+deltaforce)*self.tarp_info.Fmax*torch.sign(self.force)
            f=self.force+self.for_last_displace+deltaforce
            F0value=torch.sqrt(torch.sum(f[:,0:C0.size(0),:]**2,dim=2))
            F1value=torch.sum(torch.cross(f[:,-C1.size(0):,:],n)**2,dim=2)
            
            
            '''print((Fmax<F0value+numerical_error),
                -f[:,0:C0.size(0),2]/F0value,vertices[:,C0,2]/Lmax+numerical_error,
                (torch.sum(f[:,0:C0.size(0),:]*C0_dir,dim=2)/torch.sqrt(torch.sum(C0_dir**2,dim=2))<numerical_error))'''
            
            
            '''condition0=(Fmax<=F0value+numerical_error)\==================================================
                +(-f[:,0:C0.size(0),2]/F0value<vertices[:,C0,2]/Lmax+numerical_error)\
                +(torch.sum(f[:,0:C0.size(0),:]*C0_dir,dim=2)/
                  torch.sqrt(torch.sum(f[:,0:C0.size(0),:]**2,dim=2)*torch.sum(C0_dir**2,dim=2))<numerical_error)
            condition1=(f[:,-C1.size(0):,2]<numerical_error)\
                +((Lmax**2-vertices[:,C1,2]**2)/Lmax**2<F1value/Fmax**2+numerical_error)\
                +(torch.sum(f[:,-C1.size(0):,:]*C1_dir,dim=2)/
                  torch.sqrt(torch.sum(f[:,-C1.size(0):,:]**2,dim=2)*torch.sum(C1_dir**2,dim=2))<numerical_error)
            condition2=(vertices[:,:,2]<Hmin+numerical_error)'''
                
            condition0=(f[:,0:C0.size(0),2]>=0)+(torch.sum(f[:,0:C0.size(0),:]*C0_dir,dim=2)<=0)\
                +(F0value>=Fmax)
            condition1=(f[:,-C1.size(0):,2]<=0)+(torch.sum(f[:,-C1.size(0):,:]*C1_dir,dim=2)<=0)\
                +(torch.sqrt(torch.sum(f[:,-C1.size(0):,:]**2,dim=2))>Fmax)
            
            condition2=(vertices[:,:,2]<=Hmin*0+numerical_error)
            '''batch_size=vertices.size(0)
            newlen=torch.sqrt((vertices[:,:,0]-vertices[:,:,0].reshape(batch_size,self.nv,1))**2
                         +(vertices[:,:,1]-vertices[:,:,1].reshape(batch_size,self.nv,1))**2
                         +(vertices[:,:,2]-vertices[:,:,2].reshape(batch_size,self.nv,1))**2
                        +torch.eye(self.nv).unsqueeze(dim=0).cuda())
            condition3=((newlen-self.tarp_info.len)*self.tarp_info.adj<1e-5)'''
            
            if condition0.sum()+condition1.sum()+condition2.sum()==0:
                break
            for i in range(C0.size(0)):
                if condition0[:,i] != 0:
                    deltaposition[:,C0[i],:]=0.8*deltaposition[:,C0[i],:]
                    deltaforce[:,i,:]=0.8*deltaforce[:,i,:]
            for i in range(C1.size(0)):
                if condition1[:,i] != 0:
                    deltaposition[:,C1[i],:]=0.8*deltaposition[:,C1[i],:]
                    deltaforce[:,C0.size(0)+i,:]=0.8*deltaforce[:,C0.size(0)+i,:]
            
            deltaposition=torch.where(condition2.t().repeat(1,3),0.8*deltaposition,deltaposition)
            
            if torch.max(deltaposition.abs())+torch.max(deltaforce.abs())<1e-8:
                print('too many iteration')
                deltaposition=0.0*deltaposition
                deltaforce=0.0*deltaforce
                self.exit=True
                break
                #exit(0)
                
        
        self.pos_displace.data=self.pos_last_displace+deltaposition
        self.for_displace.data=self.for_last_displace+deltaforce
        #print('delta',self.simu_pos_displace)


def shadow_area(image):
    return torch.sum(image.abs())

class deform():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(data_dir, 'obj/hex06.obj'))
        parser.add_argument('-p', '--info-path', type=str, default=os.path.join(data_dir,'info/hex_info06.txt'))
        parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(data_dir, 'results'))
        parser.add_argument('-b', '--batch-size', type=int, default=1)
        self.args = parser.parse_args()

        os.makedirs(self.args.output_dir, exist_ok=True)

        self.model = Model(self.args).cuda()
        #此处的投影需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #transform=sr.Look(camera_direction=np.array([0,0,1]),perspective=False, eye=np.array([0,0,0]))
        self.transform=sr.LookAt(viewing_angle=3,eye=[0,0,-50])
        self.lighting=sr.Lighting()
        self.rasterizer=sr.SoftRasterizer(image_size=128,sigma_val=1e-4,aggr_func_rgb='hard')

        #self.simu_pos=self.model.template_mesh.vertices[0].clone().detach().cpu().numpy()
        self.simu_pos=(self.model.position+self.model.pos_displace)[0].clone().detach().cpu().numpy()

        self.pd=Simulation()
        self.pd.set_info(self.model.template_mesh.faces[0].clone().detach().cpu().numpy(),
                         self.simu_pos.flatten(),
                         self.model.tarp_info.C.clone().detach().cpu().numpy(),
                         self.model.tarp_info.k.clone().detach().cpu().numpy(),0.1,
                         self.model.tarp_info.mass.clone().detach().cpu().numpy(),1647)
        

    def set_init_force(self):
        self.pd.set_forces(self.model.force[0].clone().detach().cpu().numpy().flatten())
    
    def step(self):
        self.pd.Opt()
        size=self.pd.v.size
        self.simu_pos=self.pd.v.reshape(3506,3)
        #print(self.simu_pos)
