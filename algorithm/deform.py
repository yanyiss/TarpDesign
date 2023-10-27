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

import algorithm.HardLoss as HL
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
        self.register_buffer('pos_last2_displace', nn.Parameter(torch.zeros_like(self.pos_displace)))
        self.register_parameter('pos_center', nn.Parameter(torch.zeros(args.batch_size, 1, 3)))
        
        #self.register_buffer('faces', self.template_mesh.faces)
        tri=self.template_mesh.faces[0,0,:]
        pos=self.position[0,:,:]
        r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
        #let mesh orients to negative z axis
        """ if r[2] > 0:
            y=[2,1]
            self.template_mesh.faces[:,:,y]=self.template_mesh.faces[:,:,-2:] """
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
        """ f=torch.tensor([[[ -70.0,  -75.0,  -19.0],
         [  70.0,  -71.0,  -19.0],
         [ -70.0,   75.0,  -19.0],
         [  70.0,   71.0,  -19.0],
         [   0.0,  100.0,  -19.0],
         [   0.0, -100.0,  -19.0],
         [ 100.0,    0.0,   79.05],
         [-100.0,    0.0,   79.05]]]) """
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

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.position[0].cpu(), self.faces[0].cpu())
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        self.hard_loss = HL.HardLoss(self.position,self.faces[0].cpu(),data)
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

    def forward(self, batch_size,dt):
        
        rad=np.zeros([batch_size,self.position.size(1),3]).astype(np.float32)
        rad[:,:,0:2]=self.hard_loss.Rmax.cpu()
        rad[:,:,2]=self.hard_loss.H.cpu()*1.05
        rad=torch.from_numpy(rad).cuda()
        #vertices_base = torch.log((self.position+rad) / (rad - self.position.abs()))
        vertice_centroid = torch.tanh(self.pos_center)
        #vertices = (torch.sigmoid(vertices_base + self.pos_displace) * rad) * torch.sign(self.position)
        #vertices = F.relu(vertices) * (1 - vertice_centroid) - F.relu(-vertices) * (vertice_centroid + 1)
        #vertices = vertices + vertice_centroid
        #forces_base=torch.log(self.force.abs()/(self.hard_loss.Fmax-self.force.abs()))
        #forces=(torch.sigmoid(forces_base+self.for_displace)*self.hard_loss.Fmax)*torch.sign(self.force)
        vertices=self.position+self.pos_displace
        '''print('position',self.position)
        print('pos_displace',self.pos_displace)
        print('vertices',vertices)'''
        forces=self.force+self.for_displace
        self.pos_last2_displace=self.pos_last_displace.clone().detach()
        self.pos_last_displace = self.pos_displace.clone().detach()
        self.for_last_displace=self.for_displace.clone().detach()
        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        #flatten_loss = self.flatten_loss(vertices).mean()
        hard_loss=self.hard_loss(vertices,forces,self.position+self.pos_last_displace,
                                 self.position+self.pos_last2_displace,self.faces[0].cpu(),dt)
        

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss,hard_loss#,flatten_loss
        
    def line_search(self, if_LBFGS):
        if if_LBFGS == False:
            return
        rad=np.zeros([self.position.size(0),self.position.size(1),3]).astype(np.float32)
        rad[:,:,0:2]=self.hard_loss.Rmax.cpu()
        rad[:,:,2]=self.hard_loss.H.cpu()*1.1
        rad=torch.from_numpy(rad).cuda()
        #vertices_base = torch.log(self.position.abs() / (rad - self.position.abs()))
        #f_base=torch.log(self.force.abs()/(self.hard_loss.Fmax-self.force.abs()))
        deltaposition=self.pos_displace-self.pos_last_displace
        #self.for_displace.data=0.0*self.for_displace
        deltaforce=self.for_displace-self.for_last_displace
        '''with open ('/home/yanyisheshou/SoftRas/data/ref/hex20_force.txt','w') as file_object:
            file_object.write(str(deltaforce.cpu().tolist()))
        with open ('/home/yanyisheshou/SoftRas/data/ref/hex20_position.txt','w') as file_object:
            file_object.write(str(deltaposition.cpu().tolist())) '''  
        
        
        '''print('xxx',self.pos_displace[:,326,:],deltaposition[:,326,:])
        print('xxyy',self.pos_displace[:,212,:],deltaposition[:,212,:])
        print('yyy',self.pos_displace[:,0,:],deltaposition[:,0,:])'''

        C0=self.hard_loss.C0.clone()
        C1=self.hard_loss.C1.clone()
        CI=self.hard_loss.CI.clone()
        Fmax=self.hard_loss.Fmax.clone()
        Lmax=self.hard_loss.Lmax.clone()
        Hmin=self.hard_loss.Hmin.clone()
        n=self.hard_loss.n.clone()
        itertimes=0
        while 1:
            vertices=self.position+self.pos_last_displace+deltaposition
            C0_dir=vertices[:,C0,:]-vertices[:,CI,:]
            C1_dir=vertices[:,C1,:]-vertices[:,CI,:]
            #vertices=torch.sigmoid(vertices_base + self.pos_last_displace + deltaposition) * rad * torch.sign(self.position)
            #f=torch.sigmoid(f_base+self.for_last_displace+deltaforce)*self.hard_loss.Fmax*torch.sign(self.force)
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
            condition3=((newlen-self.hard_loss.len)*self.hard_loss.adj<1e-5)'''
            
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
        #print('delta',self.pos_displace)


def shadow_area(image):
    return torch.sum(image.abs())

def projective_dynamics(model, args, itertimes):
    #optimize local equivalence by fast mass spring
    loop = tqdm.tqdm(list(range(0,itertimes)))
    #optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    optimizer=torch.optim.LBFGS(model.parameters(),max_eval=1)
    is_LBFGS=True
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    for i in loop:
        def closure():
            optimizer.zero_grad()
            model.hard_loss.activated=np.array([True,False,True,False,False,True])
            mesh, laplacian_loss, hard_loss = model(args.batch_size,1-0.0005*i)
            
            loss = hard_loss
            

            loss.backward()
            if i % 10 == 0:
                tri=model.template_mesh.faces[0,0,:]
                pos=model.position[0,:,:]
                r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
                pp=False
                #let mesh orients to negative z axis
                if r[2] < 0:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                model(1,1)[0].save_obj(os.path.join(args.output_dir, 'deform_%05d.obj' % i), save_texture=False)
                if pp:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                        
                    
            return loss

        optimizer.step(closure)
        if model.exit:
            model.exit=False
            break
    return 0

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

        self.pos=self.model.template_mesh.vertices[0].clone().detach().cpu().numpy()

        self.pd=Simulation()
        self.pd.set_info(self.model.template_mesh.faces[0].clone().detach().cpu().numpy(),
                         self.pos.flatten(),
                         self.model.hard_loss.k.clone().detach().cpu().numpy(),0.1,
                         self.model.hard_loss.mass.clone().detach().cpu().numpy()/3506,1647)
        

    def get_init_force(self):
        """ loop = tqdm.tqdm(list(range(0, 10)))
        optimizer=torch.optim.LBFGS(self.model.parameters())
        is_LBFGS=True
        for i in loop:
            #break
            def closure():
                self.model.line_search(is_LBFGS)
                optimizer.zero_grad()
                self.model.hard_loss.activated=np.array([False,True,False,True,False,False])
                loss = self.model(self.args.batch_size,1)[2]
                loop.set_description('Loss: %.4f'%(loss.item()))

                loss.backward()
                return loss

            optimizer.step(closure)
            self.model.line_search(True)
            if self.model.exit:
                self.model.exit=False
                break """
        print(self.model.force+self.model.for_displace)
        force=self.model.hard_loss.G[0].clone().detach()
        force[self.model.hard_loss.C0,:]=force[self.model.hard_loss.C0,:]+self.model.force[0][0:self.model.hard_loss.C0.size(0),:]\
                                                                        +self.model.for_displace[0][0:self.model.hard_loss.C0.size(0),:]
        force[self.model.hard_loss.C1,:]=force[self.model.hard_loss.C1,:]+self.model.force[0][-self.model.hard_loss.C1.size(0):,:]\
                                                                        +self.model.for_displace[0][-self.model.hard_loss.C1.size(0):,:]
        #print(force)
        self.pd.set_forces(force.clone().detach().cpu().numpy().flatten())
    
    def step(self):

        self.pd.Opt()
        size=self.pd.v.size
        self.pos=self.pd.v.reshape(3506,3)
        #print(self.pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str, default=os.path.join(data_dir, 'obj/hex06.obj'))
    parser.add_argument('-p', '--info-path', type=str, default=os.path.join(data_dir,'info/hex_info06.txt'))
    parser.add_argument('-o', '--output-dir', type=str, default=os.path.join(data_dir, 'results'))
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(args).cuda()
    model(1,1)[0].save_obj(os.path.join(args.output_dir, 'plane.obj'), save_texture=False)
    #此处的投影需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #transform=sr.Look(camera_direction=np.array([0,0,1]),perspective=False, eye=np.array([0,0,0]))
    transform=sr.LookAt(viewing_angle=3,eye=[0,0,-50])
    lighting=sr.Lighting()
    rasterizer=sr.SoftRasterizer(image_size=128,sigma_val=1e-4,aggr_func_rgb='hard')
    
    
    
    #optimize global equivalence
    loop = tqdm.tqdm(list(range(0, 100)))
    optimizer=torch.optim.LBFGS(model.parameters())
    is_LBFGS=True
    for i in loop:
        #break
        def closure():
            model.line_search(is_LBFGS)
            optimizer.zero_grad()
            model.hard_loss.activated=np.array([False,True,False,True,False,False])
            loss = model(args.batch_size,1)[2]
            loop.set_description('Loss: %.4f'%(loss.item()))

            loss.backward()
            return loss

        optimizer.step(closure)
        model.line_search(True)
        if model.exit:
            model.exit=False
            break
    
    print(model.force+model.for_displace)
    
    for i in range(0,10):
        tri=model.template_mesh.faces[0,0,:]
        pos=model.position[0,:,:]
        r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
        pp=False
                #let mesh orients to negative z axis
        if r[2] < 0:
            y=[2,1]
            pp=True
            model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
        #model(1)[0].save_obj(os.path.join(args.output_dir, 'deform_%05d.obj' % i), save_texture=False)
        if pp:
            y=[2,1]
            pp=True
            model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]

    


                    
    #optimize local equivalence by fast mass spring
    loop = tqdm.tqdm(list(range(0,2000)))
    #optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    optimizer=torch.optim.LBFGS(model.parameters(),max_eval=1)
    is_LBFGS=True
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    for i in loop:
        #break
        def closure():
            #model.line_search(is_LBFGS)
            optimizer.zero_grad()
            model.hard_loss.activated=np.array([True,False,True,False,False,True])
            mesh, laplacian_loss, hard_loss = model(args.batch_size,1-0.0005*i)
            
            mesh = lighting(mesh)
            mesh = transform(mesh)
            mesh_shadow = rasterizer(mesh)
            loss = 0.00 * laplacian_loss + hard_loss#+0.0*flatten_loss
            loop.set_description('Loss: %.4f, laplacian_loss: %.4f, hard_loss: %.4f'%\
                (loss.item(), laplacian_loss, hard_loss))

            loss.backward()
            if i % 10 == 0:
                tri=model.template_mesh.faces[0,0,:]
                pos=model.position[0,:,:]
                r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
                pp=False
                #let mesh orients to negative z axis
                if r[2] < 0:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                model(1,1)[0].save_obj(os.path.join(args.output_dir, 'deform_%05d.obj' % i), save_texture=False)
                if pp:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                        
                image = mesh_shadow.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255*image).astype(np.uint8))
                #imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png' % i), (255*image[..., -1]).astype(np.uint8))
                    
            return loss

        optimizer.step(closure)
        #model.line_search(True)
        if model.exit:
            model.exit=False
            break
    
    #optimize local equivalence
    loop = tqdm.tqdm(list(range(1000,5000)))
    #optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    optimizer=torch.optim.LBFGS(model.parameters(),max_eval=1)
    is_LBFGS=True
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    for i in loop:
        break
        def closure():
            #model.line_search(is_LBFGS)
            optimizer.zero_grad()
            model.hard_loss.activated=np.array([True,False,True,False,False,False])
            mesh, laplacian_loss, hard_loss,flatten_loss = model(args.batch_size)
            
            mesh = lighting(mesh)
            mesh = transform(mesh)
            mesh_shadow = rasterizer(mesh)
            loss = 0.00 * laplacian_loss + hard_loss+0.0*flatten_loss
            loop.set_description('Loss: %.4f, laplacian_loss: %.4f, hard_loss: %.4f'%\
                (loss.item(), laplacian_loss, hard_loss))

            loss.backward()
            if i % 10 == 0:
                tri=model.template_mesh.faces[0,0,:]
                pos=model.position[0,:,:]
                r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
                pp=False
                #let mesh orients to negative z axis
                if r[2] < 0:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                model(1)[0].save_obj(os.path.join(args.output_dir, 'deform_%05d.obj' % i), save_texture=False)
                if pp:
                    y=[2,1]
                    pp=True
                    model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
                        
                image = mesh_shadow.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255*image).astype(np.uint8))
                #imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png' % i), (255*image[..., -1]).astype(np.uint8))
                    
            return loss

        optimizer.step(closure)
        #model.line_search(True)
        if model.exit:
            model.exit=False
            break
    
    for i in loop:
        break
        def closure():
            model.line_search(is_LBFGS)
            optimizer.zero_grad()
            model.hard_loss.activated=np.array([True,False,True,False,True])
            mesh, laplacian_loss, hard_loss = model(args.batch_size)
            
            mesh = lighting(mesh)
            mesh = transform(mesh)
            mesh_shadow = rasterizer(mesh)
            shadow_loss=-shadow_area(mesh_shadow)*0.0
            loss = 0.00*shadow_loss+ 0.00 * laplacian_loss + hard_loss
            #loss=hard_loss
            if torch.isnan(loss):
                os.system("pause")
            loop.set_description('Loss: %.4f, laplacian_loss: %.4f, hard_loss: %.4f, shadow_loss: %.4f'%\
                (loss.item(), laplacian_loss, hard_loss, shadow_loss))

            loss.backward()
            if i % 50 == 0:
                image = mesh_shadow.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255*image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png' % i), (255*image[..., -1]).astype(np.uint8))

            return loss

        optimizer.step(closure)
        model.line_search(True)
        if model.exit:
            model.exit=False
            break
    
    print('Work Done')
    with open('/home/yanyisheshou/SoftRas/data/ref/hex20_force.txt','w') as f:
        print(model.force+model.for_displace,file=f)

    # save optimized mesh
    
    tri=model.template_mesh.faces[0,0,:]
    pos=model.position[0,:,:]
    r=torch.cross(pos[tri[1]-tri[0],:],pos[tri[2]-tri[0],:])
    #let mesh orients to negative z axis
    if r[2] < 0:
        y=[2,1]
        model.template_mesh.faces[:,:,y]=model.template_mesh.faces[:,:,-2:]
    model(1,1)[0].save_obj(os.path.join(args.output_dir, 'plane.obj'), save_texture=False)


if __name__ == '__main__':
    main()
