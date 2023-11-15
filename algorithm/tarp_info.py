import torch
import torch.nn as nn
import numpy as np
import openmesh

class tarp_info():
    def __init__(self,vertex,data):

        batch_size=vertex.size(0)
        self.nv = vertex.size(1)
        
        #small coef
        self.elp=torch.tensor(data[0]).cuda()
        #stiffnesss                  unit: kg/(m * s^2)
        self.k=torch.tensor(data[1]).cuda()
        #mass
        self.mass=torch.tensor(data[2]).cuda()
        #gravity acceleration        unit: m/(s^2)
        self.g=torch.tensor(data[3]).cuda()
        #gravity                     unit: N
        G=np.zeros([batch_size,self.nv,3]).astype(np.float64)
        G[:,:,2]=-data[2]*data[3]/self.nv
        self.G=torch.from_numpy(G).cuda()
        #maximum force on the rope   unit: N
        self.Fmax=torch.tensor(data[4]).cuda()
        #minimum height of the tarp  unit: m
        self.Hmin=torch.tensor(data[5]).cuda()
        #the height of the sticks    unit: m
        self.H=torch.tensor(data[6]).cuda()
        #maximum length of the rope  unit: m
        self.Lmax=torch.tensor(data[7]).cuda()
        #index of mesh center which is fixed
        self.CI=torch.tensor(int(data[8])).cuda()
        #maximum radius of the mesh
        #sself.Rmax=torch.tensor(1.1*torch.sqrt(torch.sum(vertex[:,:,0:2]**2,dim=2).max())).cuda()
        #vertex that connect with a rope
        self.C0=torch.from_numpy(data[11:11+int(data[9])].astype(int)).cuda()
        #vertex that connect with a rope and a stick
        self.C1=torch.from_numpy(data[-int(data[10]):].astype(int)).cuda()
        #vertex that are forced
        self.C=torch.cat([self.C0,self.C1],dim=0)
        #self.C=0
        #vertically upward direction
        #n=np.zeros([batch_size,self.C.size(0),3]).astype(np.float64)
        #n[:,:,2]=1.0
        #self.n=torch.from_numpy(n).cuda()
    
def get_mesh_boundary(mesh_dir):
    mesh=openmesh.read_trimesh(mesh_dir)
    index=np.array([])
    for v in mesh.vertices():
        if mesh.is_boundary(v):
            index=np.append(index,v.idx())
    #index=np.delete(index,np.arange(1,index.size,2))
    #index=np.delete(index,np.arange(1,index.size,2))
    #index=np.delete(index,np.arange(1,index.size,2))
    return torch.from_numpy(index.astype(int)).cuda()


    
        