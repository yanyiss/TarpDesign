import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import algorithm.tarp_info as TI
import algorithm.tool as tool
import os

params=TI.tarp_params()

class Force(nn.Module):
    def __init__(self,vertices,tarp_info):
        super(Force,self).__init__()
        batch_size=vertices.shape[0]
        self.tarp_info=tarp_info

        """ f=self.get_init_force(vertices)
        self.register_buffer("force",nn.Parameter(f))
        uniquesize=self.force.shape[1]-1
        f=torch.zeros((batch_size,uniquesize,3)).double()
        self.register_parameter("force_displace",nn.Parameter(f))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        
        transform=torch.zeros((batch_size,self.force.shape[1],uniquesize)).double()
        for i in range(uniquesize):
            transform[:,i,i]=1.0
            transform[:,i+1,i]=-1.0
        #transform[:,uniquesize,:]=-1.0
        self.register_buffer("transform",nn.Parameter(transform)) """


        uniquesize=tarp_info.boundary_index.shape[0]-1
        f=torch.zeros((batch_size,uniquesize,3)).double()
        self.register_parameter("force_displace",nn.Parameter(f))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        transform=torch.zeros((batch_size,uniquesize+1,uniquesize)).double()
        for i in range(uniquesize):
            transform[:,i,i]=1.0
            transform[:,i+1,i]=-1.0
        #transform[:,uniquesize,:]=-1.0
        self.register_buffer("transform",nn.Parameter(transform))
        f=self.get_init_force(vertices)
        self.register_buffer("force_dif",nn.Parameter(f))
        
        #self.register_buffer("force",nn.Parameter(f))
        

        self.now_forces=0

        if self.tarp_info.Fmax<params.force_delay:
            print('too big force delay')
            exit(0)
        self.elp=tarp_info.elp
        self.l1_xi=params.l1_xi
        self.l1_eta=params.l1_eta
        self.l1_rho=params.l1_rho
        self.l1_epsilon=params.l1_epsilon
        self.l1_alpha=params.l1_alpha
        self.l1_beta=params.l1_beta

        if params.fnorm1_cons==2:
            self.weight=torch.tensor([1.0],device=torch.device("cuda"))
        else:
            self.weight=torch.tensor([0.0],device=torch.device("cuda"))
        self.boundary_dir=0
        self.update_boudary_dir(vertices)
        self.fmax_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.fdir_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.fnorm1_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.zero=torch.tensor([0.0],device=torch.device("cuda"))
    
    def get_init_force(self,vertices):
        #yanyisheshou
        if params.opt_method=='initial':
            boundary_index=self.tarp_info.boundary_index
            stick_index=self.tarp_info.stick_index
            val=tool.force_SOCP(vertices[0],boundary_index,self.tarp_info.CI,
                            params.horizontal_load*(self.tarp_info.mass*self.tarp_info.g/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
            batch_size=vertices.shape[0]
            f=np.zeros((batch_size,boundary_index.shape[0],3))
            #print(val[:,0].sum(),val[:,1].sum())
            f[0,:,0]=val[:,0]
            f[0,:,1]=val[:,1]
            f[0,stick_index[0],0]=f[0,stick_index[0],0]+50.0
            f[0,stick_index[1],0]=f[0,stick_index[1],0]-50.0
            f=torch.from_numpy(f).cuda()
            down_force=self.tarp_info.mass*self.tarp_info.g*params.vertical_load
            down_average_force=down_force/(boundary_index.shape[0]-self.tarp_info.C.shape[0])
            f[:,:,2]=-down_average_force
            up_force=down_force+self.tarp_info.mass*self.tarp_info.g
            up_average_force=up_force/self.tarp_info.C.shape[0]
            f[:,stick_index,2]=up_average_force
        elif params.opt_method=='manu':
            f=np.loadtxt(os.path.join(params.initial_dir,'last_force.txt'),dtype=np.float64)
            f=torch.from_numpy(f).reshape(1,f.shape[0],3).cuda().double()
            f=f[:,self.tarp_info.sparse_index,:]
            
        solver=tool.linear_solver()

        size=f.shape[1]-1
        r=np.zeros((size,3)).astype(np.float64)
        for i in range(size):
            for j in range(3):
                r[i,j]=f[0,i,j]-f[0,i+1,j]
        r[size-1,2]=r[size-1,2]+(self.tarp_info.mass*self.tarp_info.g).cpu().numpy()
        solver.compute_right(r)

        row=np.zeros((size*3+1)).astype(np.int32)
        col=np.zeros((size*3+1)).astype(np.int32)
        val=np.zeros((size*3+1)).astype(np.float64)

        row[0]=0
        col[0]=0
        val[0]=2.0
        row[1]=0
        col[1]=1
        val[1]=-1.0
            
        count=2
        for i in range(1,size-1):
            row[count]=i
            col[count]=i-1
            val[count]=-1.0
            count=count+1
            row[count]=i
            col[count]=i
            val[count]=2.0
            count=count+1
            row[count]=i
            col[count]=i+1
            val[count]=-1.0
            count=count+1
            
        row[count]=size-1
        col[count]=size-2
        val[count]=-1.0
        count=count+1
        row[count]=size-1
        col[count]=size-1
        val[count]=2.0

        solver.lu(row,col,val)
        return solver.solve().unsqueeze(dim=0).double()
        """ dis=torch.bmm(self.transform.cuda(),dis)
        dis[:,size,2]=dis[:,size,2]+self.tarp_info.mass*self.tarp_info.g
        return dis """

    def compute_now_forces(self):
        # size=self.force.shape[1]
        # self.now_forces[:,0,:]=self.force[:,0,:]+self.force_displace[:,0,:]
        # self.now_forces[:,1:size-1,:]=self.force[:,1:size-1,:]-self.force_displace[:,0:size-2,:]+self.force_displace[:,1:size-1,:]
        # self.now_forces[:,size-1,:]=self.force[:,size-1,:]-self.force[:,size-2,:]

        """ base1 = torch.log((self.vertices1[:, :, 1:2].abs() / 2 + 0.005) / (1 - (self.vertices1[:, :, 1:2].abs() / 2 + 0.005)))
        centroid1 = torch.tanh(self.center)
        vertices1y = ((torch.sigmoid(base1 + self.displace) - 0.005) * 2) * torch.sign(self.vertices1[:, :, 1:2]) """
        # delta=0.001
        # fmax=self.tarp_info.Fmax+delta
        # base=self.force.abs()+delta
        # rr=torch.bmm(self.transform,torch.exp(-self.force_displace))
        # self.now_forces=torch.sign(self.force)*(fmax*base/((1.0-rr)*base+rr*fmax)-delta)
        # #base=torch.log((self.force.abs()+delta)/(fmax-self.force.abs()-delta))
        # #self.now_forces=(torch.sigmoid(base+torch.bmm(self.transform,self.force_displace))*fmax-delta)*torch.sign(self.force)
        # print('sum',torch.sum(self.now_forces,dim=1))
        # print('sum',torch.sum(self.force+torch.bmm(self.transform,self.force_displace),dim=1))
        #self.now_forces=self.force+torch.bmm(self.transform,self.force_displace)

        # delta=0.1
        # fmax=torch.max(self.force_dif)
        # fmin=torch.min(self.force_dif)
        # fstep=fmax-fmin+2*delta
        # base=torch.log((self.force_dif-fmin+delta)/(fstep-(self.force_dif-fmin+delta)))
        # fnew=fstep*(torch.sigmoid(base+self.force_displace))+fmin-delta
        # self.now_forces=torch.bmm(self.transform,fnew)
        # size=self.now_forces.shape[1]-1
        # self.now_forces[:,size,2]=self.now_forces[:,size,2]+self.tarp_info.mass*self.tarp_info.g
        # print('sum',torch.sum(self.now_forces,dim=1))

        # fmax=torch.max(self.force_dif)
        # fmin=torch.min(self.force_dif)
        # fstep=fmax-fmin
        # self.now_forces=torch.bmm(self.transform,self.force_dif+self.force_displace*fstep)
        # size=self.now_forces.shape[1]-1
        # self.now_forces[:,size,2]=self.now_forces[:,size,2]+self.tarp_info.mass*self.tarp_info.g
        # print('sum',torch.sum(self.now_forces,dim=1))

        
        self.now_forces=torch.bmm(self.transform,self.force_dif+self.force_displace)
        size=self.now_forces.shape[1]-1
        self.now_forces[:,size,2]=self.now_forces[:,size,2]+self.tarp_info.mass*self.tarp_info.g
    
    def update_boudary_dir(self,vertices):
        self.boundary_dir=vertices[:,self.tarp_info.C,:]-vertices[:,self.tarp_info.CI,:]
        self.boundary_dir[:,:,2]=0.0
        self.boundary_dir=self.boundary_dir/torch.norm(self.boundary_dir,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def ln(self,x):
        return -torch.log(x)
    
    def FmaxConstraint(self):
        f2=torch.sum(self.now_forces**2,dim=2)
        return torch.sum(self.logelp(1.0-f2/self.tarp_info.Fmax**2,0.1))
        f2=(self.now_forces**2).sum(dim=2)
        return torch.sum(self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*params.force_delay*2-params.force_delay**2))

    def FdirConstraint(self):
        norm_forces=F.normalize(self.now_forces,p=2,dim=2)[:,:,2]
        upper_forces=norm_forces[:,self.tarp_info.stick_index]
        down_forces=norm_forces[:,self.tarp_info.rope_index]
        return self.logelp(upper_forces+1.00001,1.0).sum()+self.logelp(-down_forces+1.00001,1.0).sum()
        return -torch.log(upper_forces+1.000).sum()-torch.log(down_forces).sum()

        norm_forces=F.normalize(forces,p=2,dim=2)[:,:,2]
        return -torch.log(norm_forces**2).sum()
        upper_forces=norm_forces[:,self.stick_index]
        down_forces=-norm_forces[:,self.rope_index]
        return -torch.log(upper_forces).sum()-torch.log(down_forces).sum()
        return self.logelp(upper_forces,params.cosine_delay).sum()+self.logelp(-down_forces,params.cosine_delay).sum()
    
        forces_mag=torch.norm(forces,p=2,dim=2).clone().detach()
        norm_forces=F.normalize(forces,p=2,dim=2)[:,:,2]
        norm_forces=torch.where(forces_mag>self.l1_epsilon,norm_forces,1.0)
        upper_forces=norm_forces[:,self.stick_index]
        down_forces=norm_forces[:,self.rope_index]
        return self.logelp(upper_forces,params.cosine_delay).sum()+self.logelp(-down_forces,params.cosine_delay).sum()

        forces_mag=torch.norm(forces,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3).clone().detach()
        innerproduct=torch.sum(torch.where(forces_mag>self.l1_epsilon,self.boundary_dir*forces/forces_mag,1.0),dim=2)
        return self.logelp(innerproduct+1.0,params.cosine_delay+1.0).sum()
    
    def update_weight(self):
        self.compute_now_forces()
        prevFnorm1=self.FNorm1().clone().detach()
        self.weight=torch.sqrt((self.now_forces**2).sum(dim=2)+self.l1_epsilon).clone().detach()
        self.weight=torch.pow(self.weight+self.l1_xi,self.l1_eta-1.0)
        nowFnorm1=self.FNorm1().clone().detach()
        self.weight=self.weight*prevFnorm1/nowFnorm1
        self.l1_xi=self.l1_xi*self.l1_rho

    def get_sparse_force_index(self):
        self.compute_now_forces()
        norm2=torch.sqrt((self.now_forces**2).sum(dim=2)).squeeze()
        self.sparse_boundary_index=tool.suppleset(torch.nonzero(norm2>0.1).squeeze(),self.tarp_info.stick_index)
        self.sparse_mesh_index=self.tarp_info.boundary_index[self.sparse_boundary_index]
        self.supple_sparse_boundary_index=tool.suppleset(torch.from_numpy(np.arange(0,self.tarp_info.boundary_index.shape[0])).cuda(),self.sparse_boundary_index)
        #print(self.supple_sparse_mesh_index)

    def FNorm1(self):
        """ force_magnitude=torch.sqrt((forces[0]**2).sum(dim=1)+self.l1_epsilon)
        weight=torch.pow(force_magnitude.clone().detach()+self.l1_xi,self.l1_eta-1.0)
        self.l1_xi=self.l1_xi*self.l1_rho
        return (force_magnitude*weight).sum() """
        force_magnitude=torch.sqrt((self.now_forces**2).sum(dim=2)+self.l1_epsilon)
        return torch.sum((force_magnitude*self.weight*self.tarp_info.boundary_weight)[:,self.tarp_info.rope_index])
    
    def record_last_displace(self):
        self.force_last_displace=self.force_displace.clone().detach()

    def forward(self):
        self.compute_now_forces()
        #forces=self.force+torch.bmm(self.transform,self.force_displace)
        self.fmax_loss=self.FmaxConstraint()*params.fmax_weight if params.fmax_cons else self.zero
        self.fdir_loss=self.FdirConstraint()*params.fdir_weight if params.fdir_cons else self.zero
        self.fnorm1_loss=self.FNorm1()*params.fnorm1_weight if params.fnorm1_cons else self.zero
        if torch.any(torch.isnan(self.fmax_loss)):
            print('fmaxloss nan')
        if torch.any(torch.isnan(self.fdir_loss)):
            print('fdirloss nan')
        return self.fmax_loss+self.fdir_loss+self.fnorm1_loss

    #combine reweighted-l1 and proximal gradient --> n-1 forces rather than n
    def prox(self,g):
        g_magnitude=torch.norm(g,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)
        weight=self.l1_alpha*params.fnorm1_weight*self.weight[:,0:-1].unsqueeze(dim=2).repeat(1,1,3)
        return torch.where(g_magnitude>weight,(1.0-weight/g_magnitude)*g,0.0)

    def prox_processing(self):
        dh=self.force[:,0:-1,:]+self.force_last_displace-\
           self.prox(self.force[:,0:-1,:]+self.force_last_displace-self.l1_alpha*(self.force_last_displace-self.force_displace))
        self.force_displace.data=self.force_last_displace-self.l1_beta*dh
