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
        
        uniquesize=tarp_info.rope_index_in_mesh.shape[0]
        f=torch.zeros((batch_size,uniquesize,3)).double()
        self.register_parameter("force_displace",nn.Parameter(f))
        self.register_buffer("force_last_displace",nn.Parameter(f))
        self.register_buffer("now_force",nn.Parameter(f))
        g=self.get_init_force(vertices)
        self.register_buffer("force",nn.Parameter(g))

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

        self.weight=torch.tensor([1.0],device=torch.device("cuda"))
        self.boundary_dir=0
        self.update_boudary_dir(vertices)
        self.fmax_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.fdir_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.fnorm1_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.zero=torch.tensor([0.0],device=torch.device("cuda"))
        self.sparse_index_in_rope=torch.from_numpy(np.arange(0,self.tarp_info.rope_index_in_boundary.shape[0])).cuda()
    
    def get_init_force(self,vertices):
        boundary_index=self.tarp_info.boundary_index
        val=tool.force_SOCP(vertices[0],boundary_index,self.tarp_info.CI,
                        params.horizontal_load*(self.tarp_info.mass*self.tarp_info.g\
                        /boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        f=np.zeros((batch_size,self.tarp_info.rope_index_in_boundary.shape[0],3))
        f[0,:,0]=val[self.tarp_info.rope_index_in_boundary.cpu(),0]
        f[0,:,1]=val[self.tarp_info.rope_index_in_boundary.cpu(),1]
        f[:,:,2]=-0.1
        return torch.from_numpy(f).cuda()
            

    def compute_now_forces(self):
        self.now_force=self.force+self.force_displace
    
    def update_boudary_dir(self,vertices):
        return
        self.boundary_dir=vertices[:,self.tarp_info.C,:]-vertices[:,self.tarp_info.CI,:]
        self.boundary_dir[:,:,2]=0.0
        self.boundary_dir=self.boundary_dir/torch.norm(self.boundary_dir,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FmaxConstraint(self):
        f2=torch.sum(self.now_force**2,dim=2)
        return torch.sum(self.logelp(1.0-f2/self.tarp_info.Fmax**2,0.1))
        f2=(self.now_forces**2).sum(dim=2)
        return torch.sum(self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*params.force_delay*2-params.force_delay**2))

    def FdirConstraint(self):
        norm_forces=F.normalize(self.now_force,p=2,dim=2)[:,:,2]
        upper_forces=norm_forces[:,self.tarp_info.stick_index]
        down_forces=norm_forces[:,self.tarp_info.rope_index]
        return self.logelp(upper_forces+1.00001,1.0).sum()+self.logelp(-down_forces+1.00001,1.0).sum()
    
    def update_weight(self):
        return
        self.compute_now_forces()
        prevFnorm1=self.FNorm1().clone().detach()
        self.weight=torch.sqrt((self.now_force**2).sum(dim=2)+self.l1_epsilon).clone().detach()
        self.weight=torch.pow(self.weight+self.l1_xi,self.l1_eta-1.0)
        nowFnorm1=self.FNorm1().clone().detach()
        self.weight=self.weight#*prevFnorm1/nowFnorm1
        self.l1_xi=self.l1_xi*self.l1_rho

    def restart_sparse(self):
        if self.sparse_index_in_rope.shape[0]<8:
            return
        self.compute_now_forces()
        mag=(self.now_force**2).sum(dim=2,keepdim=True).squeeze()
        max_norm=mag.max()
        mid_value=mag[self.sparse_index_in_rope].median().item()
        self.sparse_index_in_rope=torch.nonzero(torch.where(mag>mid_value,True,False))
        fes=self.force_displace.clone().detach()
        fes[:,self.sparse_index_in_rope,:]=max_norm*F.normalize(self.now_force[:,self.sparse_index_in_rope,:],dim=2)\
                    -self.force[:,self.sparse_index_in_rope,:]
        self.force_displace.data=fes
        

    def FNorm1(self):
        force_magnitude=torch.sqrt((self.now_force**2).sum(dim=2)+self.l1_epsilon)
        return torch.sum(force_magnitude*self.tarp_info.rope_index_weight)
        return torch.sum(force_magnitude)
        return torch.sum(force_magnitude*self.weight*self.tarp_info.rope_index_weight)
    
    def record_last_displace(self):
        self.force_last_displace=self.force_displace.clone().detach()

    def loss_evaluation(self):
        self.compute_now_forces()
        self.fmax_loss=self.FmaxConstraint()*params.fmax_weight if params.fmax_cons else self.zero
        self.fdir_loss=self.FdirConstraint()*params.fdir_weight if params.fdir_cons else self.zero
        self.fnorm1_loss=self.FNorm1()*params.fnorm1_weight if params.fnorm1_cons else self.zero
        return self.fmax_loss+self.fdir_loss+self.fnorm1_loss

    def forward(self):
        self.loss_evaluation()
        return self.fmax_loss+self.fdir_loss+(self.fnorm1_loss if params.use_proximal==0 else self.zero)
    
        
