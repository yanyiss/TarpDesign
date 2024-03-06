import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import algorithm.tarp_info as TI
import algorithm.tool as tool
import os

params=TI.tarp_params()

class Force(nn.Module):
    def __init__(self,vertices,tarp_info,boundary_index,boundary_weight,stick_index):
        super(Force,self).__init__()
        batch_size=vertices.shape[0]
        self.boundary_weight=boundary_weight
        f=self.get_init_force(vertices,tarp_info,boundary_index,stick_index)
        if params.use_denseInfo:
            f=torch.from_numpy(np.loadtxt(params.force_file,
                                          dtype=np.float64)).reshape(batch_size,f.shape[1],3).cuda().double()
        self.register_buffer("force",nn.Parameter(f))
        uniquesize=self.force.shape[1]-1
        f=torch.zeros((batch_size,uniquesize,3)).double()
        if params.use_denseInfo:
            f=torch.from_numpy(np.loadtxt(params.forcedis_file,
                                          dtype=np.float64)).reshape(batch_size,f.shape[1],3).cuda().double()
        self.register_parameter("force_displace",nn.Parameter(f))
        self.register_buffer("force_last_displace",nn.Parameter(torch.zeros_like(f)))
        
        transform=torch.zeros((batch_size,self.force.shape[1],uniquesize)).double()
        for i in range(uniquesize):
            transform[:,i,i]=1.0
            transform[:,i+1,i]=-1.0
        #transform[:,uniquesize,:]=-1.0
        self.register_buffer("transform",nn.Parameter(transform))

        self.tarp_info=tarp_info
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

        self.boundary_index=boundary_index
        self.stick_index=stick_index
        self.rope_index=tool.suppleset(boundary_index,stick_index)
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
        #self.update_weight()
    
    def get_init_force(self,vertices,tarp_info,boundary_index,stick_index):
        val=tool.force_SOCP(vertices[0],boundary_index,tarp_info.CI,
                            params.horizontal_load*(tarp_info.mass*tarp_info.g/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        f=np.zeros((batch_size,boundary_index.shape[0],3))
        #print(val[:,0].sum(),val[:,1].sum())
        f[0,:,0]=val[:,0]
        f[0,:,1]=val[:,1]
        f=torch.from_numpy(f).cuda()
        down_force=tarp_info.mass*tarp_info.g*params.vertical_load
        down_average_force=down_force/(boundary_index.shape[0]-tarp_info.C.shape[0])
        f[:,:,2]=-down_average_force
        up_force=down_force+tarp_info.mass*tarp_info.g
        up_average_force=up_force/tarp_info.C.shape[0]
        f[:,stick_index,2]=up_average_force
        return f
        
    def now_force(self):
        return self.force+torch.bmm(self.transform,self.force_displace)
    
    def update_boudary_dir(self,vertices):
        self.boundary_dir=vertices[:,self.tarp_info.C,:]-vertices[:,self.tarp_info.CI,:]
        self.boundary_dir[:,:,2]=0.0
        self.boundary_dir=self.boundary_dir/torch.norm(self.boundary_dir,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def ln(self,x):
        return -torch.log(x)
    
    def FmaxConstraint(self,forces):
        f2=(forces**2).sum(dim=2)
        return torch.sum(self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*params.force_delay*2-params.force_delay**2))

    def FdirConstraint(self,forces):
        norm_forces=F.normalize(forces,p=2,dim=2)[:,:,2]
        upper_forces=norm_forces[:,self.stick_index]
        down_forces=norm_forces[:,self.rope_index]
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
        prevFnorm1=self.FNorm1(self.now_force()).clone().detach()
        self.weight=torch.sqrt((self.now_force()**2).sum(dim=2)+self.l1_epsilon).clone().detach()
        self.weight=torch.pow(self.weight+self.l1_xi,self.l1_eta-1.0)
        nowFnorm1=self.FNorm1(self.now_force()).clone().detach()
        self.weight=self.weight*prevFnorm1/nowFnorm1
        self.l1_xi=self.l1_xi*self.l1_rho

    def FNorm1(self,forces):
        """ force_magnitude=torch.sqrt((forces[0]**2).sum(dim=1)+self.l1_epsilon)
        weight=torch.pow(force_magnitude.clone().detach()+self.l1_xi,self.l1_eta-1.0)
        self.l1_xi=self.l1_xi*self.l1_rho
        return (force_magnitude*weight).sum() """
        force_magnitude=torch.sqrt((forces**2).sum(dim=2)+self.l1_epsilon)
        return torch.sum((force_magnitude*self.weight*self.boundary_weight)[:,self.rope_index])
    
    def linesearch(self,vertices):
        if params.fmax_cons+params.fdir_cons==0:
            return
        if params.fdir_cons:
            self.update_boudary_dir(vertices)
        deltaforce=self.force_displace-self.force_last_displace
        itertimes=0
        if torch.any(torch.isnan(deltaforce)):
            print('yes')
            exit(0)
        while 1:
            itertimes=itertimes+1
            if itertimes>500:
                # print('itertimes>500')
                # if (deltaforce**2).sum()<params.grad_error:
                #     print('locking')
                #     deltaforce=0.0*deltaforce
                #     break
                print('locking')
                deltaforce=0.0*deltaforce
                break
            
            if params.fmax_cons:
                forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
                f2=(forces**2).sum(dim=2)
                ForceMaxCondition=((self.tarp_info.Fmax**2)<(f2*(1.0+params.nume_error)))
                if torch.any(ForceMaxCondition):
                    deltaforce=0.8*deltaforce
                    continue
                
            # if params.fdir_cons:
            #     forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
            #     norm_forces=F.normalize(forces,p=2,dim=2)[0,:,2]
            #     ForceDirCondition=torch.zeros_like(norm_forces,dtype=bool)
            #     ForceDirCondition[self.stick_index]=norm_forces[self.stick_index]<params.nume_error
            #     ForceDirCondition[self.rope_index]=-norm_forces[self.rope_index]<params.nume_error
            #     size=ForceDirCondition.shape[0]-1
            #     ForceDirCondition=ForceDirCondition[0:size]+ForceDirCondition[1:size+1]
            #     if torch.any(ForceDirCondition):
            #         ForceDirCondition=ForceDirCondition.unsqueeze(dim=0).reshape(size,1).repeat(1,1,3)
            #         deltaforce=torch.where(ForceDirCondition,0.5*deltaforce,deltaforce)
            #         continue

                """ upper_forces=norm_forces[self.stick_index]
                down_forces=norm_forces[self.rope_index]
                upperForceDirCondition=upper_forces<params.nume_error
                downForceDirCondition=-down_forces<params.nume_error
                if upperForceDirCondition.sum()+downForceDirCondition.sum()!=0:
                    deltaforce=0.8*deltaforce
                    continue """

                # forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
                # forces_mag=torch.norm(forces,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3).clone().detach()
                # innerproduct=torch.sum(torch.where(forces_mag>0.95*self.l1_epsilon,self.boundary_dir*forces/forces_mag,1.0),dim=2)
                # ForceDirCondition=(innerproduct+1.0<params.nume_error)
                # if ForceDirCondition.sum()!=0:
                #     deltaforce=0.8*deltaforce
                #     continue
            break

        self.force_displace.data=self.force_last_displace+deltaforce
        print('linesearch:',itertimes)
    
    def record_last_displace(self):
        self.force_last_displace=self.force_displace.clone().detach()

    def forward(self):
        forces=self.force+torch.bmm(self.transform,self.force_displace)
        self.fmax_loss=self.FmaxConstraint(forces)*params.fmax_weight if params.fmax_cons else self.zero
        self.fdir_loss=self.FdirConstraint(forces)*params.fdir_weight if params.fdir_cons else self.zero
        self.fnorm1_loss=self.FNorm1(forces)*params.fnorm1_weight if params.fnorm1_cons else self.zero
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
