import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import algorithm.tarp_info as TI
import algorithm.tool as tool
import os

params=TI.tarp_params()

class Geometry():
    def __init__(self,tarp,force):
        self.tarp=tarp
        self.force=force
        # 以下有错误
        # self.stick_index=tarp.tarp_info.C
        # self.stick_value=self.tarp.vertices[:,self.stick_index,2].clone().detach()
        self.fixed_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.ropelength_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.geometry_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.zero=torch.tensor([0.0],device=torch.device("cuda"))
        self.deltaforce=self.zero

    def logelp(self,x,elp=0.0):
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def clampingX2(self,x,elp=0.0):
        return torch.where(x<elp,(x-elp)**2,0.0)
    
    def clampingX3(self,x,elp=0.0):
        return torch.where(x<elp,-(x-elp)**3,0.0)
    
    def fixed_constraint(self):
        return self.zero
    
    def ropelength_constraint(self):
        down_norm_forces=F.normalize(self.force.now_forces,p=2,dim=2)[:,self.tarp.tarp_info.rope_index,2]
        down_z_coordinate=self.tarp.vertices[:,self.tarp.tarp_info.D,2].clone().detach()
        value=-down_norm_forces-down_z_coordinate/self.tarp.tarp_info.Lmax
        minth=torch.min(value).clone().detach()
        minth=torch.min(minth,torch.zeros_like(minth)+1e-3)
        #theta=torch.min(value.clone().detach(),0.0)-1.0e-6
        #print(self.logelp(value-minth+1e-5,2e-3-minth))
        return self.logelp(value-minth+1e-5,2e-3-minth).sum()
        return self.clampingX2(-down_norm_forces-down_z_coordinate/self.tarp.tarp_info.Lmax,1.0e-2).sum()
        return self.logelp(-down_norm_forces-down_z_coordinate/self.tarp.tarp_info.Lmax,1e-3).sum()
        return self.zero
    
    def loss_evaluation(self):
        self.fixed_loss=self.fixed_constraint()*params.fixed_weight if params.fixed_cons else self.zero
        #yanyisheshou
        self.ropelength_loss=self.ropelength_constraint()*params.rope_weight if params.rope_cons & (params.opt_method=='manu') else self.zero
        #self.ropelength_loss=self.ropelength_constraint()*params.rope_weight if params.rope_cons else self.zero
        if torch.any(torch.isnan(self.ropelength_loss)):
            print('ropelength nan')
        self.geometry_loss=self.fixed_loss+self.ropelength_loss
        return self.geometry_loss
    
    def test_validity(self):
        ForceMaxCondition=lambda f: 1.0-torch.sum(f**2,dim=2)/self.tarp.tarp_info.Fmax**2<params.nume_error
        # RopeLengthCondition=lambda f: -F.normalize(f,p=2,dim=2)[:,self.tarp.tarp_info.rope_index,2]-\
        #                                        self.tarp.vertices[:,self.tarp.tarp_info.D,2]/self.tarp.tarp_info.Lmax<params.nume_error
        self.force.compute_now_forces()
        return ~torch.any(ForceMaxCondition(self.force.now_forces))

    def search(self,cons,condition,info):
        #return True
        itertimes=0
        while cons:
            itertimes=itertimes+1
            if itertimes>200:
                print(info+' locking')
                self.deltaforce=0.0*self.deltaforce
                return False
            
            #now_forces=self.force.force+torch.bmm(self.force.transform,self.force.force_last_displace+self.deltaforce)
            now_forces=torch.bmm(self.force.transform,self.force.force_dif+self.force.force_last_displace+self.deltaforce)
            size=now_forces.shape[1]-1
            now_forces[:,size,2]=now_forces[:,size,2]+self.tarp.tarp_info.mass*self.tarp.tarp_info.g
            
            c=condition(now_forces)
            if torch.any(c):
                self.deltaforce=torch.where(c,0.8*self.deltaforce,self.deltaforce)
            else:
                break
        print(info,itertimes)
        return True
                
    
    def linesearch(self):
        self.deltaforce=self.force.force_displace-self.force.force_last_displace
        if torch.any(torch.isnan(self.deltaforce)):
            print('there is nan after loss step')
            exit(0)

        size=self.deltaforce.shape[1]
        Process=lambda condition: (condition[:,0:size]|condition[:,1:size+1]).reshape(size,1).repeat(1,1,3)
        ForceMaxCondition=lambda f: Process(1.0-torch.sum(f**2,dim=2)/self.tarp.tarp_info.Fmax**2<params.nume_error)
        RopeLengthCondition=lambda f: Process((f[:,:,2]<0.0) & (-F.normalize(f,p=2,dim=2)[:,:,2]\
                            -self.tarp.vertices[:,self.tarp.tarp_info.boundary_index,2]/self.tarp.tarp_info.Lmax<params.nume_error))
        self.search(params.fmax_cons,ForceMaxCondition,'fmax')
        #self.search(params.rope_cons,RopeLengthCondition,'ropelength')
        self.force.force_displace.data=self.force.force_last_displace+self.deltaforce



        
