import torch
import torch.nn as nn
import numpy as np

import algorithm.tarp_info as TI
import algorithm.tool as tool
import os

params=TI.tarp_params()

class ExternalForce(nn.Module):
    def __init__(self,vertices,tarp_info,boundary_index):
        super(ExternalForce,self).__init__()
        batch_size=vertices.shape[0]
        f=torch.zeros(batch_size,tarp_info.C.size(0)-1,3).cuda().double()
        """ f=torch.tensor([[[   9.5,  13.09,   -10.0],[   -9.5,    13.09,  -10.0 ],
                             [  15.4,    -5.0,  -10.0],[   -15.4,     -5.0,   -10.0],
                             [   0.0,    -16.18,   -10.0],
                             [ 0.0,  20.0,  18.82 ],
                             [  19.0,  6.18,    18.82],[  -19.0,     6.18,   18.82 ],
                             [  11.8,  -16.18,  18.82],[  -11.8,   -16.18,  18.82 ]
                             ]]).cuda().double()  """
        f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ -17.0,   17.50,  -19.0 ],[  17.0,   17.50,  -19.0 ],
                             [   0.0,  30.0,  -19.0 ],[   0.0, -30.0,  -19.0 ],
                             [ 20.0,    0.0,   79.05],[-20.0,    0.0,   79.05]]]).cuda().double()
        """ f=torch.tensor([[[ -17.0,  -17.50,  -19.0 ],[  17.0,  -17.50,  -19.0 ],
                             [ 17.0,   17.50,  -19.0 ],[  -17.0,   17.50,  -19.0 ],
                             [ -20.0,    0.0,   60.05],[20.0,    0.0,   60.05]]]).cuda().double() """
        """ f=torch.tensor([[[ -17.32,  -10.0,  -20.0 ],[  17.32,  -10.0,  -20.0 ],
                             [ 0.0,    20.0,   84.1]]]).cuda().double() """
        
        
        """ #f[:,:,0:2]=0
        #f[:,:,2]=44.1/8
        f[:,:,2]=(60-44.1)/(boundary_index.shape[0]-2)
        f[:,2,2]=3
        f[:,3,2]=3
        #f[0,0,0]=f[0,0,0]+0.0001
        #f[0,-1,0]=f[0,-1,0]-0.0001
        rrr=np.array([0,1,4,5,56,156,2,205])
        fo=torch.zeros((batch_size,boundary_index.shape[0],3)).cuda().double()
        fo[0,rrr,:]=f[0,0:8,:]
        f=fo
        #data=np.loadtxt(args.info_path,dtype=np.float64) """
        f=self.get_init_force(vertices,tarp_info,boundary_index)


        """ er=torch.from_numpy(np.loadtxt(os.path.join(params.data_dir,'results/sparseresult/last_force.txt'),dtype=np.float64)).cuda()
        f=er.reshape(1,206,3) """

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
        """ self.theta2=0.04
        self.dtheta=0.02
        self.dF=2.0

        self.alpha=0.2
        self.lamda=1e-3
        self.beta=1.0 """
        self.l1_xi=params.l1_xi
        self.l1_eta=params.l1_eta
        self.l1_rho=params.l1_rho
        self.l1_epsilon=params.l1_epsilon
        self.l1_alpha=params.l1_alpha
        self.l1_beta=params.l1_beta

        self.weight=1
        self.boundary_dir=0
        self.update_boudary_dir(vertices)
        self.fmax_loss=0
        self.fdir_loss=0
        self.fnorm1_loss=0
        #self.update_weight()
    
    def get_init_force(self,vertices,tarp_info,boundary_index):
        val=tool.force_SOCP(vertices[0],boundary_index,tarp_info.CI,
                            (tarp_info.mass*tarp_info.g/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        f=np.zeros((batch_size,boundary_index.shape[0],3))
        #print(val[:,0].sum(),val[:,1].sum())
        f[0,:,0]=val[:,0]
        f[0,:,1]=val[:,1]
        """ f[0,::2,0]=val[:,0]*0.5
        f[0,1::2,0]=val[:,0]*0.5
        f[0,::2,1]=val[:,1]*0.5
        f[0,1::2,1]=val[:,1]*0.5 """

        """ #cut the boundary to several segments and set upward and downward initial force separately
        segment_num=8 
        upward=np.array([]).astype(int)
        downward=np.array([]).astype(int)
        for i in range(segment_num):
            start=np.floor(boundary_index.shape[0]*i/segment_num).astype(int)
            end=np.floor(boundary_index.shape[0]*(i+1)/segment_num).astype(int)
            if i%2==0:
                upward=np.append(upward,boundary_index[start:end].cpu().numpy())
            else:
                downward=np.append(downward,boundary_index[start:end].cpu().numpy())
        
        average_weight=tarp_info.mass*tarp_info.g/boundary_index.shape[0]
        f[:,downward,2]=-average_weight.cpu().numpy()
        f[:,upward,2]=(tarp_info.mass*tarp_info.g+average_weight*downward.shape[0]).cpu().numpy()/upward.shape[0]
        #print(f[:,:,0].sum(),f[:,:,1].sum(),f[:,:,2].sum())
        return torch.from_numpy(f).cuda() """

        average_weight=(tarp_info.mass*tarp_info.g/boundary_index.shape[0]).cpu().numpy()
        f[:,:,2]=average_weight
        """ f[:,::2,2]=average_weight*2
        f[:,1::2,2]=-average_weight """
        return torch.from_numpy(f).cuda()
        
    def now_force(self):
        return self.force+torch.bmm(self.transform,self.force_displace)
    
    def update_boudary_dir(self,vertices):
        self.boundary_dir=vertices[:,self.tarp_info.C,:]-vertices[:,self.tarp_info.CI,:]
        self.boundary_dir[:,:,2]=0.0
        self.boundary_dir=self.boundary_dir/torch.norm(self.boundary_dir,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FmaxConstraint(self,forces):
        f2=(forces**2).sum(dim=2)
        return self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*params.force_delay*2-params.force_delay**2).sum()

    def FdirConstraint(self,forces):
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
        return (force_magnitude*self.weight).sum()
    
    def linesearch(self,vertices):
        if params.fmax_cons+params.fdir_cons==0:
            return
        if params.fdir_cons:
            self.update_boudary_dir(vertices)
        deltaforce=self.force_displace-self.force_last_displace
        itertimes=0
        while 1:
            itertimes=itertimes+1
            if itertimes>500:
                if (deltaforce**2).sum()<params.grad_error:
                    print('locking')
                    deltaforce=0.0*deltaforce
                    break
            
            if params.fmax_cons:
                forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
                f2=(forces**2).sum(dim=2)
                ForceMaxCondition=((self.tarp_info.Fmax**2)<(f2*(1.0+params.nume_error)))
                if ForceMaxCondition.sum()!=0:
                    deltaforce=0.8*deltaforce
                    continue
            if params.fdir_cons:
                forces=self.force+torch.bmm(self.transform,self.force_last_displace+deltaforce)
                forces_mag=torch.norm(forces,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3).clone().detach()
                innerproduct=torch.sum(torch.where(forces_mag>0.95*self.l1_epsilon,self.boundary_dir*forces/forces_mag,1.0),dim=2)
                ForceDirCondition=(innerproduct+1.0<params.nume_error)
                if ForceDirCondition.sum()!=0:
                    deltaforce=0.8*deltaforce
                    continue
            break

        self.force_displace.data=self.force_last_displace+deltaforce
    
    def record_last_displace(self):
        self.force_last_displace=self.force_displace.clone().detach()

    def forward(self):
        forces=self.force+torch.bmm(self.transform,self.force_displace)
        zero=torch.tensor([0]).cuda()
        self.fmax_loss=self.FmaxConstraint(forces)*params.fmax_weight if params.fmax_cons else zero
        self.fdir_loss=self.FdirConstraint(forces)*params.fdir_weight if params.fdir_cons else zero
        self.fnorm1_loss=self.FNorm1(forces)*params.fnorm1_weight if params.fnorm1_cons else zero
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
