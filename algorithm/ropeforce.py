import torch
import numpy as np
import algorithm.tool as tool


class RopeForce:
    def __init__(self,vertices,faces,tarp_info,boundary_index,params):
        
        batch_size=vertices.shape[0]
        """ uniquesize=boundary_index.shape[0]-1
        self.transform=torch.zeros((batch_size,uniquesize+1,uniquesize)).cuda()
        for i in range(uniquesize):
            self.transform[:,i,i]=1.0
            self.transform[:,i+1,i]=-1.0 """
        #transform[:,uniquesize,:]=-1.0

        self.tarp_info=tarp_info
        self.mg=tarp_info.G[0,0,:]*vertices.shape[1]
        if self.tarp_info.Fmax<params.force_delay:
            print('too big force delay')
            exit(0)
        self.elp=tarp_info.elp
        self.l1_xi=params.l1_xi
        self.params=params
        self.weight=0
        self.boundary_dir=0
        self.update_boudary_dir(vertices)
        self.fmax_loss=0
        self.fdir_loss=0
        self.fnorm1_loss=0
        self.global_balance_loss=0
        self.local_balance_loss=0
        self.force=self.get_init_force(vertices,boundary_index)
        self.nf=self.force.shape[1]
        self.boundary_index=boundary_index

        self.nv=vertices.shape[1]
        adj=np.zeros([1,self.nv,self.nv]).astype(np.float32)
        faces_numpy=faces[0].cpu()
        adj[0,faces_numpy[:,0],faces_numpy[:,1]]=1
        adj[0,faces_numpy[:,1],faces_numpy[:,0]]=1
        adj[0,faces_numpy[:,1],faces_numpy[:,2]]=1
        adj[0,faces_numpy[:,2],faces_numpy[:,1]]=1
        adj[0,faces_numpy[:,2],faces_numpy[:,0]]=1
        adj[0,faces_numpy[:,0],faces_numpy[:,2]]=1
        self.adj=torch.from_numpy(adj).cuda()

        self.ori_len=torch.sqrt( (vertices[:,:,0]-vertices[:,:,0].reshape(batch_size,self.nv,1))**2
                                +(vertices[:,:,1]-vertices[:,:,1].reshape(batch_size,self.nv,1))**2
                                +(vertices[:,:,2]-vertices[:,:,2].reshape(batch_size,self.nv,1))**2
                                +torch.eye(self.nv).unsqueeze(dim=0).cuda())
    
    def get_init_force(self,vertices,boundary_index):
        val=tool.force_SOCP(vertices[0],boundary_index,self.tarp_info.CI,
                            (self.tarp_info.mass*self.tarp_info.g/boundary_index.shape[0]).cpu().numpy()).reshape(boundary_index.shape[0],2)
        batch_size=vertices.shape[0]
        f=np.zeros((batch_size,boundary_index.shape[0],3))
        #print(val[:,0].sum(),val[:,1].sum())
        f[0,:,0]=val[:,0]
        f[0,:,1]=val[:,1]

        average_weight=self.tarp_info.mass*self.tarp_info.g/boundary_index.shape[0]
        f[:,:,2]=average_weight.cpu().numpy()
        return torch.from_numpy(f.astype(np.float32)).cuda()
        
    def now_force(self,force_displace):
        return force_displace[:,-self.nf:,:]
    
    def update_boudary_dir(self,vertices):
        self.boundary_dir=vertices[:,self.tarp_info.C,:]-vertices[:,self.tarp_info.CI,:]
        self.boundary_dir[:,:,2]=0.0
        self.boundary_dir=self.boundary_dir/torch.norm(self.boundary_dir,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3)

    def update_weight(self,force_displace):
        self.weight=torch.sqrt((self.now_force(force_displace)**2).sum(dim=2)+self.params.l1_epsilon).clone().detach()
        self.weight=torch.pow(self.weight+self.l1_xi,self.params.l1_eta-1.0)
        self.l1_xi=self.l1_xi*self.params.l1_rho

    def logelp(self,x,elp=0.0):
        elp=max(elp,self.elp)
        return torch.where(x<elp,-(x/elp-1.0)**2*torch.log(x/elp),0.0)
    
    def FmaxConstraint(self,forces):
        f2=(forces**2).sum(dim=2)
        return self.logelp(self.tarp_info.Fmax**2-f2,self.tarp_info.Fmax*self.params.force_delay*2-self.params.force_delay**2).sum()

    def FdirConstraint(self,forces):
        forces_mag=torch.norm(forces,p=2,dim=2).unsqueeze(dim=2).repeat(1,1,3).clone().detach()
        innerproduct=torch.sum(torch.where(forces_mag>self.params.l1_epsilon,self.boundary_dir*forces/forces_mag,1.0),dim=2)
        return self.logelp(innerproduct+1.0,self.params.cosine_delay+1.0).sum()
    
    def FNorm1(self,forces):
        force_magnitude=torch.sqrt((forces**2).sum(dim=2)+self.params.l1_epsilon)
        return torch.sum(force_magnitude*self.weight)
    
    def GlobalBalance(self,forces):
        return torch.sum((forces[0].t().sum(dim=1)+self.mg)**2)
    
    def LocalBalance(self,vf,forces):
        batch_size=vf.shape[0]
        vertices=vf[:,0:self.nv,:]
        newlen=torch.sqrt( (vertices[:,:,0]-vertices[:,:,0].reshape(batch_size,self.nv,1))**2
                          +(vertices[:,:,1]-vertices[:,:,1].reshape(batch_size,self.nv,1))**2
                          +(vertices[:,:,2]-vertices[:,:,2].reshape(batch_size,self.nv,1))**2
                          +torch.eye(self.nv).unsqueeze(dim=0).cuda())
        weight=self.tarp_info.k*(newlen-self.ori_len)/newlen*self.adj
        r,c=np.diag_indices(weight.size(1))
        weight[0,r,c]=-weight[0].sum(1)
        x=torch.matmul(weight,vertices)
        x[:,self.boundary_index,:]=x[:,self.boundary_index,:]+forces
        x[:,:,2]=x[:,:,2]-self.tarp_info.mass*self.tarp_info.g/self.nv
        return torch.sum(x**2)
    
    """ def linesearch(self,vertices):
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
        self.force_last_displace=self.force_displace.clone().detach() """

    def loss_evaluation(self,force_displace):
        forces=self.now_force(force_displace)
        zero=torch.tensor([0.0]).cuda()
        self.fmax_loss=self.FmaxConstraint(forces)*self.params.fmax_weight if self.params.fmax_cons else zero
        self.fdir_loss=self.FdirConstraint(forces)*self.params.fdir_weight if self.params.fdir_cons else zero
        self.fnorm1_loss=self.FNorm1(forces)*self.params.fnorm1_weight if self.params.fnorm1_cons else zero
        self.global_balance_loss=self.GlobalBalance(forces)*self.params.rf_weight*0.1
        self.local_balance_loss=self.LocalBalance(force_displace,forces)*0.01
        return self.fmax_loss+self.fdir_loss+self.fnorm1_loss+self.global_balance_loss+self.local_balance_loss
