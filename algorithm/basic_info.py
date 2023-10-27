import torch
import torch.nn as nn
import numpy as np

class basic_info():
    def __init__(self, vertex, faces, data):
        
        batch_size=vertex.size(0)
        self.nv = vertex.size(1)
        self.nf = faces.size(0)
        
        #small coef
        self.register_buffer('elp',torch.tensor(data[0]))
        
        #stiffnesss                  unit: kg/(m * s^2)
        self.register_buffer('k',torch.tensor(data[1]))
        
        #mass
        self.register_buffer('mass',torch.tensor(data[2]))
        
        #gravity                     unit: N
        G=np.zeros([batch_size,self.nv,3]).astype(np.float32)
        G[:,:,2]=-data[2]*data[3]/self.nv
        self.register_buffer('G',torch.from_numpy(G))
        
        #maximum force on the rope   unit: N
        self.register_buffer('Fmax',torch.tensor(data[4]))
        
        #minimum height of the tarp  unit: m
        self.register_buffer('Hmin',torch.tensor(data[5]))
        
        #the height of the sticks    unit: m
        self.register_buffer('H',torch.tensor(data[6]))
        
        #maximum length of the rope  unit: m
        self.register_buffer('Lmax',torch.tensor(data[7]))
        
        #index of mesh center
        self.register_buffer('Center',torch.tensor(int(data[8])))
        
        #maximum radius of the mesh
        self.register_buffer('Rmax',torch.tensor(1.1*torch.sqrt(torch.sum(vertex[:,:,0:2]**2,dim=2).max())))
        
        #vertex that connect with a rope
        self.register_buffer('C0',torch.from_numpy(data[11:11+int(data[9])].astype(int)))
        
        #vertex that connect with a rope and a stick
        self.register_buffer('C1',torch.from_numpy(data[-int(data[10]):].astype(int)))
        
        self.register_buffer('fixed',vertex[:,self.C1,:])
        prr=vertex[:,self.C0,:]
        prr[:,:,2]=2.0
        self.register_buffer('fixedp',prr)
        
        
        #vertically upward direction
        n=np.zeros([batch_size,self.C1.size(0),3]).astype(np.float32)
        n[:,:,2]=1.0
        self.register_buffer('n',torch.from_numpy(n))

    def __init__(self, vertex, faces, data):
        super(HardLoss, self).__init__()
        
        batch_size=vertex.size(0)
        self.nv = vertex.size(1)
        self.nf = faces.size(0)
        self.activated=np.ones([6])
        
        #small coef
        self.register_buffer('elp',torch.tensor(data[0]))
        
        #stiffnesss                  unit: kg/(m * s^2)
        self.register_buffer('k',torch.tensor(data[1]))
        
        #mass
        self.register_buffer('mass',torch.tensor(data[2]))
        
        #gravity                     unit: N
        G=np.zeros([batch_size,self.nv,3]).astype(np.float32)
        G[:,:,2]=-data[2]*data[3]/self.nv
        self.register_buffer('G',torch.from_numpy(G))
        
        #maximum force on the rope   unit: N
        self.register_buffer('Fmax',torch.tensor(data[4]))
        
        #minimum height of the tarp  unit: m
        self.register_buffer('Hmin',torch.tensor(data[5]))
        
        #the height of the sticks    unit: m
        self.register_buffer('H',torch.tensor(data[6]))
        
        #maximum length of the rope  unit: m
        self.register_buffer('Lmax',torch.tensor(data[7]))
        
        #index of mesh center
        self.register_buffer('CI',torch.tensor(int(data[8])))
        
        #maximum radius of the mesh
        self.register_buffer('Rmax',torch.tensor(1.1*torch.sqrt(torch.sum(vertex[:,:,0:2]**2,dim=2).max())))
        
        #vertex that connect with a rope
        self.register_buffer('C0',torch.from_numpy(data[11:11+int(data[9])].astype(int)))
        
        #vertex that connect with a rope and a stick
        self.register_buffer('C1',torch.from_numpy(data[-int(data[10]):].astype(int)))
        
        self.register_buffer('fixed',vertex[:,self.C1,:])
        prr=vertex[:,self.C0,:]
        prr[:,:,2]=2.0
        self.register_buffer('fixedp',prr)
        
        
        #vertically upward direction
        n=np.zeros([batch_size,self.C1.size(0),3]).astype(np.float32)
        n[:,:,2]=1.0
        self.register_buffer('n',torch.from_numpy(n))
        
        
        #incidence matrix
        adj = np.zeros([1,self.nv, self.nv]).astype(np.float32)
        adj[0,faces[:,0],faces[:,1]]=1
        adj[0,faces[:,1],faces[:,0]]=1
        adj[0,faces[:,1],faces[:,2]]=1
        adj[0,faces[:,2],faces[:,1]]=1
        adj[0,faces[:,2],faces[:,0]]=1
        adj[0,faces[:,0],faces[:,2]]=1
        self.register_buffer('adj', torch.from_numpy(adj))
        
        
        #identity matrix
        len=torch.sqrt((vertex[:,:,0]-vertex[:,:,0].reshape(batch_size,self.nv,1))**2
                    +(vertex[:,:,1]-vertex[:,:,1].reshape(batch_size,self.nv,1))**2
                    +(vertex[:,:,2]-vertex[:,:,2].reshape(batch_size,self.nv,1))**2
                        +torch.eye(self.nv).unsqueeze(dim=0).cuda())
        self.register_buffer('len', len)
        
        fa0=faces[:,0].type(torch.long)
        fa1=faces[:,1].type(torch.long)
        fa2=faces[:,2].type(torch.long)
        len01=vertex[0,fa0,:]-vertex[0,fa1,:]
        self.register_buffer('len01',torch.sqrt(torch.sum(len01**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3))
        len12=vertex[0,fa1,:]-vertex[0,fa2,:]
        self.register_buffer('len12',torch.sqrt(torch.sum(len12**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3))
        len20=vertex[0,fa2,:]-vertex[0,fa0,:]
        self.register_buffer('len20',torch.sqrt(torch.sum(len20**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3))
        
    
    def logelp(self,x, elp=0.0):
        if elp>0.0:
            return torch.where(x<elp,-(x-elp)**2*torch.log(x/elp)/(elp**2),0.0)
        else:
            return torch.where(x<self.elp,-(x-self.elp)**2*torch.log(x/self.elp)/(self.elp**2),0.0)
        
    def local_equivalenceloss(self, x, f, activated):
        if activated == False:
            return torch.zeros([1]).cuda()
        batch_size=x.size(0)
        newlen=torch.sqrt((x[:,:,0]-x[:,:,0].reshape(batch_size,self.nv,1))**2
                         +(x[:,:,1]-x[:,:,1].reshape(batch_size,self.nv,1))**2
                         +(x[:,:,2]-x[:,:,2].reshape(batch_size,self.nv,1))**2
                        +torch.eye(self.nv).unsqueeze(dim=0).cuda())
        
       # print('fesf',torch.min((newlen-self.len)*self.adj))
        
        weight=self.k*(newlen-self.len)/newlen*self.adj
        r, c = np.diag_indices(weight.size(1))
        weight[0,r, c] = -weight[0].sum(1)
        x=torch.matmul(weight,x)
        f_temp=f.clone().detach()
        x[:,self.C0,:]=x[:,self.C0,:]+f_temp[:,0:self.C0.size(0),:]
        x[:,self.C1,:]=x[:,self.C1,:]+f_temp[:,-self.C1.size(0):,:]
        #x=x+self.G
        print('xxxxxxxxxxxxx0',x[:,0,:]+self.G[:,0,:])
        print('xxxxxxxxxxxxx1',x[:,1,:]+self.G[:,1,:])
        print('xxxxxxxxxxxxx2',x[:,2,:]+self.G[:,2,:])
        print('xxxxxxxxxxxxx3',x[:,3,:]+self.G[:,3,:])
        print('xxxxxxxxxxxxx4',x[:,4,:]+self.G[:,4,:])
        print('xxxxxxxxxxxxx5',x[:,5,:]+self.G[:,5,:])
        print('xxxxxxxxxxxxx326',x[:,326,:]+self.G[:,326,:])
        print('xxxxxxxxxxxxx212',x[:,212,:]+self.G[:,212,:])
        print('xxxxxxxxxxxxx170',x[:,170,:]+self.G[:,170,:])
        
        print('local',((x+self.G)**2).sum())
        return torch.zeros([1]).cuda()
        return ((x+self.G)**2).sum()#-(x[:,self.C1,:]**2).sum()
    
    def global_equivalenceloss(self,f,activated):
        if activated == False:
            return torch.zeros([1]).cuda()
        #print(torch.sum(f,dim=1))
        return ((torch.sum(f,dim=1)+torch.sum(self.G,dim=1))**2).sum()
    
    def fixedloss(self,x,activated):
        if activated == False:
            return torch.zeros([1]).cuda()
        return ((x[:,self.C1,2]-self.H)**2).sum()#+(x[:,self.CI,0:2]**2).sum()
        return ((x[:,self.C1,:]-self.fixed)**2).sum()#+1e-3*((x[:,:,2]**2).sum())#+((x[:,self.C0,2]-2.0)**2).sum()#+(x[:,self.CI,0:2]**2).sum()
    
    def forceconstraintloss(self,x,f,activated):
        if activated == False:
            return torch.zeros([1]).cuda()
        batch_size=x.size(0)
        F0value=torch.sqrt(torch.sum(f[:,0:self.C0.size(0),:]**2,dim=2))
        F1value=torch.sum(torch.cross(f[:,-self.C1.size(0):,:],self.n)**2,dim=2)
        y=x.clone().detach()
        C0_dir=(y[:,self.C0,:]-y[:,self.CI,:])
        C1_dir=(y[:,self.C1,:]-y[:,self.CI,:])
        #print('0000000',self.Fmax-F0value)
        #print('11111111',-f[:,0:self.C0.size(0),2]/F0value-x[:,self.C0,2]/self.Lmax)
        #print('2222222222',f[:,-self.C1.size(0):,2])
        #print('333333',(self.Lmax**2-x[:,self.C1,2]**2)/self.Lmax**2-F1value/self.Fmax**2)
        #print('44333333334444',torch.sum(f[:,0:self.C0.size(0),:]*C0_dir,dim=2),torch.sqrt(torch.sum(C0_dir**2,dim=2)))
        #print('44444',torch.sum(f[:,0:self.C0.size(0),:]*C0_dir,dim=2)/torch.sqrt(torch.sum(C0_dir**2,dim=2)))
        #print('555555555',torch.sum(f[:,-self.C1.size(0):,:]*C1_dir,dim=2)/torch.sqrt(torch.sum(C1_dir**2,dim=2)))
        return  self.logelp(-f[:,0:self.C0.size(0),2]).sum()\
            +self.logelp(f[:,-self.C1.size(0):,2]).sum()\
            +self.logelp(torch.sum(f[:,0:self.C0.size(0),:]*C0_dir,dim=2)/torch.sqrt(torch.sum(C0_dir**2,dim=2))).sum()\
            +self.logelp(torch.sum(f[:,-self.C1.size(0):,:]*C1_dir,dim=2)/torch.sqrt(torch.sum(C1_dir**2,dim=2))).sum()\
            +self.logelp(self.Fmax-F0value).sum()\
            +self.logelp(self.Fmax-torch.sqrt(torch.sum(f[:,-self.C1.size(0):,:]**2,dim=2))).sum()
        return  self.logelp(self.Fmax/F0value-1.0).sum()\
            +self.logelp(-f[:,0:self.C0.size(0),2]*self.Lmax/(F0value*x[:,self.C0,2])-1.0).sum()\
            +self.logelp(f[:,-self.C1.size(0):,2]).sum()\
            +self.logelp((self.Lmax**2-x[:,self.C1,2]**2)*self.Fmax**2/(self.Lmax**2*F1value)-1.0).sum()\
            +self.logelp(torch.sum(f[:,0:self.C0.size(0),:]*C0_dir,dim=2)/torch.sqrt(torch.sum(C0_dir**2,dim=2))).sum()\
            +self.logelp(torch.sum(f[:,-self.C1.size(0):,:]*C1_dir,dim=2)/torch.sqrt(torch.sum(C1_dir**2,dim=2))).sum()
        
    def positionloss(self,x,activated):
        if activated == False:
            return torch.zeros([1]).cuda()
        #print('4444444',torch.any(x[:,:,2]-self.Hmin)<=0)
        #print('55555555',torch.any(self.Rmax**2-torch.sum(x[:,:,0:2]**2,dim=2))<=0)
        return self.logelp(x[:,:,2]-self.Hmin).sum()
            #+self.logelp(self.Rmax**2-torch.sum(x[:,:,0:2]**2,dim=2)).sum()
    
    def length_loss(self,x,f,x0,x1,faces,activated,dt):
        if activated==False:
            return torch.zeros([1]).cuda()
        '''batch_size=x.size(0)
        newlen=torch.sqrt((x[:,:,0]-x[:,:,0].reshape(batch_size,self.nv,1))**2
                         +(x[:,:,1]-x[:,:,1].reshape(batch_size,self.nv,1))**2
                         +(x[:,:,2]-x[:,:,2].reshape(batch_size,self.nv,1))**2
                        +torch.eye(self.nv).unsqueeze(dim=0).cuda())
        delta=-self.G[:,:,2]*0.01/self.k
        return self.logelp((newlen-self.len)*self.adj+1e-5,1e-5)'''
        batch_size=x.size(0)
        y=x.clone().detach()
        fa0=faces[:,0].type(torch.long)
        fa1=faces[:,1].type(torch.long)
        fa2=faces[:,2].type(torch.long)
        y01=y[0,fa0,:]-y[0,fa1,:]
        y01=y01/torch.sqrt(torch.sum(y01**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3)
        y12=y[0,fa1,:]-y[0,fa2,:]
        y12=y12/torch.sqrt(torch.sum(y12**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3)
        y20=y[0,fa2,:]-y[0,fa0,:]
        y20=y20/torch.sqrt(torch.sum(y20**2,dim=1)).unsqueeze(dim=0).t().repeat(1,3)
        x01=x[0,fa0,:]-x[0,fa1,:]
        x12=x[0,fa1,:]-x[0,fa2,:]
        x20=x[0,fa2,:]-x[0,fa0,:]
        ff=f.clone().detach()
        
        #x[:,self.C0,:]=x[:,self.C0,:]-ff[:,0:self.C0.size(0),:]
        #x[:,self.C1,:]=x[:,self.C1,:]-ff[:,-self.C1.size(0):,:]
        
        return 0.5*self.mass*((x-x0)**2).sum()-dt*dt*(self.G*x).sum()\
            -dt*dt*((x[:,self.C0,:]*ff[:,0:self.C0.size(0),:]).sum()+(x[:,self.C1,:]*ff[:,-self.C1.size(0):,:]).sum())\
                +0.5*self.k*dt*dt*(((x01-y01*self.len01)**2).sum()
                                  +((x12-y12*self.len12)**2).sum()
                                  +((x20-y20*self.len20)**2).sum())#*(torch.matmul(weight,x)**2).sum()
    
    def forward(self,x,f,x0,x1,faces,dt):
        print("each one",self.local_equivalenceloss(x,f,self.activated[0]).cpu().detach().numpy(),
              self.global_equivalenceloss(f,self.activated[1]).cpu().detach().numpy(),
              self.fixedloss(x,self.activated[2]).cpu().detach().numpy(),
              self.forceconstraintloss(x,f,self.activated[3]).cpu().detach().numpy(),
              self.positionloss(x,self.activated[4]).cpu().detach().numpy(),
              self.length_loss(x,f,x0,x1,faces,self.activated[5],dt).cpu().detach().numpy())
        return 1e2*self.local_equivalenceloss(x,f,self.activated[0])\
            +1e2*self.global_equivalenceloss(f,self.activated[1])\
            +1e5*self.fixedloss(x,self.activated[2])\
            +self.forceconstraintloss(x,f,self.activated[3])\
            +self.positionloss(x,self.activated[4])\
            +1e3*self.length_loss(x,f,x0,x1,faces,self.activated[5],dt)