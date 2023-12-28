from typing import Any
import cvxpy as cp
import numpy as np

def force_SOCP(vertices,index,center_id,reference_norm):
    dir=(vertices[index,0:2]-vertices[center_id,0:2]).cpu().numpy()
    dir_norm=np.linalg.norm(dir,axis=1,keepdims=True)
    dir=np.divide(dir,dir_norm).flatten()
    
    #dir[:]=1.0

    m=index.cpu().numpy().shape[0]
    n=m*2
    p=2
    

    A=[]
    d=[]
    for i in range(m):
        A.append(np.zeros((2,n)))
        A[i][0,i*2]=1.0
        A[i][1,i*2+1]=1.0
        d.append(np.array([reference_norm]))
    F=np.zeros((p,n))
    g=np.zeros(p)
    for i in range(n):
        if (i%2)==0:
            F[0,i]=1
        else:
            F[1,i]=1

    x=cp.Variable(n)
    soc_constraints=[
        cp.SOC(d[i],A[i]@x) for i in range(m)
        ]
    prob=cp.Problem(cp.Maximize(dir.T@x),
                    soc_constraints+[F@x==g])
    prob.solve()
    return x.value

import torch
import openmesh
def get_mesh_boundary(mesh_dir):
    mesh=openmesh.read_trimesh(mesh_dir)
    hl=0
    v_index=np.array([])
    for hl_iter in mesh.halfedges():
        if mesh.is_boundary(hl_iter):
            hl=hl_iter
            break
    hl_iter=mesh.next_halfedge_handle(hl)
    v_index=np.append(v_index,mesh.to_vertex_handle(hl_iter).idx())
    while(hl_iter!=hl):
        hl_iter=mesh.next_halfedge_handle(hl_iter)
        v_index=np.append(v_index,mesh.to_vertex_handle(hl_iter).idx())
    return torch.from_numpy(v_index.astype(int)).cuda()

def get_adj(mesh_dir):
    mesh=openmesh.read_trimesh(mesh_dir)
    adjcols=9
    adj=np.zeros((mesh.n_vertices(),adjcols))
    len=np.zeros((mesh.n_vertices(),adjcols-1))
    for v in mesh.vertices():
        if mesh.valence(v)>adjcols-1:
            print('adjcols is too smalle')
            exit(0)
        adj[v.idx(),adjcols-1]=mesh.valence(v)
        i=0
        """ for vv in mesh.vv(v):
            adj[v.idx(),i]=vv.idx()
            len[v.idx(),i]=(mesh.point(v)-mesh.point(vv)).Norm()
            i=i+1 """
        for voh in mesh.voh(v):
            adj[v.idx(),i]=mesh.to_vertex_handle(voh).idx()
            len[v.idx(),i]=mesh.calc_edge_length(voh)
            i=i+1
    return torch.from_numpy(adj.astype(np.int32)).cuda(),torch.from_numpy(len.astype(np.float32)).cuda()

def get_v2f_id(nv,boundary_index):
    v2f_id=torch.IntTensor(nv).cuda()
    for i in range(nv):
        v2f_id[i]=-1
    it=0
    for i in range(boundary_index.shape[0]):
        v2f_id[boundary_index[i]]=it
        it=it+1
    return v2f_id

import yaml
import configargparse
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
def read_params():
    params=configargparse.ArgumentParser().parse_args()
    params.config='algorithm/params.yaml'
    with open(params.config,'r') as stream:
        meta_params=yaml.safe_load(stream)
    
    meta_params['current_dir']=current_dir
    meta_params['data_dir']=data_dir

    meta_params['template_mesh']=os.path.join(data_dir,meta_params['template_mesh'])
    meta_params['image']=os.path.join(data_dir,meta_params['image'])
    meta_params['info_path']=os.path.join(data_dir,meta_params['info_path'])
    meta_params['output_dir']=os.path.join(data_dir,meta_params['output_dir'])

    meta_params['force_file']=os.path.join(data_dir,meta_params['force_file'])
    meta_params['forcedis_file']=os.path.join(data_dir,meta_params['forcedis_file'])
    meta_params['result_mesh']=os.path.join(data_dir,meta_params['result_mesh'])
    return meta_params

import parallel_energy
class py_simulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,force_displace,diff_simulator):
        v_size=int(diff_simulator.v.size/3)
        vertices=diff_simulator.v.reshape(v_size,3).astype(np.float32)
        ctx.jacobi=torch.from_numpy(diff_simulator.jacobi.astype(np.float32)).unsqueeze(dim=0).cuda()
        return torch.from_numpy(vertices).unsqueeze(dim=0).cuda()
    
    @staticmethod
    def backward(ctx,grad_vertices):
        grad_vertices=grad_vertices.reshape(1,1,grad_vertices.size(1)*3)
        force_num=int(ctx.jacobi.size(2)/3)
        grad_force_displace=torch.bmm(grad_vertices,ctx.jacobi).reshape(1,force_num,3)
        return grad_force_displace,None
    
class balance_energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,vf,mg,nv,nf,adj,len,v2f_id,weight):
        ctx.vf=vf
        ctx.nv=nv
        ctx.nf=nf
        ctx.adj=adj
        ctx.len=len
        ctx.v2f_id=v2f_id
        ctx.weight=weight
        ctx.balance_value=torch.zeros(nv*3).cuda()
        parallel_energy.energy_forward(vf,mg,nv,nf,adj,len,v2f_id,ctx.balance_value)
        return torch.sum(ctx.balance_value**2)*weight
    
    @staticmethod
    def backward(ctx,grad_loss):
        vf_grad=torch.zeros((ctx.nv+ctx.nf)*3).cuda()
        parallel_energy.energy_backward(ctx.vf,ctx.nv,ctx.nf,ctx.adj,ctx.len,ctx.v2f_id,ctx.balance_value,vf_grad)
        return vf_grad.reshape(ctx.nv+ctx.nf,3).unsqueeze(dim=0)*ctx.weight,None,None,None,None,None,None,None
        
    

import matplotlib.pyplot as plt
class LossDrawer():
    def __init__(self):
        self.figure_x=[]
        self.figure_fmax_loss=[]
        self.figure_fdir_loss=[]
        self.figure_fnorm1_loss=[]
        self.figure_fglobal_loss=[]
        self.figure_flocal_loss=[]
        self.figure_shadow_loss=[]
        self.figure_total_loss=[]
        plt.ion()
    
    def truncate(self,loss):
        return min(loss.clone().detach().cpu().numpy(),0.1)
    
    def update(self,id,ropeforce,meshrender):
        fmax_loss_cpu=self.truncate(ropeforce.fmax_loss)
        fdir_loss_cpu=self.truncate(ropeforce.fdir_loss)
        fnorm1_loss_cpu=self.truncate(ropeforce.fnorm1_loss)
        fglobal_loss_cpu=self.truncate(ropeforce.global_balance_loss)
        flocal_loss_cpu=self.truncate(ropeforce.local_balance_loss)
        shadow_loss_cpu=self.truncate(meshrender.shadow_loss)
        total_loss_cpu=fmax_loss_cpu+fdir_loss_cpu+fnorm1_loss_cpu+fglobal_loss_cpu+flocal_loss_cpu+shadow_loss_cpu
        self.figure_x.append(id)
        self.figure_fmax_loss.append(fmax_loss_cpu)
        self.figure_fdir_loss.append(fdir_loss_cpu)
        self.figure_fnorm1_loss.append(fnorm1_loss_cpu)
        self.figure_fglobal_loss.append(fglobal_loss_cpu)
        self.figure_flocal_loss.append(flocal_loss_cpu)
        self.figure_shadow_loss.append(shadow_loss_cpu)
        self.figure_total_loss.append(total_loss_cpu)
        print('total,shadow,l1,dir,max,global,local')
        print(total_loss_cpu,shadow_loss_cpu,fnorm1_loss_cpu,fdir_loss_cpu,fmax_loss_cpu,fglobal_loss_cpu,flocal_loss_cpu)
        print(meshrender.shadow_loss)
        print(ropeforce.global_balance_loss)
        print(ropeforce.local_balance_loss)
        plt.clf()
        plt.plot(self.figure_x,self.figure_total_loss,label="total loss",color='red')
        plt.plot(self.figure_x,self.figure_shadow_loss,label="shadow loss",color='lightgreen')
        plt.plot(self.figure_x,self.figure_fnorm1_loss,label="force l1-norm loss",color='peru')
        plt.plot(self.figure_x,self.figure_fdir_loss,label="force direction barrier loss",color="cyan")
        plt.plot(self.figure_x,self.figure_fmax_loss,label="force maximum barrier loss",color="magenta")
        plt.plot(self.figure_x,self.figure_fglobal_loss,label="global balance loss",color='purple')
        plt.plot(self.figure_x,self.figure_flocal_loss,label="local balance loss",color='orange')
        plt.legend(loc="upper right")

    def save_loss(self,result_dir):
        plt.savefig(os.path.join(result_dir, 'loss.png'))

class GUIInfo():
    def __init__(self,vf,faces,boundary_index,nv,nf,params):
        self.nv=nv
        self.nf=nf
        self.vertices=vf[0,0:nv,:].clone().detach().cpu().numpy()
        self.forces=vf[0,-nf:,:].clone().detach().cpu().numpy()
        self.vertices_grad=0
        self.forces_grad=0
        self.vertices_optgrad=0
        self.forces_optgrad=0
        self.faces=faces[0].clone().detach().cpu().numpy()
        self.boundary_index=boundary_index.clone().detach().cpu().numpy()
        self.params=params

    def update(self,vf,vf_grad,opt_grad):
        self.vertices=vf[0,0:self.nv,:].clone().detach().cpu().numpy()
        self.forces=vf[0,-self.nf:,:].clone().detach().cpu().numpy()
        if self.params.use_vertgrad:
            self.vertices_grad=vf_grad[0,0:self.nv,:].clone().detach().cpu().numpy()
        if self.params.use_forcegrad:
            self.forces_grad=vf_grad[0,-self.nf:,:].clone().detach().cpu().numpy()
        if self.params.use_voptgrad:
            self.vertices_optgrad=opt_grad[0,0:self.nv,:].clone().detach().cpu().numpy()
        if self.params.use_foptgrad:
            self.forces_optgrad=opt_grad[0,-self.nf:,:].clone().detach().cpu().numpy()

from datetime import datetime
def get_datetime():
    return datetime.now()

import shutil
def copy_file(file_name,file_dir):
    shutil.copy(file_name,os.path.join(file_dir,os.path.split(file_name)[1]))

def write_data(data,file_dir_name,x3=True):
    np.set_printoptions(threshold=data.shape[0])
    if os.path.exists(file_dir_name)==False:
        os.mknod(file_dir_name)
    file=open(file_dir_name,'w')
    if x3:
        for i in range(data.shape[0]):
            print(data[i,0],data[i,1],data[i,2],file=file)
    else:
        for i in range(data.shape[0]):
            print(data[i],file=file)
    file.close()

def write_readme(info,file_dir_name):
    if os.path.exists(file_dir_name)==False:
        os.mknod(file_dir_name)
    file=open(file_dir_name,'w')
    print(info,file=file)
    file.close()
    
