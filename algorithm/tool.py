import cvxpy as cp
import numpy as np

def shadow_intersection(all_shadow_image):
    return torch.min(all_shadow_image,dim=0)[0]

    num=all_shadow_image.shape[0]
    shadow_image=all_shadow_image[0]
    if num>1:
        for i in range(1,num-1):
            shadow_image=all_shadow_image[i]*shadow_image
    return shadow_image

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
import torch.nn.functional as F
import math
import openmesh
def calc_z_axis(mesh,hl):
    hl=mesh.opposite_halfedge_handle(hl)
    v0=mesh.point(mesh.from_vertex_handle(hl))
    v1=mesh.point(mesh.to_vertex_handle(hl))
    v2=mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(hl)))
    v01=F.normalize(torch.tensor([v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2]]).cuda(),p=2,dim=0)
    v02=F.normalize(torch.tensor([v2[0]-v0[0],v2[1]-v0[1],v2[2]-v0[2]]).cuda(),p=2,dim=0)
    return F.normalize(v01.cross(v02),p=2,dim=0)

def calc_angle_weight(v0,v1,v2,z):
    v10=F.normalize(torch.tensor([v0[0]-v1[0],v0[1]-v1[1],v0[2]-v1[2]]).cuda(),p=2,dim=0)
    v12=F.normalize(torch.tensor([v2[0]-v1[0],v2[1]-v1[1],v2[2]-v1[2]]).cuda(),p=2,dim=0)
    angle=torch.atan2(v10.cross(v12).dot(z),v10.dot(v12))
    if angle<0:
        angle=angle+math.pi*2
    return angle.cpu()/math.pi
    
def get_mesh_boundary(mesh_dir):
    mesh=openmesh.read_trimesh(mesh_dir)
    hl=0
    v_index=np.array([])
    v_weight=np.array([])
    for hl_iter in mesh.halfedges():
        if mesh.is_boundary(hl_iter):
            hl=hl_iter
            break
    z=calc_z_axis(mesh,hl)
    hl_iter=mesh.next_halfedge_handle(hl)
    v_index=np.append(v_index,mesh.to_vertex_handle(hl_iter).idx())
    v0=mesh.point(mesh.from_vertex_handle(hl_iter))
    v1=mesh.point(mesh.to_vertex_handle(hl_iter))
    v2=mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(hl_iter)))
    v_weight=np.append(v_weight,calc_angle_weight(v0,v1,v2,z))

    while(hl_iter!=hl):
        v0=mesh.point(mesh.to_vertex_handle(hl_iter))
        hl_iter=mesh.next_halfedge_handle(hl_iter)
        v1=mesh.point(mesh.to_vertex_handle(hl_iter))
        v2=mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(hl_iter)))
        v_index=np.append(v_index,mesh.to_vertex_handle(hl_iter).idx())
        v_weight=np.append(v_weight,calc_angle_weight(v0,v1,v2,z))
    return torch.from_numpy(v_index.astype(int)).cuda(),torch.from_numpy(v_weight).cuda()


    index=np.array([])
    for v in mesh.vertices():
        if mesh.is_boundary(v):
            index=np.append(index,v.idx())
    """ index=np.delete(index,np.arange(1,index.size,2))
    index=np.delete(index,np.arange(1,index.size,2))
    index=np.delete(index,np.arange(1,index.size,2))
    index=np.delete(index,np.arange(1,index.size,2)) """
    """ index[3]=205
    index[205]=3 """
    return torch.from_numpy(index.astype(int)).cuda()

def suppleset(a,b):
    c=a
    for i in b:
        if i in c:
            mask=c!=i
            c=torch.masked_select(c,mask)
    return c

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
    meta_params['example_dir']=os.path.join(data_dir,'example')
    
    meta_params['image']=os.path.join(data_dir,meta_params['image'])
    meta_params['output_dir']=os.path.join(data_dir,meta_params['output_dir'])

    meta_params['force_file']=os.path.join(data_dir,meta_params['force_file'])
    meta_params['forcedis_file']=os.path.join(data_dir,meta_params['forcedis_file'])
    meta_params['result_mesh']=os.path.join(data_dir,meta_params['result_mesh'])
    return meta_params


class py_simulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,force_displace,diff_simulator,jacobi):
        v_size=int(diff_simulator.v.size/3)
        vertices=diff_simulator.v.reshape(v_size,3).astype(np.float32)
        #ctx.jacobi=torch.from_numpy(diff_simulator.jacobi.astype(np.float32)).unsqueeze(dim=0).cuda()
        ctx.jacobi=jacobi
        return torch.from_numpy(vertices).unsqueeze(dim=0).cuda()
    
    @staticmethod
    def backward(ctx,grad_vertices):
        grad_vertices=grad_vertices.reshape(1,1,grad_vertices.size(1)*3)
        force_num=int(ctx.jacobi.size(2)/3)
        grad_force_displace=torch.bmm(grad_vertices,ctx.jacobi).reshape(1,force_num,3)
        return grad_force_displace,None,None

import cupy
from cupyx.scipy.sparse import coo_matrix,csr_matrix, linalg as sla
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import time
class jacobi_solver():
    def __init__(self):
        self.right=0
        self.lu_solver=0

    def compute_right(self,r):
        self.right=cupy.zeros((r.shape[0],r.shape[1]))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                self.right[i,j]=r[i,j]

    def lu(self,row,col,val):
        curow=cupy.asarray(row)
        cucol=cupy.asarray(col)
        cuval=cupy.asarray(val.astype(np.float32))
        left=coo_matrix((cuval,(curow,cucol)),shape=(self.right.shape[0],self.right.shape[0]))
        self.lu_solver=sla.splu(left)

    def solve(self):
        x=self.lu_solver.solve(self.right)
        return from_dlpack(x.toDlpack()).unsqueeze(dim=0)


def compute_jacobi(row,col,val,right):
    curow=cupy.asarray(row)
    cucol=cupy.asarray(col)
    cuval=cupy.asarray(val.astype(np.float32))
    left=coo_matrix((cuval,(curow,cucol)),shape=(right.shape[0],right.shape[0]))
    lu=sla.splu(left)
    x=lu.solve(right)
    return from_dlpack(x.toDlpack()).unsqueeze(dim=0)

def compute_jacobiright(right):
    wideright=cupy.zeros((right.shape[0],right.shape[1]))
    for i in range(right.shape[0]):
        for j in range(right.shape[1]):
            wideright[i,j]=right[i,j]
    return wideright

from datetime import datetime
def get_datetime():
    return str(datetime.now()).replace(' ','-').replace(':','-').replace('.','-')

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


import matplotlib.pyplot as plt
class LossDrawer():
    def __init__(self,params):
        self.figure_x=[]
        self.figure_fmax_loss=[]
        self.figure_fdir_loss=[]
        self.figure_fnorm1_loss=[]
        self.figure_geometry_loss=[]
        self.figure_shadow_loss=[]
        self.figure_total_loss=[]

        plt.ion()
        self.params=params
        self.start=0
    
    def transfer(self,loss):
        return loss.clone().detach().cpu().numpy()
    
    def update(self,id,force,geometry,shadow):
        if id%self.params.updateplt_hz==0:
            mod_id=id/self.params.updateplt_hz

            fmax_loss_cpu=self.transfer(force.fmax_loss)
            fdir_loss_cpu=self.transfer(force.fdir_loss)
            fnorm1_loss_cpu=self.transfer(force.fnorm1_loss)
            geometry_loss_cpu=self.transfer(geometry.geometry_loss)
            shadow_loss_cpu=self.transfer(shadow.shadow_loss)
            total_loss_cpu=fmax_loss_cpu+fdir_loss_cpu+fnorm1_loss_cpu+geometry_loss_cpu+shadow_loss_cpu
            self.figure_x.append(id)
            self.figure_fmax_loss.append(fmax_loss_cpu)
            self.figure_fdir_loss.append(fdir_loss_cpu)
            self.figure_fnorm1_loss.append(fnorm1_loss_cpu)
            self.figure_geometry_loss.append(geometry_loss_cpu)
            self.figure_shadow_loss.append(shadow_loss_cpu)
            self.figure_total_loss.append(total_loss_cpu)

            plt.clf()
            plt.plot(self.figure_x,self.figure_total_loss,label="total loss",color='red')
            plt.plot(self.figure_x,self.figure_shadow_loss,label="shadow loss",color='lightgreen')
            plt.plot(self.figure_x,self.figure_geometry_loss,label="geometry loss",color='cornflowerblue')
            plt.plot(self.figure_x,self.figure_fnorm1_loss,label="force l1-norm loss",color='peru')
            plt.plot(self.figure_x,self.figure_fdir_loss,label="force direction barrier loss",color="cyan")
            plt.plot(self.figure_x,self.figure_fmax_loss,label="force maximum barrier loss",color="magenta")
            plt.legend(loc="upper right")

    def save_loss(self,result_dir,id):
        if id%self.params.saveloss_hz==0:
            plt.savefig(os.path.join(result_dir, 'loss.png'))
    
class LossyDrawer():
    def __init__(self,params):
        self.figure_index=np.zeros((params.max_iter),np.int32)
        self.figure_data=np.zeros((6,params.max_iter),np.float32)
        self.index_buffer=torch.zeros(params.updateplt_hz,dtype=torch.int32)
        self.data_buffer=torch.zeros(6,params.updateplt_hz,dtype=torch.float32)

        plt.ion()
        self.params=params
        self.start=0
    
    def transfer(self,loss):
        return loss.clone().detach().cpu().numpy()
    
    def update(self,id,force,geometry,shadow):
        mod_id=id%self.params.updateplt_hz
        self.index_buffer[mod_id]=id
        self.data_buffer[0,mod_id]=force.fmax_loss.clone().detach()
        self.data_buffer[1,mod_id]=force.fdir_loss.clone().detach()
        self.data_buffer[2,mod_id]=force.fnorm1_loss.clone().detach()
        self.data_buffer[3,mod_id]=geometry.geometry_loss.clone().detach()
        self.data_buffer[4,mod_id]=shadow.shadow_loss.clone().detach()
        self.data_buffer[5,mod_id]=self.data_buffer[0,mod_id]+self.data_buffer[1,mod_id]+\
                                   self.data_buffer[2,mod_id]+self.data_buffer[3,mod_id]+self.data_buffer[4,mod_id]
        
        if mod_id+1==self.params.updateplt_hz:
            self.figure_index[id-self.params.updateplt_hz+1:id+1]=self.index_buffer.cpu().numpy()
            self.figure_data[:,id-self.params.updateplt_hz+1:id+1]=self.data_buffer.cpu().numpy()
            plt.clf()
            plt.plot(self.figure_index[0:id+1],self.figure_data[0,0:id+1],label="force maximum barrier loss",color="magenta")
            plt.plot(self.figure_index[0:id+1],self.figure_data[1,0:id+1],label="force direction barrier loss",color="cyan")
            plt.plot(self.figure_index[0:id+1],self.figure_data[2,0:id+1],label="force l1-norm loss",color='peru')
            plt.plot(self.figure_index[0:id+1],self.figure_data[3,0:id+1],label="geometry loss",color='cornflowerblue')
            plt.plot(self.figure_index[0:id+1],self.figure_data[4,0:id+1],label="shadow loss",color='lightgreen')
            plt.plot(self.figure_index[0:id+1],self.figure_data[5,0:id+1],label="total loss",color='red')
            plt.legend(loc="upper right")

    def save_loss(self,result_dir,id):
        if id%self.params.saveloss_hz==0:
            plt.savefig(os.path.join(result_dir, 'loss.png'))
    