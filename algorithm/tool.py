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
    
