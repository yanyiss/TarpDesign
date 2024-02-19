
""" hessians = torch.tensor([]).cuda()
hessians = torch.cat((hessians, torch.autograd.grad(
    outputs=gradients[:, :1],
    inputs=input_points,
    grad_outputs=d_output,
    create_graph=True,
    retain_graph=True,
    only_inputs=True)[0])
                     )
hessians = torch.cat((hessians, torch.autograd.grad(
    outputs=gradients[:, 1:2],
    inputs=input_points,
    grad_outputs=d_output,
    create_graph=True,
    retain_graph=True,
    only_inputs=True)[0]), dim=1
                     )
hessians = torch.cat((hessians, torch.autograd.grad(
    outputs=gradients[:, 2:],
    inputs=input_points,
    grad_outputs=d_output,
    create_graph=True,
    retain_graph=True,
    only_inputs=True)[0]), dim=1
                     )
hessians = torch.reshape(hessians, (hessians.size()[0], 3, -1)) """

""" import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod=SourceModule(
    """
""" __global__ void multiply_them(float *dest, float *a, float *b)
{
const int i=threadIdx.x;
dest[i]=a[i]*b[i];
} """
"""
)
multiply_them=mod.get_function("multiply_them")
a=np.random.randn(400).astype(np.float32)
b=np.random.randn(400).astype(np.float32)
dest=np.zeros_like(a)
multiply_them(
    cuda.Out(dest),cuda.In(a),cuda.In(b),
    block=(400,1,1),grid=(1,1)
)
print(dest) """
import time
import cupy
import numpy as np
""" a=cupy.sparse.rand(10000,10000,density=0.0004).toarray()
b=cupy.sparse.random(10000,1).toarray() """
#print(a)
""" start=time.perf_counter()
for i in range(10):
    lu=sla.splu(a)
    x=lu.solve(b)
    #x=cupy.linalg.solve(a,b)
print(time.perf_counter()-start) """

""" from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
ca=cupy.array([[1,2,0],[0,0,3],[4,0,5]],dtype=cupy.float64)
cb=cupy.array([2,3,4],dtype=cupy.float64)
cc=cupy.linalg.solve(ca,cb)
print(cc) """

""" from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix,linalg as sla
row=np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/row.txt',dtype=np.int64)
col=np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/col.txt',dtype=np.int64)
value=np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/value.txt',dtype=np.float64)
right=np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/right.txt',dtype=np.float64)
wideright=np.zeros((right.shape[0],100)) """
#wideright[0:right.shape[0],:]=right[:]

from cupyx.scipy.sparse import coo_matrix,csr_matrix, linalg as sla
row=cupy.asarray(np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/row.txt',dtype=np.int64))
col=cupy.asarray(np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/col.txt',dtype=np.int64))
value=cupy.asarray(np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/value.txt',dtype=np.float64))
right=cupy.asarray(np.loadtxt('/home/yanyisheshou/Program/TarpDesign/algorithm/data/denseInfo/right.txt',dtype=np.float64))
wideright=cupy.zeros((right.shape[0],200))
for i in range(200):
    wideright[:,i]=right

""" A=np.zeros((right.shape[0],right.shape[0]))
for i in range(row.shape[0]):
    A[row[i],col[i]]=value[i]
alu=linalg.lu(A) """


start_1=time.perf_counter()
left=coo_matrix((value,(row,col)),shape=(right.shape[0],right.shape[0]))
#left=csr_matrix(left)
lefttemp=coo_matrix((value,(row,col)),shape=(right.shape[0],right.shape[0]))
lefttemp=csr_matrix(lefttemp)
print('-1',time.perf_counter()-start_1)

start0=time.perf_counter()
x=sla.lsqr(left,right)
print('0',time.perf_counter()-start0)

start1=time.perf_counter()
lu=sla.splu(lefttemp)
print('1',time.perf_counter()-start1)

start2=time.perf_counter()
y=lu.solve(wideright)
print('2',time.perf_counter()-start2)
""" for i in range(10):
    lu=sla.splu(left)
    x=lu.solve(wideright) """
""" left=np.zeros(int(row.shape[0])*3,dtype=np.float64)
left=left.reshape(int(row.shape[0]),3)
left[:,0]=row[:]
left[:,1]=col[:]
left[:,2]=value[:] """

#cright=cupy.asarray(right)

#left=cupy.array(left,dtype=cupy.float64)
""" b=cupy.array(cright,dtype=cupy.float64)
for i in range(100):
    rr=cupy.linalg.solve(left,b) """

""" import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

# Create a PyTorch tensor.
tx1 = torch.randn(1, 2, 3, 4).cuda()

# Convert it into a DLPack tensor.
dx = to_dlpack(tx1)

# Convert it into a CuPy array.
cx = cupy.from_dlpack(dx)

# Convert it back to a PyTorch tensor.
tx2 = from_dlpack(cx.toDlpack())
 """