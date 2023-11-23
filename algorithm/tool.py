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


# Generate a random feasible SOCP.
""" m = 3
n = 10
p = 5
n_i = 5
np.random.seed(2)
f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)
for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
F = np.random.randn(p, n)
g = F @ x0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
# We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
soc_constraints = [
      cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
]
prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints + [F @ x == g])
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
for i in range(m):
    print("SOC constraint %i dual variable solution" % i)
    print(soc_constraints[i].dual_value) """