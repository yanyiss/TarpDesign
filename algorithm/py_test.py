"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
#import example

import soft_renderer as sr

def ev(x):
    y = x[0]**2+x[1]**3+x[2]**4+x[3]**5
    return y

def main():
    
 
    x = torch.ones(4)
    x.requires_grad_()

    y = x[0]**2+x[1]**3+x[2]**4+x[3]**5
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    #grad2=torch.zeros((2,2,1,2,2))
    #grad2[:,:,:,0:2,0:2] = torch.autograd.grad(outputs=grad[:,:,:], inputs=x, grad_outputs=torch.ones_like(grad[:,:,:]))[0]
    #grad2[:,:,:,0:2,0:2] = torch.autograd.grad(outputs=grad[:,:,:], inputs=x, grad_outputs=torch.ones_like(grad[:,:,:]))[0]
    #grad2=torch.autograd.grad(outputs=grad,inputs=x,grad_outputs=torch.ones_like(grad))[0]

    grad2=torch.zeros((4,4))
    """ for i in range(4):
        grad2[i,0:4]=torch.autograd.grad(outputs=grad[i],inputs=x,grad_outputs=torch.ones_like(grad[i]),retain_graph=True)[0] """
    
    
    print(grad)
    print(grad2)

    """ x=torch.tensor(np.array([[[1.0,2.0,3.0],[2.0,3.0,14.0],[3.0,4.0,5.0],[4.0,5.0,6.0]]])).cuda()
    print(x.shape)
    y=x[:,0:3,:]
    print(torch.cat((x,y),dim=1)) """
    


    """ print((x[0].t().sum(dim=1)**2).sum())
    print((x[0].sum(dim=1)**2).sum())
    print(x.unsqueeze(dim=-1))
    
    c = np.array([[0,1,2],[1,3,2],[2,3,4],[1,3,2],[2,3,4]]).flatten()
    print(c.reshape(5,3))
    #print(x-x.reshape(1,))

    i=torch.from_numpy(np.array([1,2,3])).cuda()
    j=torch.from_numpy(np.array([3,7,1,2])).cuda()
    print(torch.cat([i,j],dim=0)) """


if __name__ == '__main__':
    main()
