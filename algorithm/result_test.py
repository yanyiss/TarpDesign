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
import openmesh


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

import soft_renderer as sr
def main():
    mesh1=openmesh.read_trimesh(os.path.join(data_dir,'obj/result.obj'))
    mesh2=openmesh.read_trimesh(os.path.join(data_dir,'results/9/result.obj'))

    #print(example.add([[4,5,6,7,7],[5,6,7,2,1]]))
    print(np.array([1,2,3]))
    x=torch.tensor(np.array([[[1.0,2.0,3.0],[2.0,3.0,14.0],[3.0,4.0,5.0],[4.0,5.0,6.0]]])).cuda()
    print(x)
    print(torch.norm(x,p=2,dim=2))
    ff=torch.norm(x,p=2,dim=2)
    print(ff.unsqueeze(dim=2).repeat(1,1,3))

    f=x[0].flatten()
    print(f)
    g=f.reshape(1,4,3)
    print(g)
    print(torch.zeros((4,2,3)))


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
