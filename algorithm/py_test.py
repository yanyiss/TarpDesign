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
def main():
    #print(example.add([[4,5,6,7,7],[5,6,7,2,1]]))
    
    x=torch.tensor(np.array([[[1.0,2.0,3.0],[2.0,3.0,4.0],[3.0,4.0,5.0],[4.0,5.0,6.0]]]))
    print(x)
    
    c = np.array([[0,1,2],[1,3,2],[2,3,4],[1,3,2],[2,3,4]]).flatten()
    print(c.reshape(5,3))
    #print(x-x.reshape(1,))

    i=torch.from_numpy(np.array([1,2,3])).cuda()
    j=torch.from_numpy(np.array([3,7,1,2])).cuda()
    print(torch.cat([i,j],dim=0))


if __name__ == '__main__':
    main()
