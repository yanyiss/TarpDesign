import torch
import torch.nn as nn
import numpy as np

import algorithm.tarp_info as TI
import algorithm.tool as tool
import os

params=TI.tarp_params()

class Geometry():
    def __init__(self,tarp):
        self.tarp=tarp
        self.stick_index=tarp.tarp_info.C
        self.stick_value=self.tarp.vertices[:,self.stick_index,2].clone().detach()
        self.fixed_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.geometry_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.zero=torch.tensor([0.0],device=torch.device("cuda"))

    def fixed_constraint(self):
        return self.zero
    
    def loss_evaluation(self):
        self.fixed_loss=self.fixed_constraint()
        self.geometry_loss=self.fixed_loss
        return self.geometry_loss


        
