import torch
import torch.nn as nn
import numpy as np
import imageio
import algorithm.tarp_info as TI
import algorithm.tool as tool
import soft_renderer as sr
import os
import time

params=TI.tarp_params()

class Shadow():
    def __init__(self,tarp):

        self.transform=sr.LookAt(perspective=False,viewing_scale=params.view_scale,eye=[0.0,0.0,-1.0])
        self.lighting=sr.Lighting(directions=[0,0,1.0])
        self.rasterizer=sr.SoftRasterizer(image_size=params.image_size,sigma_val=params.sigma_value,gamma_val=params.gamma_value,aggr_func_rgb='hard')

        self.tarp=tarp
        self.shadow_image=0
        self.shadow_loss=torch.tensor([0.0],device=torch.device("cuda"))
        self.target_image=imageio.imread(params.image).astype('float32')/255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).unsqueeze(dim=0).cuda()

    def save_image(self,result_dir,id):
        if id%params.saveshadow_hz==0:
            image=self.shadow_image.clone().detach().cpu().numpy().transpose((1,2,0))
            imageio.imsave(os.path.join(result_dir,'shadow/deform_%05d.png'%id),(255*image).astype(np.uint8))

    def loss_evaluation(self):
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        all_shadow_image=self.rasterizer(mesh)
        self.shadow_image=tool.shadow_intersection(all_shadow_image)
        
        if params.loss_type=='image':
            self.shadow_loss=torch.sum((self.shadow_image-self.target_image)**2)/(params.image_size*params.image_size)*params.image_weight
        elif params.loss_type=='area':
            self.shadow_loss=(0.2-torch.sum(self.shadow_image)/(params.image_size*params.image_size))*params.image_weight
        else:
            self.shadow_loss=torch.tensor([0]).cuda()
            print('error in shadow type')
            exit(0)
        return self.shadow_loss