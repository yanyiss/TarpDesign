from typing import Any
import torch
import os
import numpy as np
import imageio
import soft_renderer as sr


class MeshRender():
    def __init__(self,tarp,params):
        
        self.transform=sr.LookAt(perspective=False,viewing_scale=params.view_scale,eye=[0,0,-1.0])
        #self.transform=sr.LookAt(perspective=False,viewing_scale=params.view_scale,eye=[-3,3,-3.0])
        #self.transform=sr.LookAt(viewing_angle=VIEW_ANGLE,eye=[0,0,-50])
        self.lighting=sr.Lighting(directions=[0,0,1.0])
        self.rasterizer=sr.SoftRasterizer(image_size=params.image_size,sigma_val=params.sigma_value,gamma_val=params.gamma_value,aggr_func_rgb='hard')

        self.tarp=tarp
        self.vertices=self.tarp.vertices
        self.params=params
        self.shadow_image=0
        self.shadow_loss=0
        self.target_image = imageio.imread(params.image).astype('float32') / 255.
        self.target_image=torch.from_numpy(self.target_image.transpose(2,0,1)).cuda().unsqueeze(dim=0)
        self.nv=self.vertices.shape[1]

    def set_target_image(self,target):
        self.target_image=target

    def save_image(self,result_dir,id):
        image=self.shadow_image.clone().detach().cpu().numpy()[0].transpose((1,2,0))
        imageio.imsave(os.path.join(result_dir,'deform_%05d.png'%id),(255*image).astype(np.uint8))

    def loss_evaluation(self,vertice_displace):
        self.tarp.vertices=vertice_displace[:,0:self.nv,:]
        mesh=self.tarp.get_render_mesh()
        mesh=self.lighting(mesh)
        mesh=self.transform(mesh)
        self.shadow_image=self.rasterizer(mesh)

        if self.params.loss_type=='image':
            self.shadow_loss=((self.shadow_image-self.target_image)**2).sum()/(self.params.image_size*self.params.image_size)*self.params.image_weight
        elif self.params.loss_type=='area':
            self.shadow_loss=-torch.sum(self.shadow_image)/(self.params.image_size*self.params.image_size)*self.params.image_weight
        else:
            self.shadow_loss=torch.tensor([0]).cuda()
        return self.shadow_loss
        

