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
import math

import soft_renderer as sr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')

delta = 0.05 * 1

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x, vertex_weight=1):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x * vertex_weight
        x = torch.norm(x, dim=dims) * x.shape[1]

        # batch_size = x.size(0)
        # x = torch.matmul(self.laplacian, x[:, :, 1:2])
        # x = x * vertex_weight
        # x = torch.norm(x)
        # #x = torch.norm(x) * math.sqrt(x.shape[1])
        if self.average:
            return x.sum() / batch_size
        else:
            return x

def readinputmesh(template_path):
    template_mesh1 = sr.Mesh.from_obj(os.path.join(template_path, 'plate.obj'), load_texture=True, texture_res=5, texture_type='surface')
    template_mesh2 = sr.Mesh.from_obj(os.path.join(template_path, 'replate.obj'), load_texture=True, texture_res=5,
                                      texture_type='surface')
    dataheight = template_mesh1.vertices[:, :, 1:2].abs().clone().detach()
    return template_mesh1, template_mesh2, dataheight, np.shape(dataheight)[1]

class Model(nn.Module):
    def __init__(self, template_mesh1, template_mesh2, dataheight, vertex_weight):
        super(Model, self).__init__()

        # set template mesh
        self.register_buffer('vertices1', template_mesh1.vertices)
        self.register_buffer('faces1', template_mesh1.faces)
        self.register_buffer('textures1', template_mesh1.textures)
        self.register_buffer('vertices2', template_mesh2.vertices)
        self.register_buffer('faces2', template_mesh2.faces)
        self.register_buffer('textures2', template_mesh2.textures)
        self.register_buffer('reference_height', dataheight)

        nvw = torch.from_numpy(np.ones(vertex_weight.shape, dtype=np.float32))
        self.register_buffer('vertex_weight', nvw)
        self.register_buffer('last_displace', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.vertices)[1], 1)))
        self.register_buffer('randomtextures', torch.rand(template_mesh1.textures.size()))


        # optimize for displacement map and center
        #self.register_parameter('displace',  nn.Parameter(torch.zeros(1, 1352, 1)))
        self.register_parameter('displace', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.vertices)[1], 1)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 1)))
        self.register_parameter('dtextures', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.textures)[1], np.shape(template_mesh1.textures)[2], 1)))
        self.register_parameter('dtextures1', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.textures)[1], np.shape(template_mesh1.textures)[2], 1)))
        self.register_parameter('dtextures2', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.textures)[1], np.shape(template_mesh1.textures)[2], 1)))
        self.register_parameter('dtextures3', nn.Parameter(torch.zeros(1, np.shape(template_mesh1.textures)[1], np.shape(template_mesh1.textures)[2], 1)))
        self.register_parameter('texturescenter', nn.Parameter(torch.zeros(1, 1, 1)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = LaplacianLoss(self.vertices1[0].cpu(), self.faces1[0].cpu())
        #self.flatten_loss = sr.FlattenLoss(self.faces1[0].cpu())
        #self.data_loss = torch.norm(self.displace)

    def forward(self, batch_size):
        base1 = torch.log((self.vertices1[:, :, 1:2].abs() / 2 + 0.005) / (1 - (self.vertices1[:, :, 1:2].abs() / 2 + 0.005)))
        centroid1 = torch.tanh(self.center)
        vertices1y = ((torch.sigmoid(base1 + self.displace) - 0.005) * 2) * torch.sign(self.vertices1[:, :, 1:2])
        # vertices1y = F.relu(vertices1y) * (1 - centroid1) - F.relu(-vertices1y) * (centroid1 + 1)
        # vertices1y = vertices1y + centroid1
        vertices1 = torch.clone(self.vertices1)
        vertices1[:, :, 1:2] = vertices1y
        vertices2 = torch.clone(self.vertices2)
        vertices2[:, :, 1:2] = vertices1y
        height_map = torch.zeros_like(vertices1)
        height_map[:, :, 1:2] = vertices1y - self.reference_height
        self.last_displace = self.displace.clone().detach()
        ###
        # tbase = torch.log((self.textures1[:, :, :, 0:1] * 0.99 + 0.005) / (1 - (self.textures1[:, :, :, 0:1] * 0.99 + 0.005)))
        # #tcentroid = torch.tanh(self.texturescenter)
        # dtext = (torch.sigmoid(tbase + self.dtextures) - 0.005) / 0.99
        #dtext = F.relu(dtext) * (1 - tcentroid) - F.relu(-dtext) * (tcentroid + 1)
        #dtext = dtext + tcentroid
        # tbase1 = torch.log((self.textures1[:, :, :, 0:1] * 0.99) / (1 - (self.textures1[:, :, :, 0:1] * 0.99)))
        # tbase2 = torch.log((self.textures1[:, :, :, 1:2] * 0.99) / (1 - (self.textures1[:, :, :, 1:2] * 0.99)))
        # tbase3 = torch.log((self.textures1[:, :, :, 2:3] * 0.99) / (1 - (self.textures1[:, :, :, 2:3] * 0.99)))
        # dtext1 = (torch.sigmoid(tbase1 + self.dtextures1))
        # dtext2 = (torch.sigmoid(tbase2 + self.dtextures2))
        # dtext3 = (torch.sigmoid(tbase3 + self.dtextures3))

        # ###################混合初始
        # tbase1 = torch.log((self.textures1[:, :, :, 0:1] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 0:1] * 0.8 + 0.1)))
        # tbase2 = torch.log((self.textures1[:, :, :, 1:2] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 1:2] * 0.8 + 0.1)))
        # tbase3 = torch.log((self.textures1[:, :, :, 2:3] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 2:3] * 0.8 + 0.1)))
        # dtext1 = (torch.sigmoid(tbase1 + self.dtextures1))
        # dtext2 = (torch.sigmoid(tbase2 + self.dtextures2))
        # dtext3 = (torch.sigmoid(tbase3 + self.dtextures3))
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = dtext1
        # textures1[:, :, :, 1:2] = dtext2
        # textures1[:, :, :, 2:3] = dtext3

        # #################随机初始
        # tbase1 = torch.log((self.randomtextures[:, :, :, 0:1] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 0:1] * 0.9 + 0.05)))
        # tbase2 = torch.log((self.randomtextures[:, :, :, 1:2] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 1:2] * 0.9 + 0.05)))
        # tbase3 = torch.log((self.randomtextures[:, :, :, 2:3] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 2:3] * 0.9 + 0.05)))
        # dtext1 = (torch.sigmoid(tbase1 + self.dtextures1))
        # dtext2 = (torch.sigmoid(tbase2 + self.dtextures2))
        # dtext3 = (torch.sigmoid(tbase3 + self.dtextures3))
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = dtext1
        # textures1[:, :, :, 1:2] = dtext2
        # textures1[:, :, :, 2:3] = dtext3

        #################0.5 0.5 0.5初始
        #sigmoid化
        dtext1 = torch.sigmoid(self.dtextures1)
        dtext2 = torch.sigmoid(self.dtextures2)
        dtext3 = torch.sigmoid(self.dtextures3)
        textures1 = torch.clone(self.textures1)
        textures1[:, :, :, 0:1] = dtext1
        textures1[:, :, :, 1:2] = dtext2
        textures1[:, :, :, 2:3] = dtext3

        # #log化
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = self.dtextures1 + 0.5
        # textures1[:, :, :, 1:2] = self.dtextures2 + 0.5
        # textures1[:, :, :, 2:3] = self.dtextures3 + 0.5
        # color_loss = (-torch.log((0.5) - self.dtextures1) - torch.log(self.dtextures1 + (0.5))).sum()\
        #              + (-torch.log((0.5) - self.dtextures2) - torch.log(self.dtextures2 + (0.5))).sum()\
        #              + (-torch.log((0.5) - self.dtextures3) - torch.log(self.dtextures3 + (0.5))).sum()


        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(height_map, self.vertex_weight).mean()
        #flatten_loss = self.flatten_loss(vertices1).mean()
        #textures_loss = torch.norm(self.dtextures)
        data_loss = (-torch.log((delta + 1e-6) - height_map) - torch.log(height_map + (delta + 1e-6))).sum()


        return sr.Mesh(vertices1.repeat(batch_size, 1, 1),
                       self.faces1.repeat(batch_size, 1, 1), textures1.repeat(batch_size, 1, 1, 1)), \
               sr.Mesh(vertices2.repeat(batch_size, 1, 1),
                       self.faces2.repeat(batch_size, 1, 1), textures1.repeat(batch_size, 1, 1, 1)), \
               laplacian_loss, data_loss

    def set_dtextures(self, dtextures, dtextures1, dtextures2, dtextures3):
        self.dtextures = torch.nn.Parameter(dtextures)
        self.dtextures1 = torch.nn.Parameter(dtextures1)
        self.dtextures2 = torch.nn.Parameter(dtextures2)
        self.dtextures3 = torch.nn.Parameter(dtextures3)

    def output_mesh(self, batch_size):
        base1 = torch.log((self.vertices1[:, :, 1:2].abs() / 2 + 0.005) / (1 - (self.vertices1[:, :, 1:2].abs() / 2 + 0.005)))
        centroid1 = torch.tanh(self.center)
        vertices1y = ((torch.sigmoid(base1 + self.displace) - 0.005) * 2) * torch.sign(self.vertices1[:, :, 1:2])
        # vertices1y = F.relu(vertices1y) * (1 - centroid1) - F.relu(-vertices1y) * (centroid1 + 1)
        # vertices1y = vertices1y + centroid1
        vertices1 = torch.clone(self.vertices1)
        vertices1[:, :, 1:2] = vertices1y
        vertices2 = torch.clone(self.vertices2)
        vertices2[:, :, 1:2] = vertices1y


        ###################混合初始
        # tbase1 = torch.log((self.textures1[:, :, :, 0:1] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 0:1] * 0.8 + 0.1)))
        # tbase2 = torch.log((self.textures1[:, :, :, 1:2] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 1:2] * 0.8 + 0.1)))
        # tbase3 = torch.log((self.textures1[:, :, :, 2:3] * 0.8 + 0.1) / (1 - (self.textures1[:, :, :, 2:3] * 0.8 + 0.1)))
        # dtext1 = (torch.sigmoid(tbase1 + self.dtextures1))
        # dtext2 = (torch.sigmoid(tbase2 + self.dtextures2))
        # dtext3 = (torch.sigmoid(tbase3 + self.dtextures3))
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = dtext1
        # textures1[:, :, :, 1:2] = dtext2
        # textures1[:, :, :, 2:3] = dtext3

        # #################随机初始
        # tbase1 = torch.log((self.randomtextures[:, :, :, 0:1] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 0:1] * 0.9 + 0.05)))
        # tbase2 = torch.log((self.randomtextures[:, :, :, 1:2] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 1:2] * 0.9 + 0.05)))
        # tbase3 = torch.log((self.randomtextures[:, :, :, 2:3] * 0.9 + 0.05) / (1 - (self.randomtextures[:, :, :, 2:3] * 0.9 + 0.05)))
        # dtext1 = (torch.sigmoid(tbase1 + self.dtextures1))
        # dtext2 = (torch.sigmoid(tbase2 + self.dtextures2))
        # dtext3 = (torch.sigmoid(tbase3 + self.dtextures3))
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = dtext1
        # textures1[:, :, :, 1:2] = dtext2
        # textures1[:, :, :, 2:3] = dtext3



        ##################0.5 0.5 0.5初始
        #sigmoid化
        dtext = torch.sigmoid(self.dtextures)
        dtext1 = torch.sigmoid(self.dtextures1)
        dtext2 = torch.sigmoid(self.dtextures2)
        dtext3 = torch.sigmoid(self.dtextures3)
        textures1 = torch.clone(self.textures1)
        textures1[:, :, :, 0:1] = dtext1
        textures1[:, :, :, 1:2] = dtext2
        textures1[:, :, :, 2:3] = dtext3

        # #log化
        # textures1 = torch.clone(self.textures1)
        # textures1[:, :, :, 0:1] = self.dtextures1 + 0.5
        # textures1[:, :, :, 1:2] = self.dtextures2 + 0.5
        # textures1[:, :, :, 2:3] = self.dtextures3 + 0.5

        newvertices1 = vertices1.clone().detach()
        newtextures1 = textures1.clone().detach()
        newvertices2 = vertices2.clone().detach()
        return sr.Mesh(newvertices1.repeat(batch_size, 1, 1),
                       self.faces1.repeat(batch_size, 1, 1), newtextures1.repeat(batch_size, 1, 1, 1)), \
               sr.Mesh(newvertices2.repeat(batch_size, 1, 1),
                       self.faces2.repeat(batch_size, 1, 1), newtextures1.repeat(batch_size, 1, 1, 1)), \
               self.dtextures.clone().detach(), \
               self.dtextures1.clone().detach(), self.dtextures2.clone().detach(), self.dtextures3.clone().detach()

    def line_search(self, is_LBFGS):
        if is_LBFGS == False:
            return
        base1 = torch.log((self.vertices1[:, :, 1:2].abs() / 2 + 0.005) / (1 - (self.vertices1[:, :, 1:2].abs() / 2 + 0.005)))
        deltaDisplace = self.displace - self.last_displace
        while 1:
            vertices1y = ((torch.sigmoid(base1 + self.last_displace + deltaDisplace) - 0.005) * 2) * torch.sign(self.vertices1[:, :, 1:2])
            conditions = abs(vertices1y - self.reference_height) >= (delta)
            if conditions.sum() == 0:
                break
            deltaDisplace = torch.where(~conditions, deltaDisplace, 0.9 * deltaDisplace)

        self.displace.data = self.last_displace + deltaDisplace

        # while 1:
        #     conditions1 = abs(self.dtextures1) >= (0.5)
        #     conditions2 = abs(self.dtextures2) >= (0.5)
        #     conditions3 = abs(self.dtextures3) >= (0.5)
        #     if conditions1.sum() == 0 and conditions2.sum() == 0 and conditions3.sum() == 0:
        #         break
        #     self.dtextures1.data = torch.where(~conditions1, self.dtextures1.data, 0.9 * self.dtextures1.data)
        #     self.dtextures2.data = torch.where(~conditions2, self.dtextures1.data, 0.9 * self.dtextures2.data)
        #     self.dtextures3.data = torch.where(~conditions3, self.dtextures1.data, 0.9 * self.dtextures3.data)


    def text_grad(self):
        if self.displace.grad is not None:
            self.displace.grad.data = nn.Parameter(torch.zeros(1, np.shape(self.vertices1)[1], 1)).cuda()
            print(self.displace.grad[0, 0, 0])

def neg_iou_loss(predict, target):
    temp = (predict - target)
    return torch.sum(temp.pow(2))


def my_loss_function(img_loss, laplacian_loss, data_loss):
    return img_loss + laplacian_loss + data_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--mesh-path', type=str,
        default=os.path.join(data_dir, 'input'))
    parser.add_argument('-id1', '--filename-input-d1', type=str,
        default=os.path.join(data_dir, 'input/testD1.png'))
    parser.add_argument('-ir1', '--filename-input-r1', type=str,
        default=os.path.join(data_dir, 'input/testR1.png'))
    parser.add_argument('-id2', '--filename-input-d2', type=str,
        default=os.path.join(data_dir, 'input/testD2.png'))
    parser.add_argument('-ir2', '--filename-input-r2', type=str,
        default=os.path.join(data_dir, 'input/testR2.png'))
    parser.add_argument('-c', '--vertex-weight', type=str,
        default=os.path.join(data_dir, 'input/vertexWeight.txt'))
    parser.add_argument('-md', '--template-meshd', type=str,
        default=os.path.join(data_dir, 'input/plate.obj'))
    parser.add_argument('-mr', '--template-meshr', type=str,
        default=os.path.join(data_dir, 'input/replate.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join(data_dir, 'results/output'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reference_mesh1, reference_mesh2, reference_height, n_vertices = readinputmesh(os.path.join(args.mesh_path, 'reference'))
    template_mesh1, template_mesh2, data_height, n_vertices = readinputmesh(args.mesh_path)

    nrm1 = sr.Mesh(reference_mesh1.vertices, reference_mesh2.faces, reference_mesh2.textures)
    nrm2 = sr.Mesh(reference_mesh2.vertices, reference_mesh1.faces, reference_mesh1.textures)
    rendererd = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=True,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, -0.8, 0],
                                viewing_angle=4.5, light_mode='none')
    rendererr = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=True,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, 0.1, 0],
                                viewing_angle=3, light_mode='none')
    ntargetd = rendererd.render_mesh(nrm1)
    ntargetr = rendererr.render_mesh(nrm2)
    new_tartget = torch.cat((ntargetd, ntargetr), 0)

    # model = Model(args.template_mesh)
    # for k, v in model.named_parameters():
    #     print(k, v.size(), v.requires_grad)
    # exit()



    ###我的代码
    vertex_weight = np.loadtxt(args.vertex_weight, dtype=np.float32)
    vertex_weight = vertex_weight.reshape(1, vertex_weight.size, 1)

    ###我的代码
    imaged = imageio.imread(args.filename_input_d1).astype('float32') / 255.
    imaged = imaged.transpose(2, 0, 1)
    imager = imageio.imread(args.filename_input_r1).astype('float32') / 255.
    imager = imager.transpose(2, 0, 1)
    images_gt1 = torch.stack([torch.from_numpy(imaged).cuda(), torch.from_numpy(imager).cuda()], 0)
    imaged = imageio.imread(args.filename_input_d2).astype('float32') / 255.
    imaged = imaged.transpose(2, 0, 1)
    imager = imageio.imread(args.filename_input_r2).astype('float32') / 255.
    imager = imager.transpose(2, 0, 1)
    images_gt2 = torch.stack([torch.from_numpy(imaged).cuda(), torch.from_numpy(imager).cuda()], 0)

    imgweight = neg_iou_loss(new_tartget[:, 0:4], images_gt1[:, 0:4])

    # rendererd = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=False,
    #                            camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, -0.8, 0],
    #                             viewing_angle=4.5, light_mode='none')
    # rendererr = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=False,
    #                            camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, 0.1, 0],
    #                             viewing_angle=3, light_mode='none')
    # template_mesh1, template_mesh2, data_height, n_vertices = readinputmesh(os.path.join(args.mesh_path, 'reference'))
    # imgd = rendererd.render_mesh(template_mesh1)
    # imgr = rendererr.render_mesh(template_mesh2)
    # image = imgd.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, 'testD1.png'), (255*image).astype(np.uint8))
    # image = imgr.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, 'testR1.png'), (255*image).astype(np.uint8))
    # rendererd = sr.SoftRenderer(image_size=512, sigma_val=1e-7, gamma_val=1e-4, dist_eps=1e-6, fill_back=False,
    #                            camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, -0.8, 0],
    #                             viewing_angle=4.5, light_mode='none')
    # rendererr = sr.SoftRenderer(image_size=512, sigma_val=1e-7, gamma_val=1e-4, dist_eps=1e-6, fill_back=False,
    #                            camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, 0.1, 0],
    #                             viewing_angle=3, light_mode='none')
    # template_mesh1, template_mesh2, data_height, n_vertices = readinputmesh(os.path.join(args.mesh_path, 'reference'))
    # imgd = rendererd.render_mesh(template_mesh1)
    # imgr = rendererr.render_mesh(template_mesh2)
    # image = imgd.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, 'testD2.png'), (255*image).astype(np.uint8))
    # image = imgr.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, 'testR2.png'), (255*image).astype(np.uint8))
    # exit()
    imgLoss = np.zeros(2000)
    dataLoss = np.zeros(2000)
    smoothLoss = np.zeros(2000)

    writerd = imageio.get_writer(os.path.join(args.output_dir, 'deformd.gif'), mode='I')
    writerr = imageio.get_writer(os.path.join(args.output_dir, 'deformr.gif'), mode='I')
    ## main_loop = tqdm.tqdm(list(range (0, 1)))
    ## for iter in main_loop:
    iter_num = 1000
    loop = tqdm.tqdm(list(range(0, iter_num)))
    rendererd = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=False,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, -0.8, 0],
                                viewing_angle=4.5, light_mode='none')
    rendererr = sr.SoftRenderer(image_size=512, sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-4, fill_back=False,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, 0.1, 0],
                                viewing_angle=3, light_mode='none')
    model = Model(template_mesh1, template_mesh2, reference_height, vertex_weight).cuda()
    # read training images and camera poses
    optimizer = torch.optim.Adamax(model.parameters(), 0.01)
    # optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    is_LBFGS = False
    for i in loop:
        # meshd, meshr, laplacian_loss, data_loss = model(args.batch_size)
        # imgd = rendererd.render_mesh(meshd)
        # imgr = rendererr.render_mesh(meshr)
        # resultsImg = torch.cat((imgd, imgr), 0)


        def closure():

            model.line_search(is_LBFGS)
            optimizer.zero_grad()
            meshd, meshr, laplacian_loss, data_loss = model(args.batch_size)
            imgd = rendererd.render_mesh(meshd)
            imgr = rendererr.render_mesh(meshr)
            resultsImg = torch.cat((imgd, imgr), 0)

            img_loss = neg_iou_loss(resultsImg[:, 0:4], images_gt1[:, 0:4])

            loss = my_loss_function(img_loss / imgweight, 0.2676 / 58000 * laplacian_loss * 1 * (1.1), data_loss / imgweight)
            # loss = my_loss_function(img_loss, 0.2676 * laplacian_loss * 1, data_loss)

            imgLoss[i] = img_loss
            dataLoss[i] = data_loss
            smoothLoss[i] = 100000 * laplacian_loss

            loop.set_description('Loss: %.4f, laplacian_loss: %.4f, data_loss: %.4f, Img_loss: %.4f'%(loss.item(), laplacian_loss, data_loss, img_loss))

            loss.backward()
            if i % 200 == 0:
                image = resultsImg.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writerd.append_data((255*image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'process/deform_d%05d.png'%(i)), (255*image).astype(np.uint8))
                image = resultsImg.detach().cpu().numpy()[1].transpose((1, 2, 0))
                writerr.append_data((255*image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'process/deform_r%05d.png'%(i)), (255*image).astype(np.uint8))


            return loss

        optimizer.step(closure)
        model.line_search(True)

        # if i % 100 == 0:
        #     model(1)[0].save_obj(os.path.join(args.output_dir, 'videoresult/plate%05d.obj' % (i)), save_texture=True)
        #     model(1)[1].save_obj(os.path.join(args.output_dir, 'videoresult/replate%05d.obj' % (i)), save_texture=True)

        np.savetxt(os.path.join(args.output_dir, 'imgLoss.txt'), imgLoss)
        np.savetxt(os.path.join(args.output_dir, 'dataLoss.txt'), dataLoss)
        np.savetxt(os.path.join(args.output_dir, 'smoothLoss.txt'), smoothLoss)


    template_mesh1, template_mesh2, dtextures, dtextures1, dtextures2, dtextures3 = model.output_mesh(args.batch_size)
    # save optimized mesh
    model(1)[0].save_obj(os.path.join(args.output_dir, 'halfplate.obj'), save_texture=True)
    model(1)[1].save_obj(os.path.join(args.output_dir, 'halfreplate.obj'), save_texture=True)

    loop = tqdm.tqdm(list(range(0, iter_num)))
    rendererd = sr.SoftRenderer(image_size=512, sigma_val=1e-7, gamma_val=1e-4, dist_eps=1e-6, fill_back=False,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, -0.8, 0],
                                viewing_angle=4.5, light_mode='none')
    rendererr = sr.SoftRenderer(image_size=512, sigma_val=1e-7, gamma_val=1e-4, dist_eps=1e-6, fill_back=False,
                               camera_mode='look_at', eye=[0, 5, 5.5], camera_look_at=[0, 0.1, 0],
                                viewing_angle=3, light_mode='none')
    model = Model(template_mesh1, template_mesh2, reference_height, vertex_weight).cuda()
    model.set_dtextures(dtextures, dtextures1, dtextures2, dtextures3)
    optimizer = torch.optim.Adamax(model.parameters(), 0.01)
    # optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.LBFGS(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for i in loop:
        # meshd, meshr, laplacian_loss, data_loss = model(args.batch_size)
        # imgd = rendererd.render_mesh(meshd)
        # imgr = rendererr.render_mesh(meshr)
        # resultsImg = torch.cat((imgd, imgr), 0)

        def closure():
            model.line_search(is_LBFGS)
            optimizer.zero_grad()
            meshd, meshr, laplacian_loss, data_loss = model(args.batch_size)
            imgd = rendererd.render_mesh(meshd)
            imgr = rendererr.render_mesh(meshr)
            resultsImg = torch.cat((imgd, imgr), 0)


            img_loss = neg_iou_loss(resultsImg[:, 0:4], images_gt2[:, 0:4])

            loss = my_loss_function(img_loss / imgweight, 1.4 / 58000 * laplacian_loss * 0.5 * (1.3), data_loss / imgweight)
            # loss = my_loss_function(img_loss, 1.4 * laplacian_loss, data_loss)

            imgLoss[i + iter_num] = img_loss
            dataLoss[i + iter_num] = data_loss
            smoothLoss[i + iter_num] = 100000 * laplacian_loss

            loop.set_description('Loss: %.4f, laplacian_loss: %.4f, data_loss: %.4f, Img_loss: %.4f' % (
            loss.item(), laplacian_loss, data_loss, img_loss))

            optimizer.zero_grad()
            loss.backward()
            if i % 200 == 0:
                image = resultsImg.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writerd.append_data((255*image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'process/deform_d%05d.png'%(i + iter_num)), (255*image).astype(np.uint8))
                image = resultsImg.detach().cpu().numpy()[1].transpose((1, 2, 0))
                writerr.append_data((255*image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'process/deform_r%05d.png'%(i + iter_num)), (255*image).astype(np.uint8))


            return loss

        optimizer.step(closure)
        model.line_search(True)
        # if i % 100 == 0:
        #     model(1)[0].save_obj(os.path.join(args.output_dir, 'videoresult/plate%05d.obj'%(i + iter_num)), save_texture=True)
        #     model(1)[1].save_obj(os.path.join(args.output_dir, 'videoresult/replate%05d.obj'%(i + iter_num)), save_texture=True)

        np.savetxt(os.path.join(args.output_dir, 'imgLoss.txt'), imgLoss)
        np.savetxt(os.path.join(args.output_dir, 'dataLoss.txt'), dataLoss)
        np.savetxt(os.path.join(args.output_dir, 'smoothLoss.txt'), smoothLoss)

    # save optimized mesh
    model(1)[0].save_obj(os.path.join(args.output_dir, 'plate.obj'), save_texture=True)
    model(1)[1].save_obj(os.path.join(args.output_dir, 'replate.obj'), save_texture=True)

    np.savetxt(os.path.join(args.output_dir, 'imgLoss.txt'), imgLoss)
    np.savetxt(os.path.join(args.output_dir, 'dataLoss.txt'), dataLoss)
    np.savetxt(os.path.join(args.output_dir, 'smoothLoss.txt'), smoothLoss)

if __name__ == '__main__':
    main()