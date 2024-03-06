import torch
import numpy as np
import algorithm.tool as tool
import os

class tarp_info():
    def __init__(self,vertex,data):

        batch_size=vertex.size(0)
        self.nv = vertex.size(1)
        
        #small coef
        self.elp=torch.tensor(data[0]).cuda()
        #stiffnesss                  unit: kg/(m * s^2)
        self.k=torch.tensor(data[1]).cuda()
        #mass
        self.mass=torch.tensor(data[2]).cuda()
        #gravity acceleration        unit: m/(s^2)
        self.g=torch.tensor(data[3]).cuda()
        #gravity                     unit: N
        G=np.zeros([batch_size,self.nv,3]).astype(np.float64)
        G[:,:,2]=-data[2]*data[3]/self.nv
        self.G=torch.from_numpy(G).cuda()
        #maximum force on the rope   unit: N
        self.Fmax=torch.tensor(data[4]).cuda()
        #minimum height of the tarp  unit: m
        self.Hmin=torch.tensor(data[5]).cuda()
        #the height of the sticks    unit: m
        self.H=torch.tensor(data[6]).cuda()
        #maximum length of the rope  unit: m
        self.Lmax=torch.tensor(data[7]).cuda()
        #index of mesh center which is fixed
        self.CI=torch.tensor(int(data[8])).cuda()
        #vertex that connect with a rope
        self.C=torch.from_numpy(data[10:10+int(data[9])].astype(int)).cuda()

class tarp_params():
    def __init__(self):
        meta_params=tool.read_params()
        #dir
        self.current_dir=meta_params['current_dir']
        self.data_dir=meta_params['data_dir']
        self.example_dir=meta_params['example_dir']
        #source file
        self.example_name=meta_params['example_name']
        self.template_mesh=os.path.join(self.example_dir,self.example_name,self.example_name+'.obj')
        self.info_path=os.path.join(self.example_dir,self.example_name,'setting.txt')
        self.image=meta_params['image']
        self.output_dir=meta_params['output_dir']
        #dense file
        self.force_file=meta_params['force_file']
        self.forcedis_file=meta_params['forcedis_file']
        self.result_mesh=meta_params['result_mesh']
        #save file setting
        self.saveshadow_hz=meta_params['saveshadow_hz']
        self.saveloss_hz=meta_params['saveloss_hz']
        self.saveresult_hz=meta_params['saveresult_hz']
        #rendering
        self.image_size=meta_params['IMAGE_SIZE']
        self.sigma_value=meta_params['SIGMA_VAL']
        self.gamma_value=meta_params['GAMMA_VAL']
        self.view_angle=meta_params['VIEW_ANGLE']
        self.view_scale=meta_params['VIEW_SCALE']
        #solar codition
        self.loc_rad=meta_params['loc_latitude']/180.0*math.pi
        self.dir_rad=meta_params['dir_latitude']/180.0*math.pi
        self.start_rad=(12.0-meta_params['start_moment'])/12.0*math.pi
        self.end_rad=(12.0-meta_params['end_moment'])/12.0*math.pi
        self.sample_num=meta_params['sample_num']
        self.sample_type=meta_params['sample_type']
        #gui
        self.use_vertgrad=meta_params['use_vertgrad']
        self.use_forcegrad=meta_params['use_forcegrad']
        self.use_adamgrad=meta_params['use_adamgrad']
        self.updategl_hz=meta_params['updategl_hz']
        self.updateplt_hz=meta_params['updateplt_hz']
        #simulation
        self.balance_cof=meta_params['BALANCE_COF']
        self.balance_min=meta_params['BALANCE_MIN']
        self.newton_rate=meta_params['NEWTON_RATE']
        self.accurate_hz=meta_params['ACCURATE_HZ']
        self.rot=meta_params['rot']
        self.horizontal_load=meta_params['horizontal_load']
        self.vertical_load=meta_params['vertical_load']
        #optimization setting
        self.step_size=meta_params['STEP_SIZE']
        self.decay_gamma=meta_params['DECAY_GAMMA']
        self.learning_rate=meta_params['LEARNING_RATE']
        self.max_iter=meta_params['MAX_ITER']
        self.update_start=meta_params['update_start']
        self.update_w_hz=meta_params['update_w_hz']
        self.enable_prox=meta_params['enable_prox']
        self.grad_error=meta_params['grad_error']
        self.nume_error=meta_params['nume_error']
        #optimization parameter
        self.l1_xi=meta_params['l1_xi']
        self.l1_eta=meta_params['l1_eta']
        self.l1_rho=meta_params['l1_rho']
        self.l1_epsilon=meta_params['l1_epsilon']
        self.l1_alpha=meta_params['l1_alpha']
        self.l1_beta=meta_params['l1_beta']
        self.force_delay=meta_params['force_delay']
        self.cosine_delay=meta_params['cosine_delay']
        self.image_weight=meta_params['image_weight']
        self.fixed_weight=meta_params['fixed_weight']
        self.fmax_weight=meta_params['fmax_weight']
        self.fdir_weight=meta_params['fdir_weight']
        self.fnorm1_weight=meta_params['fnorm1_weight']
        #loss type
        self.loss_type=meta_params['loss_type']
        self.fixed_cons=meta_params['fixed_cons']
        self.fmax_cons=meta_params['fmax_cons']
        self.fdir_cons=meta_params['fdir_cons']
        self.fnorm1_cons=meta_params['fnorm1_cons']
        #other
        self.batch_size=meta_params['batch_size']
        self.use_denseInfo=meta_params['use_denseInfo']

        if self.enable_prox:
            self.fnorm1_cons=0

import soft_renderer as sr
import math
class Tarp():
    def __init__(self,params):
        template_mesh=sr.Mesh.from_obj(params.template_mesh)
        self.batch_size=params.batch_size
        #self.vertices=template_mesh.vertices
        self.vertices=torch.zeros_like(template_mesh.vertices)
        rot=params.rot/180.0*math.pi
        self.vertices[:,:,0]=math.cos(rot)*template_mesh.vertices[:,:,0]-math.sin(rot)*template_mesh.vertices[:,:,1]
        self.vertices[:,:,1]=math.sin(rot)*template_mesh.vertices[:,:,0]+math.cos(rot)*template_mesh.vertices[:,:,1]
        self.vertices[:,:,2]=template_mesh.vertices[:,:,2]
        self.faces=template_mesh.faces
        self.loc_rad=torch.tensor([params.loc_rad]).cuda()
        self.dir_rad=torch.tensor([params.dir_rad]).cuda()
        self.start_rad=torch.tensor([params.start_rad]).cuda()
        self.end_rad=torch.tensor([params.end_rad]).cuda()
        
        self.sample_num=params.sample_num
        self.sample_type=params.sample_type
        self.x_shift=0
        self.y_shift=0

        data=np.loadtxt(params.info_path,dtype=np.float64)
        self.tarp_info=tarp_info(self.vertices,data)

    def set_sampling(self,sampling_lambda):
        if self.sample_num>1:
            if self.sample_type=='time':
                ascension_dif=self.start_rad+torch.range(0,self.sample_num-1).cuda()*(self.end_rad-self.start_rad)/(self.sample_num-1)
            elif self.sample_type=='arc':
                ascension_dif=sampling_lambda
            else:
                print('get render mesh error')
                exit(0)
        else:
            ascension_dif=self.start_rad.unsqueeze(dim=0)
        #print('\n\n\nascension',ascension_dif)
        ascension_dif=ascension_dif.reshape(self.sample_num,1).repeat(1,self.vertices.shape[0])
        self.x_shift=(torch.sin(self.loc_rad)*torch.cos(self.dir_rad)*torch.cos(ascension_dif)-torch.cos(self.loc_rad)*torch.sin(self.dir_rad))\
                /(torch.cos(self.loc_rad)*torch.cos(self.dir_rad)*torch.cos(ascension_dif)+torch.sin(self.loc_rad)*torch.sin(self.dir_rad))
        self.y_shift=torch.cos(self.dir_rad)*torch.sin(ascension_dif)\
                /(torch.cos(self.loc_rad)*torch.cos(self.dir_rad)*torch.cos(ascension_dif)+torch.sin(self.loc_rad)*torch.sin(self.dir_rad))

    def get_render_mesh(self):
        render_vertices=self.vertices.repeat(self.sample_num,1,1)
        render_vertices[:,:,0]=render_vertices[:,:,0]-render_vertices[:,:,2]*self.x_shift
        render_vertices[:,:,1]=render_vertices[:,:,1]-render_vertices[:,:,2]*self.y_shift
        return sr.Mesh(render_vertices,self.faces.repeat(self.sample_num,1,1))
    
    def get_mesh(self):
        return sr.Mesh(self.vertices.repeat(self.batch_size,1,1),self.faces.repeat(self.batch_size,1,1))


    
        