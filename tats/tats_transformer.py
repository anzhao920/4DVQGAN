# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules.utils import shift_dim, accuracy, comp_getattr, ForkedPdb,enable_running_stats,disable_running_stats
from .modules.gpt import GPT
from .modules.encoders import Labelator, SOSProvider, Identity
from .modules.create_latent_ode_model import create_LatentODE_model
from einops import rearrange,repeat
import tensorboard
import os
import nibabel as nib
import numpy as np
from itertools import chain
from .modules.sam import SAM
from .modules.conv_odegru import VidODE
from .modules.video_swin_transformer import SwinTransformer3D
from collections import OrderedDict
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 args,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.args = args
        self.mode = args.mode
        self.class_cond_dim = args.class_cond_dim
        self.be_unconditional = args.unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.vtokens = args.vtokens
        self.sample_every_n_latent_frames = getattr(args, 'sample_every_n_latent_frames', 0)
        
        self.init_first_stage_from_ckpt(args)
        self.init_cond_stage_from_ckpt(args)

        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size
        obsrv_std = 0.01
        self.embedding_dim=args.embedding_dim
        self.input_dim = self.embedding_dim*args.scale**3
        self.scale = args.scale
        self.batch_size_ode = args.batch_size_ode
        self.timepoints = args.timepoints
	    # obsrv_std = torch.Tensor([obsrv_std]).to(device)
	    # z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        temp_device = torch.device('cuda')
        z0_prior = torch.distributions.Normal(torch.Tensor([0.0]).to(temp_device), torch.Tensor([1.]).to(temp_device))
        # ODEdecoder = nn.Linear(args.latents, args.n_embd)
        ODEdecoder = nn.Sequential(
		   nn.Linear(args.latents, args.embedding_dim),)
        # self.latentODE_model = create_LatentODE_model(args, self.input_dim, z0_prior, obsrv_std,temp_device,ODEdecoder)
        self.latentODE_model = VidODE(args, 24, temp_device)
        # self.transformer = GPT(args, gpt_vocab_size, args.block_size, n_layer=args.n_layer, n_head=args.n_head, 
        #                         n_embd=args.n_embd, vtokens_pos=args.vtokens_pos, n_unmasked=args.n_unmasked, head_output_dim=self.input_dim)
        
        # self.output_head = nn.Sequential(
		#    nn.Linear(self.embedding_dim, int(args.embedding_dim*2)),
		#    nn.Tanh(),
		#    nn.Linear(int(args.embedding_dim*2), args.embedding_dim),)
        # self.transformer =SwinTransformer3D(
        #          patch_size=(1,1,1),
        #          in_chans=16,
        #          embed_dim=96,
        #         #  depths=[2, 2, 6, 2],
        #         #  num_heads=[3, 6, 12, 24],
        #          depths=[2],
        #          num_heads=[3],                
        #          window_size=(8,8,8),
        #          patch_norm=True)
        
        self.output_head = nn.Sequential(
		   nn.Linear(16, int(args.embedding_dim*2)),
		   nn.Tanh(),
		   nn.Linear(int(args.embedding_dim*2), args.embedding_dim),)
        
        # checkpoint = torch.load('./swin_tiny_patch244_window877_kinetics400_1k.pth')

        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     if 'backbone' in k and 'layers.0' in k and 'relative_position_' not in k and 'patch_embed' not in k:
        #         name = k[9:]
        #         new_state_dict[name] = v 

        # self.transformer.load_state_dict(new_state_dict,strict=False) 



        # self.output_head = nn.Sequential(
		#    nn.Linear(self.embedding_dim, int(gpt_vocab_size*2)),
		#    nn.Tanh(),
		#    nn.Linear(int(gpt_vocab_size*2), gpt_vocab_size),)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        self.save_hyperparameters()
        if self.args.optimizer == 'SAM':
            self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if not args.vtokens:
            self.first_stage_model = load_vqgan(args.vqvae)
            for p in self.first_stage_model.parameters():
                p.requires_grad = False
            self.first_stage_model.codebook._need_init = False
            self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            self.first_stage_vocab_size = self.first_stage_model.codebook.n_codes
        else:
            self.first_stage_model = None
            self.first_stage_vocab_size = 16384
            # self.first_stage_vocab_size = self.args.first_stage_vocab_size

    def init_cond_stage_from_ckpt(self, args):
        from .download import load_vqgan
        if self.cond_stage_key=='label' and not self.be_unconditional:
            model = Labelator(n_classes=args.class_cond_dim)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model
            self.cond_stage_vocab_size = self.class_cond_dim
        elif self.cond_stage_key=='stft':
            self.cond_stage_model = load_vqgan(args.stft_vqvae)
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False
            self.cond_stage_model.codebook._need_init = False
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            self.cond_stage_vocab_size = self.cond_stage_model.codebook.n_codes
        elif self.cond_stage_key=='text':
            self.cond_stage_model = Identity()
            self.cond_stage_vocab_size = 49408
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
            self.cond_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond_stage_key)

    # def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, cbox=None,save_nii=False,all_time_points = False,patient_IDs=None):
    #     # # one step to produce the logits
    #     # if x.size()[1]>1:
    #     #     x = rearrange(x,'b c h w d -> (b c) h w d') 
    #     #     x = x[:,None]
    #     # if time_points is not None:
    #     #     time_points = rearrange(time_points,'b t -> (b t)')
    #     #     observed_mask = ~torch.isnan(time_points)
    #     logits_list = []
    #     logits_masked_list = []
    #     targets_list = []
    #     targets_masked_list=[]
    #     time_points_list = []
    #     time_points_list_all = []
    #     for idx in range(x.size()[0]):
    #         temp = x[idx,:]
    #         temp = temp[:,None]
    #         vq_embeddings, z_indices = self.encode_to_z(temp)  
    #         lung_masks_idx = lung_masks[idx,:]
    #         lung_masks_idx = lung_masks_idx[:,None]
    #         # size_vq = vq_embeddings.size()
    #         # size_vq[0]=x.size()[0]
    #         # size_zindex = z_indices.size()
    #         # size_zindex[0]=x.size()[0]
    #         # vq_embeddings_latentODE = torch.zeros(size_vq)
    #         # z_indices_latentODE = torch.zeros(size_zindex)
    #         # vq_embeddings_latentODE[]
    #         # _, c_indices = self.encode_to_c(c)
    #         z_indices = z_indices + self.cond_stage_vocab_size

    #         if self.training and self.pkeep < 1.0:
    #             mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
    #                                                         device=z_indices.device))
    #             mask = mask.round().to(dtype=torch.int64)
    #             r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
    #             a_indices = mask*z_indices+(1-mask)*r_indices
    #         else:
    #             a_indices = z_indices

    #         # cz_indices = torch.cat((c_indices, a_indices), dim=1)
    #         # target includes all sequence elements (no need to handle first one
    #         # differently because we are conditioning)

    #         z_indices=rearrange(z_indices,'t c h w -> t (c h w)')
    #         observed_time_mask = ~torch.isnan(time_points[idx])
    #         z_indices = z_indices[observed_time_mask,:]  
    #         # print("the number of time points is:")
    #         # print(z_indices.shape[0])  
    #         time_points_list.append(time_points[idx,observed_time_mask])        
    #         targets_list.append(z_indices)
    #         b,c1,h1,w1,d = vq_embeddings.size()
    #         # latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
    #         latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
    #         latent_mask[latent_mask>0]=1
    #         latent_mask = rearrange(latent_mask,'b t c h w->b t (c h w)')
    #         data_object_list = self.latent_ODE_data_object(vq_embeddings,latent_mask.bool(),time_points[idx],self.scale,self.batch_size_ode)
    #         sol_y_list = []
            
    #         for i in range(len(data_object_list)):
    #             sol_y,sol_y_all = self.latentODE_model.compute_all_losses(data_object_list[i],n_traj_samples=1)
    #             if all_time_points:
    #                 sol_y_list.append(sol_y_all.mean(dim=0))
    #             else:
    #                 sol_y_list.append(sol_y.mean(dim=0))
    #             # sol_y=data_object_list[i]['data_to_predict']#temp
    #             # sol_y_list.append(sol_y)#temp

    #         data_transformer = torch.cat(sol_y_list)
    #         data_transformer = rearrange(data_transformer,'l t d->t l d')

    #         # classtoken_mask =  torch.zeros(latent_mask.shape[0],latent_mask.shape[1],1)
    #         gpt_mask = torch.matmul(latent_mask.transpose(1,2),latent_mask)
    #         gpt_mask=gpt_mask.unsqueeze(1)
    #         if all_time_points:
    #             transformer_outputs_list = []
    #             for t in range(0,data_transformer.shape[0],4):
    #                 transformer_outputs_temp, _ = self.transformer(embeddings=data_transformer[t,:], masks = gpt_mask, cbox=cbox)
    #                 transformer_outputs_list.append(transformer_outputs_temp)
    #             transformer_outputs=torch.cat(transformer_outputs_list)
    #             time_points_list_all.append(range(0,data_transformer.shape[0],4))
    #             time_points_list = time_points_list_all

    #         else:
    #             transformer_outputs, _ = self.transformer(embeddings=data_transformer, masks = gpt_mask, cbox=cbox)
    #         # # vq_embeddings=rearrange(vq_embeddings,'t (c s1) (h s2) (w s3) d-> (c h w) t (s1 s2 s3 d)',s1 =scale,s2 =scale,s3 =scale)

    #         # transformer_outputs = torch.cat(sol_y_list)#temp
    #         # transformer_outputs = rearrange(transformer_outputs,'l t d->t l d')#temp

    #         c = int(c1/self.scale)
    #         h = int(h1/self.scale)
    #         w = int(w1/self.scale)
    #         transformer_outputs = data_transformer           
    #         transformer_outputs = rearrange(transformer_outputs,'t (c h w) (s1 s2 s3 d) -> t (c s1) (h s2) (w s3) d', s1 =self.scale,s2 =self.scale,s3 =self.scale,c=c,h=h,w=w)
    #         transformer_outputs = rearrange(transformer_outputs,'t c h w d-> t (c h w) d')
    #         logits = self.output_head(transformer_outputs)
                        
            
    #         # logits = transformer_outputs
    #         # flat_inputs = rearrange(logits,'t l d-> (t l) d')
    #         # distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
    #         #             - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
    #         #             + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c]
    #         # logits = rearrange(-distances,'(t l) d->t l d',t=logits.shape[0])

    #         # output_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(c1,h1,w1),mode='trilinear',align_corners=False)
            
    #         c = int(c1/self.scale)
    #         h = int(h1/self.scale)
    #         w = int(w1/self.scale)
    #         latent_mask = rearrange(latent_mask,'b t (c h w)->b t c h w',c=c,h=h,w=w)
    #         output_mask = torch.nn.functional.interpolate(latent_mask,size=(c1,h1,w1),mode='trilinear',align_corners=False)
    #         output_mask[output_mask>0]=1
    #         output_mask = rearrange(output_mask,'b t c h w->b t (c h w)').squeeze().bool()

    #         logits_list.append(logits)
    #         logits_masked_list.append(logits[:,output_mask,:].reshape(-1,logits.shape[-1]))
    #         targets_masked_list.append(z_indices[:,output_mask].reshape(-1))
    #         # batch_size_transformer=4
    #         # logits_list = []
    #         # for i in range(self.args.timepoints/4):
    #         #     logits, _ = self.transformer(data_transformer, cbox=cbox)
    #         #     logits_list.append(logits)
        
    #         # logits_all = logits_list
    #     logits_maksed_all= torch.cat(logits_masked_list)
    #     target_maksed_all= torch.cat(targets_masked_list)
    #     # # make the prediction by using transformer as decoder
    #     # logits, _ = self.transformer(cz_indices[:, :-1], cbox=cbox)
    #     # # logits, _ = self.transformer(cz_indices[:, :-1], cbox=cbox)
    #     # # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
    #     # logits = logits[:, c_indices.shape[1]-1:]
    #     for idx in range(x.size()[0]):
    #         if save_nii and logits_list[idx].shape[0]>1:
    #             with torch.no_grad():
    #                 if self.training:
    #                     split='train'
    #                 else:
    #                     split='val'
                    
    #                 lung_masks_0 = lung_masks[idx]
    #                 lung_masks_0 = lung_masks_0[:,None]
    #                 output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1),mode='trilinear',align_corners=False)
    #                 # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1))
    #                 output_mask[output_mask>0]=1
    #                 output_mask = rearrange(output_mask.squeeze(),'c h w->(c h w)').bool()

    #                 for t in range(logits_list[idx].shape[0]):
    #                     predicted_indices = logits_list[idx][t,:]
    #                     predicted_indices = predicted_indices.max(1).indices 
    #                     baseline_indices = targets_list[idx][0,:].clone()
    #                     baseline_indices[output_mask]=predicted_indices[output_mask]
    #                     predicted_indices = baseline_indices
    #                     predicted_indices = rearrange(predicted_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
    #                     niis = self.first_stage_model.decode(predicted_indices.unsqueeze(0))
    #                     niis = niis.squeeze().cpu().numpy()
    #                     niis = niis.transpose(2,1,0)
    #                     niis = np.flip(niis,axis=1)
    #                     niis = np.flip(niis,axis=0)
    #                     image_type = 'reconstruction'
    #                     self.save_nii(self.logger.save_dir, split,image_type,niis ,
    #                         self.global_step, self.current_epoch, patient_IDs[idx], time_points_list[idx][t])

    #                     if not all_time_points:
    #                         target_indices = targets_list[idx][t,:]
    #                         baseline_indices = targets_list[idx][0,:].clone()
    #                         baseline_indices[output_mask]=target_indices[output_mask]
    #                         target_indices = baseline_indices
    #                         target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
    #                         niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
    #                         niis = niis.squeeze().cpu().numpy()
    #                         niis = niis.transpose(2,1,0)
    #                         niis = np.flip(niis,axis=1)
    #                         niis = np.flip(niis,axis=0)
    #                         image_type = 'target'
    #                         self.save_nii(self.logger.save_dir, split,image_type,niis,
    #                             self.global_step, self.current_epoch,  patient_IDs[idx], time_points_list[idx][t])
            

    #     return logits_maksed_all, target_maksed_all


# # only using ode
#     def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, cbox=None,save_nii=False,all_time_points = False,patient_IDs=None):
#         # # one step to produce the logits
#         # if x.size()[1]>1:
#         #     x = rearrange(x,'b c h w d -> (b c) h w d') 
#         #     x = x[:,None]
#         # if time_points is not None:
#         #     time_points = rearrange(time_points,'b t -> (b t)')
#         #     observed_mask = ~torch.isnan(time_points)
#         logits_list = []
#         logits_masked_list = []
#         targets_list = []
#         targets_masked_list=[]
#         time_points_list = []
#         time_points_list_all = []
#         for idx in range(x.size()[0]):
#             temp = x[idx,:]
#             temp = temp[:,None]
#             vq_embeddings, z_indices = self.encode_to_z(temp)  
#             lung_masks_idx = lung_masks[idx,:]
#             lung_masks_idx = lung_masks_idx[:,None]
#             # size_vq = vq_embeddings.size()
#             # size_vq[0]=x.size()[0]
#             # size_zindex = z_indices.size()
#             # size_zindex[0]=x.size()[0]
#             # vq_embeddings_latentODE = torch.zeros(size_vq)
#             # z_indices_latentODE = torch.zeros(size_zindex)
#             # vq_embeddings_latentODE[]
#             # _, c_indices = self.encode_to_c(c)
#             z_indices = z_indices + self.cond_stage_vocab_size

#             if self.training and self.pkeep < 1.0:
#                 mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
#                                                             device=z_indices.device))
#                 mask = mask.round().to(dtype=torch.int64)
#                 r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
#                 a_indices = mask*z_indices+(1-mask)*r_indices
#             else:
#                 a_indices = z_indices

#             # cz_indices = torch.cat((c_indices, a_indices), dim=1)
#             # target includes all sequence elements (no need to handle first one
#             # differently because we are conditioning)

#             z_indices=rearrange(z_indices,'t c h w -> t (c h w)')
#             observed_time_mask = ~torch.isnan(time_points[idx])
#             z_indices = z_indices[observed_time_mask,:]  
#             time_points_list.append(time_points[idx,observed_time_mask])        
#             targets_list.append(z_indices)
#             b,c1,h1,w1,d = vq_embeddings.size()
#             latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
#             latent_mask[latent_mask>0]=1
#             vq_embeddings = vq_embeddings[:,latent_mask.bool().squeeze(),:]
#             latent_mask = rearrange(latent_mask,'b t c h w->b t (c h w)')
#             z_indices = z_indices[:,latent_mask.bool().squeeze()]
#             data_object_list = self.latent_ODE_data_object(vq_embeddings,latent_mask[:,:,latent_mask.bool().squeeze()].bool(),time_points[idx],self.scale,z_indices.shape[1])
#             sol_y_list = []
            
#             for i in range(len(data_object_list)):
#                 sol_y,sol_y_all = self.latentODE_model.compute_all_losses(data_object_list[i],n_traj_samples=1)
#                 if all_time_points:
#                     sol_y_list.append(sol_y_all.mean(dim=0))
#                 else:
#                     sol_y_list.append(sol_y.mean(dim=0))
#                 # sol_y=data_object_list[i]['data_to_predict']#temp
#                 # sol_y_list.append(sol_y)#temp

#             logits = torch.cat(sol_y_list)

#             logits_masked_list.append(logits.reshape(-1,logits.shape[-1]))
#             targets_masked_list.append(z_indices.reshape(-1))
#             # batch_size_transformer=4
#             # logits_list = []
#             # for i in range(self.args.timepoints/4):
#             #     logits, _ = self.transformer(data_transformer, cbox=cbox)
#             #     logits_list.append(logits)
        
#             # logits_all = logits_list
#         logits_maksed_all= torch.cat(logits_masked_list)
#         target_maksed_all= torch.cat(targets_masked_list)
#         target_maksed_embedding_all=self.first_stage_model.codebook.embeddings[target_maksed_all,:]

# only using covgru-3dode
    def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, cbox=None,save_nii=False,all_time_points = False,patient_IDs=None):
        # # one step to produce the logits
        # if x.size()[1]>1:
        #     x = rearrange(x,'b c h w d -> (b c) h w d') 
        #     x = x[:,None]
        # if time_points is not None:
        #     time_points = rearrange(time_points,'b t -> (b t)')
        #     observed_mask = ~torch.isnan(time_points)
        logits_list = []
        # logits_masked_list = []
        targets_list = []
        # targets_masked_list=[]
        time_points_list =[]
        b,t,c,h,w=x.shape
        with torch.no_grad():
            vq_embeddings, z_indices = self.encode_to_z(rearrange(x,'b t (n c) h w->(b t) n c h w',n=1))
            vq_embeddings = rearrange(vq_embeddings,'(b t) d h w c->b t c d h w',b=b)
            z_indices = rearrange(z_indices,'(b t) d h w->b t d h w',b=b)
            observed_time_mask = (~torch.isnan(time_points)).long()
            z_indices = z_indices + self.cond_stage_vocab_size

            b, t, c, d, h, w = vq_embeddings.size()
            latent_mask = torch.nn.functional.interpolate(lung_masks,size=(d,h,w),mode='trilinear',align_corners=False)
            latent_mask[latent_mask>0]=1
            
            batch_dict = self.latent_ODE_data_object(vq_embeddings,z_indices,observed_time_mask,time_points,self.scale)
            sol_y_list = []
            # latent_mask = repeat(latent_mask,'b t c h w->b t (c h w)')

            z_indices = batch_dict['z_indices']     
            # targets_list.append(z_indices.detach())
            temp_mask = repeat(latent_mask.bool(),'b t c h w->b (r t) c h w',r=z_indices.shape[1])[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
            z_indices_target_flat = z_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:][temp_mask]
            # time_points_list.append(time_points[observed_time_mask.bool()])    

        
        sol_y,sol_y_all = self.latentODE_model.compute_all_losses(batch_dict) 
        logits = sol_y[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        logits = rearrange(logits,'t c d h w->t d h w c')
        logits_flat = logits[temp_mask,:]
        logits_list.append(logits_flat.detach())
        # if all_time_points:
        #     sol_y_list.append(sol_y_all[:,:,:,latent_mask.bool().squeeze()])
        # else:
        #     sol_y_list.append(sol_y.mean(dim=0))
            # sol_y=data_object_list[i]['data_to_predict']#temp
            # sol_y_list.append(sol_y)#temp

        # logits = torch.cat(sol_y_list)

        # logits_masked_list.append(logits.reshape(-1,logits.shape[-1]))
        # targets_masked_list.append(z_indices.reshape(-1))
            # batch_size_transformer=4
            # logits_list = []
            # for i in range(self.args.timepoints/4):
            #     logits, _ = self.transformer(data_transformer, cbox=cbox)
            #     logits_list.append(logits)
        
            # logits_all = logits_list
        
        

        # logits_maksed_all= rearrange(logits,'b t d l->(b t l) d')
        # target_maksed_all= rearrange(z_indices,'b t l->(b t l)')
        logits_maksed_all = logits_flat
        target_maksed_all = z_indices_target_flat
        target_maksed_embedding_all=self.first_stage_model.codebook.embeddings[target_maksed_all,:] 
        
        # mask_predicted_data
        # target_image_all = x[target_maksed_all]
        # reconstructed =       
        # lung_masks_0 = lung_masks[0]
        # lung_masks_0 = lung_masks_0[:,None]
        # c1,h1,w1 = d,h,w
        # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(d,h,w),mode='trilinear',align_corners=False)
        # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1))
        # output_mask[output_mask>0]=1
        # output_mask = rearrange(output_mask.squeeze(),'c h w->(c h w)').bool()

        # flat_inputs = logits_list[0][0,1,:].transpose(0,1)
        distances = (logits_flat ** 2).sum(dim=1, keepdim=True) \
                    - 2 * logits_flat @ self.first_stage_model.codebook.embeddings.t() \
                    + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
        predicted_indices = torch.argmin(distances,dim=1)
        predicted_indices = distances.min(1).indices 
        baseline_indices =  z_indices[:,0,:].clone()
        baseline_indices = repeat(baseline_indices.unsqueeze(1),'b t c h w->b (r t) c h w',r=z_indices.shape[1])
        baseline_indices=baseline_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        baseline_indices[temp_mask]=predicted_indices
        predicted_indices = baseline_indices.clone()
        input_CTs = x[observed_time_mask.bool(),:]
        reconstructed_CTs = torch.zeros(input_CTs.shape).cuda()
        lung_masks = repeat(lung_masks.bool(),'b t c h w->b (r t) c h w',r=z_indices.shape[1])[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        mse_loss = nn.MSELoss()
        psnr = PeakSignalNoiseRatio().cuda()
        mse_loss_sum=0
        psnr_sum=0
        ssim_sum = 0
        # ms_ssim_sum = 0
        padding = nn.ReplicationPad3d(5)
        if not self.training:
            with torch.no_grad():
                for i in range(0,predicted_indices.shape[0]):
                    reconstructed_CTs[i,:]= self.first_stage_model.decode(predicted_indices[i:i+1,:])
                    mse_loss_sum=mse_loss_sum+mse_loss(reconstructed_CTs[i,lung_masks[i]],input_CTs[i,lung_masks[i]])
                    psnr_sum = psnr_sum+psnr(reconstructed_CTs[i,lung_masks[i]],input_CTs[i,lung_masks[i]])
                    _,ssim_map=ssim(padding(reconstructed_CTs[i:i+1].unsqueeze(0)), padding(input_CTs[i:i+1].unsqueeze(0)), data_range=1, size_average=False)
                    ssim_sum =ssim_sum+ssim_map[:,:,lung_masks[i]].mean()
                    # ms_ssim_sum = ms_ssim(padding(reconstructed_CTs[i:i+1].unsqueeze(0)), padding(input_CTs[i:i+1].unsqueeze(0)), data_range=1, size_average=False) MS is too high
        ssim_step = ssim_sum/predicted_indices.shape[0]
        psnr_step = psnr_sum/predicted_indices.shape[0]
        mse_step = mse_loss_sum/predicted_indices.shape[0]


         

        # visualization
        # with torch.no_grad():
        #     if self.training:
        #         split='train'
        #     else:
        #         split='val'
            
        #     lung_masks_0 = lung_masks[0]
        #     lung_masks_0 = lung_masks_0[:,None]
        #     c1,h1,w1 = d,h,w
        #     output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(d,h,w),mode='trilinear',align_corners=False)
        #     # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1))
        #     output_mask[output_mask>0]=1
        #     # output_mask = rearrange(output_mask.squeeze(),'c h w->(c h w)').bool()

        #     flat_inputs = logits_list[0][0,1,:].transpose(0,1)
        #     distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
        #                 - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
        #                 + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
        #     predicted_indices = torch.argmin(distances,dim=1)
        #     predicted_indices = distances.min(1).indices 
        #     baseline_indices = targets_list[0][0,0,:].clone()
        #     baseline_indices[output_mask.squeeze().bool()]=predicted_indices
        #     predicted_indices = baseline_indices
        #     # predicted_indices = rearrange(predicted_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
        #     niis = self.first_stage_model.decode(predicted_indices.unsqueeze(0))
        #     niis = niis.squeeze().cpu().numpy()
        #     niis = niis.transpose(2,1,0)
        #     niis = np.flip(niis,axis=1)
        #     niis = np.flip(niis,axis=0)
        #     image_type = 'reconstruction'
        #     self.save_nii(self.logger.save_dir, split,image_type,niis ,
        #         self.global_step, self.current_epoch, patient_IDs[0], time_points_list[0][1])

        #     if not all_time_points:
        #         target_indices = targets_list[0][0,1,:]
        #         baseline_indices = targets_list[0][0,0,:].clone()
        #         baseline_indices[output_mask.squeeze().bool()]=target_indices[output_mask.squeeze().bool()]
        #         target_indices = baseline_indices
        #         # target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
        #         niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
        #         niis = niis.squeeze().cpu().numpy()
        #         niis = niis.transpose(2,1,0)
        #         niis = np.flip(niis,axis=1)
        #         niis = np.flip(niis,axis=0)
        #         image_type = 'target'
        #         self.save_nii(self.logger.save_dir, split,image_type,niis,
        #             self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][1])
                
        #         target_indices = targets_list[0][0,0,:]
        #         baseline_indices = targets_list[0][0,0,:].clone()
        #         baseline_indices[output_mask.squeeze().bool()]=target_indices[output_mask.squeeze().bool()]
        #         target_indices = baseline_indices
        #         # target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
        #         niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
        #         niis = niis.squeeze().cpu().numpy()
        #         niis = niis.transpose(2,1,0)
        #         niis = np.flip(niis,axis=1)
        #         niis = np.flip(niis,axis=0)
        #         image_type = 'baseline'
        #         self.save_nii(self.logger.save_dir, split,image_type,niis,
        #             self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][0])
        #     else:
        #         for idx in range(0,self.timepoints):
        #             flat_inputs = sol_y_list[0][0,idx,:].transpose(0,1)
        #             distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
        #                         - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
        #                         + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
        #             predicted_indices = torch.argmin(distances,dim=1)
        #             predicted_indices = distances.min(1).indices 
        #             baseline_indices = targets_list[0][0,0,:].clone()
        #             baseline_indices[output_mask.squeeze().bool()]=predicted_indices
        #             predicted_indices = baseline_indices
        #             # predicted_indices = rearrange(predicted_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
        #             niis = self.first_stage_model.decode(predicted_indices.unsqueeze(0))
        #             niis = niis.squeeze().cpu().numpy()
        #             niis = niis.transpose(2,1,0)
        #             niis = np.flip(niis,axis=1)
        #             niis = np.flip(niis,axis=0)
        #             image_type = 'reconstruction'
        #             self.save_nii(self.logger.save_dir, split,image_type,niis ,
        #                 self.global_step, self.current_epoch, patient_IDs[0], idx)                    



        return logits_maksed_all, target_maksed_all,target_maksed_embedding_all,ssim_step,psnr_step,mse_step

# # only using transformer for predicting next frame
#     def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, cbox=None,save_nii=False,all_time_points = False,patient_IDs=None):

#         logits_list = []
#         logits_masked_list = []
#         targets_list = []
#         targets_masked_list=[]
#         time_points_list = []
#         time_points_list_all = []
#         for idx in range(x.size()[0]):
#             temp = x[idx,:]
#             temp = temp[:,None]
#             vq_embeddings, z_indices = self.encode_to_z(temp)  
#             lung_masks_idx = lung_masks[idx,:]
#             lung_masks_idx = lung_masks_idx[:,None]
#             z_indices = z_indices + self.cond_stage_vocab_size

#             if self.training and self.pkeep < 1.0:
#                 mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
#                                                             device=z_indices.device))
#                 mask = mask.round().to(dtype=torch.int64)
#                 r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
#                 a_indices = mask*z_indices+(1-mask)*r_indices
#             else:
#                 a_indices = z_indices

#             z_indices=rearrange(z_indices,'t c h w -> t (c h w)')
#             observed_time_mask = ~torch.isnan(time_points[idx])
#             z_indices = z_indices[observed_time_mask,:]  

#             time_points_list.append(time_points[idx,observed_time_mask])        
#             targets_list.append(z_indices)
#             b,c1,h1,w1,d = vq_embeddings.size()
#             # latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
#             latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
#             latent_mask[latent_mask>0]=1
#             latent_mask = rearrange(latent_mask,'b t c h w->b t (c h w)')
#             data_transformer=rearrange(vq_embeddings,'t (c s1) (h s2) (w s3) d-> t (c h w) (s1 s2 s3 d)',s1 =self.scale,s2 =self.scale,s3 =self.scale)
#             gpt_mask = torch.matmul(latent_mask.transpose(1,2),latent_mask)
#             gpt_mask=gpt_mask.unsqueeze(1)
#             transformer_outputs, _ = self.transformer(embeddings=data_transformer[0:1,:], masks = gpt_mask, cbox=cbox)
#             c = int(c1/self.scale)
#             h = int(h1/self.scale)
#             w = int(w1/self.scale)      
#             transformer_outputs = rearrange(transformer_outputs,'t (c h w) (s1 s2 s3 d) -> t (c s1) (h s2) (w s3) d', s1 =self.scale,s2 =self.scale,s3 =self.scale,c=c,h=h,w=w)
#             transformer_outputs = rearrange(transformer_outputs,'t c h w d-> t (c h w) d')
#             logits = self.output_head(transformer_outputs)
            
#             c = int(c1/self.scale)
#             h = int(h1/self.scale)
#             w = int(w1/self.scale)
#             latent_mask = rearrange(latent_mask,'b t (c h w)->b t c h w',c=c,h=h,w=w)
#             output_mask = torch.nn.functional.interpolate(latent_mask,size=(c1,h1,w1),mode='trilinear',align_corners=False)
#             output_mask[output_mask>0]=1
#             output_mask = rearrange(output_mask,'b t c h w->b t (c h w)').squeeze().bool()

#             logits_list.append(logits)
#             logits_masked_list.append(logits[:,output_mask,:].reshape(-1,logits.shape[-1]))
#             targets_masked_list.append(z_indices[1:2,output_mask].reshape(-1))
#             # batch_size_transformer=4
#             # logits_list = []
#             # for i in range(self.args.timepoints/4):
#             #     logits, _ = self.transformer(data_transformer, cbox=cbox)
#             #     logits_list.append(logits)
        
#             # logits_all = logits_list
#         logits_maksed_all= torch.cat(logits_masked_list)
#         target_maksed_all= torch.cat(targets_masked_list)   
#         target_maksed_embedding_all=self.first_stage_model.codebook.embeddings[target_maksed_all,:]         



#             # if save_nii and logits_list[idx].shape[0]>1:
#         with torch.no_grad():
#             if self.training:
#                 split='train'
#             else:
#                 split='val'
            
#             lung_masks_0 = lung_masks[0]
#             lung_masks_0 = lung_masks_0[:,None]
#             output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1),mode='trilinear',align_corners=False)
#             # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1))
#             output_mask[output_mask>0]=1
#             output_mask = rearrange(output_mask.squeeze(),'c h w->(c h w)').bool()

#             flat_inputs = logits_list[0][0,:]
#             distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
#                         - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
#                         + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
#             predicted_indices = torch.argmin(distances,dim=1)
#             predicted_indices = distances.min(1).indices 
#             baseline_indices = targets_list[0][0,:].clone()
#             baseline_indices[output_mask]=predicted_indices[output_mask]
#             predicted_indices = baseline_indices
#             predicted_indices = rearrange(predicted_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#             niis = self.first_stage_model.decode(predicted_indices.unsqueeze(0))
#             niis = niis.squeeze().cpu().numpy()
#             niis = niis.transpose(2,1,0)
#             niis = np.flip(niis,axis=1)
#             niis = np.flip(niis,axis=0)
#             image_type = 'reconstruction'
#             self.save_nii(self.logger.save_dir, split,image_type,niis ,
#                 self.global_step, self.current_epoch, patient_IDs[0], time_points_list[0][1])

#             if not all_time_points:
#                 target_indices = targets_list[idx][1,:]
#                 baseline_indices = targets_list[idx][0,:].clone()
#                 baseline_indices[output_mask]=target_indices[output_mask]
#                 target_indices = baseline_indices
#                 target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#                 niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
#                 niis = niis.squeeze().cpu().numpy()
#                 niis = niis.transpose(2,1,0)
#                 niis = np.flip(niis,axis=1)
#                 niis = np.flip(niis,axis=0)
#                 image_type = 'target'
#                 self.save_nii(self.logger.save_dir, split,image_type,niis,
#                     self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][1])
                
#                 target_indices = targets_list[idx][0,:]
#                 baseline_indices = targets_list[idx][0,:].clone()
#                 baseline_indices[output_mask]=target_indices[output_mask]
#                 target_indices = baseline_indices
#                 target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#                 niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
#                 niis = niis.squeeze().cpu().numpy()
#                 niis = niis.transpose(2,1,0)
#                 niis = np.flip(niis,axis=1)
#                 niis = np.flip(niis,axis=0)
#                 image_type = 'baseline'
#                 self.save_nii(self.logger.save_dir, split,image_type,niis,
#                     self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][0])
                            
#         return logits_maksed_all, target_maksed_all,target_maksed_embedding_all

# # only using video swin transformer for predicting next frame
#     def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, cbox=None,save_nii=False,all_time_points = False,patient_IDs=None):

#         logits_list = []
#         logits_masked_list = []
#         targets_list = []
#         targets_masked_list=[]
#         time_points_list = []
#         time_points_list_all = []
#         for idx in range(x.size()[0]):
#             temp = x[idx,:]
#             temp = temp[:,None]
#             vq_embeddings, z_indices = self.encode_to_z(temp)  
#             lung_masks_idx = lung_masks[idx,:]
#             lung_masks_idx = lung_masks_idx[:,None]
#             z_indices = z_indices + self.cond_stage_vocab_size

#             if self.training and self.pkeep < 1.0:
#                 mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
#                                                             device=z_indices.device))
#                 mask = mask.round().to(dtype=torch.int64)
#                 r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
#                 a_indices = mask*z_indices+(1-mask)*r_indices
#             else:
#                 a_indices = z_indices

#             observed_time_mask = ~torch.isnan(time_points[idx])
#             z_indices = z_indices[observed_time_mask,:]  

#             time_points_list.append(time_points[idx,observed_time_mask])        
#             targets_list.append(z_indices)
#             b,c1,h1,w1,d = vq_embeddings.size()
#             # latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
#             latent_mask = torch.nn.functional.interpolate(lung_masks_idx,size=(int(c1/self.scale),int(h1/self.scale),int(w1/self.scale)),mode='trilinear',align_corners=False)
#             latent_mask[latent_mask>0]=1
#             data_transformer = rearrange(vq_embeddings,'t c h w d -> t d c h w')
#             transformer_outputs=self.transformer(data_transformer[0:1,:])

#             output_mask = latent_mask.squeeze().bool()
#             logits = self.output_head(rearrange(transformer_outputs,'t d c h w  -> t c h w d'))
#             logits_list.append(logits)
#             logits_masked_list.append(logits[:,output_mask,:].reshape(-1,logits.shape[-1]))
#             targets_masked_list.append(z_indices[1:2,output_mask].reshape(-1))
#             # batch_size_transformer=4
#             # logits_list = []
#             # for i in range(self.args.timepoints/4):
#             #     logits, _ = self.transformer(data_transformer, cbox=cbox)
#             #     logits_list.append(logits)
        
#             # logits_all = logits_list
#         logits_maksed_all= torch.cat(logits_masked_list)
#         target_maksed_all= torch.cat(targets_masked_list)   
#         target_maksed_embedding_all=self.first_stage_model.codebook.embeddings[target_maksed_all,:]         



#         #     # if save_nii and logits_list[idx].shape[0]>1:
#         # with torch.no_grad():
#         #     if self.training:
#         #         split='train'
#         #     else:
#         #         split='val'
            
#         #     lung_masks_0 = lung_masks[0]
#         #     lung_masks_0 = lung_masks_0[:,None]
#         #     output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1),mode='trilinear',align_corners=False)
#         #     # output_mask = torch.nn.functional.interpolate(lung_masks_0,size=(c1,h1,w1))
#         #     output_mask[output_mask>0]=1
#         #     output_mask = rearrange(output_mask.squeeze(),'c h w->(c h w)').bool()

#         #     flat_inputs = logits_list[0][0,:]
#         #     distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
#         #                 - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
#         #                 + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
#         #     predicted_indices = torch.argmin(distances,dim=1)
#         #     predicted_indices = distances.min(1).indices 
#         #     baseline_indices = targets_list[0][0,:].clone()
#         #     baseline_indices[output_mask]=predicted_indices[output_mask]
#         #     predicted_indices = baseline_indices
#         #     predicted_indices = rearrange(predicted_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#         #     niis = self.first_stage_model.decode(predicted_indices.unsqueeze(0))
#         #     niis = niis.squeeze().cpu().numpy()
#         #     niis = niis.transpose(2,1,0)
#         #     niis = np.flip(niis,axis=1)
#         #     niis = np.flip(niis,axis=0)
#         #     image_type = 'reconstruction'
#         #     self.save_nii(self.logger.save_dir, split,image_type,niis ,
#         #         self.global_step, self.current_epoch, patient_IDs[0], time_points_list[0][1])

#         #     if not all_time_points:
#         #         target_indices = targets_list[idx][1,:]
#         #         baseline_indices = targets_list[idx][0,:].clone()
#         #         baseline_indices[output_mask]=target_indices[output_mask]
#         #         target_indices = baseline_indices
#         #         target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#         #         niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
#         #         niis = niis.squeeze().cpu().numpy()
#         #         niis = niis.transpose(2,1,0)
#         #         niis = np.flip(niis,axis=1)
#         #         niis = np.flip(niis,axis=0)
#         #         image_type = 'target'
#         #         self.save_nii(self.logger.save_dir, split,image_type,niis,
#         #             self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][1])
                
#         #         target_indices = targets_list[idx][0,:]
#         #         baseline_indices = targets_list[idx][0,:].clone()
#         #         baseline_indices[output_mask]=target_indices[output_mask]
#         #         target_indices = baseline_indices
#         #         target_indices = rearrange(target_indices,'(c h w)->c h w',c=c1,h=h1,w=w1)
#         #         niis = self.first_stage_model.decode(target_indices.unsqueeze(0))
#         #         niis = niis.squeeze().cpu().numpy()
#         #         niis = niis.transpose(2,1,0)
#         #         niis = np.flip(niis,axis=1)
#         #         niis = np.flip(niis,axis=0)
#         #         image_type = 'baseline'
#         #         self.save_nii(self.logger.save_dir, split,image_type,niis,
#         #             self.global_step, self.current_epoch,  patient_IDs[0], time_points_list[0][0])
                            
#         return logits_maksed_all, target_maksed_all,target_maksed_embedding_all
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    # # for ode+transformer
    # def latent_ODE_data_object(self,vq_embeddings,latent_mask,time_points,scale=4,batch_size_ode=1):
    #     # stack batch across l of different patients
    #     vq_embeddings=rearrange(vq_embeddings,'t (c s1) (h s2) (w s3) d-> (c h w) t (s1 s2 s3 d)',s1 =scale,s2 =scale,s3 =scale)
    #     latent_mask = rearrange(latent_mask.squeeze(1),'t l->l t')
    #     data_object_list = []
    #     observed_time_mask = ~torch.isnan(time_points)
    #     observed_data = vq_embeddings[:,observed_time_mask,:]       
    #     latent_mask = repeat(latent_mask,'l t->l (t repeat)',repeat=observed_data.shape[1])
    #     if observed_data.size()[1]>2:
    #         observed_mask = torch.zeros(observed_data.size(),device=observed_data.device)
    #         observed_mask[latent_mask,:]=1
    #         observed_mask[:,-1,:]=0           
    #     else:
    #         observed_mask = torch.zeros(observed_data.size(),device=observed_data.device)
    #         observed_mask[latent_mask,:]=1
    #     observed_mask_predicted = torch.zeros(observed_data.size(),device=observed_data.device)
    #     observed_mask_predicted[latent_mask,:]=1
    #     observed_data[observed_mask==0]=0
    #     for i in range(0,vq_embeddings.size()[0],batch_size_ode):
    #         observed_data_batch=observed_data[i:(i+batch_size_ode),:]
    #         batch_dict = {"tp_to_predict":torch.arange(0,self.timepoints,device=observed_data.device)*1.0,
    #                       "observed_data":observed_data_batch,
    #                       "observed_tp":time_points[observed_time_mask],
    #                       "observed_mask":observed_mask[i:(i+batch_size_ode),:],
    #                       "data_to_predict":observed_data_batch.clone(),
    #                       "mask_predicted_data":observed_mask_predicted[i:(i+batch_size_ode),:]
    #                       }
    #         data_object_list.append(batch_dict)

    #     return data_object_list

    # # # for only ode
    # def latent_ODE_data_object(self,vq_embeddings,latent_mask,time_points,scale=4,batch_size_ode=1):
    #     # stack batch across l of different patients
    #     vq_embeddings=rearrange(vq_embeddings,'t l d-> l t d')
    #     latent_mask = rearrange(latent_mask.squeeze(1),'t l->l t')
    #     data_object_list = []
    #     observed_time_mask = ~torch.isnan(time_points)
    #     observed_data = vq_embeddings[:,observed_time_mask,:]       
    #     latent_mask = repeat(latent_mask,'l t->l (t repeat)',repeat=observed_data.shape[1])
    #     # if observed_data.size()[1]>2:
    #     #     observed_mask = torch.zeros(observed_data.size(),device=observed_data.device)
    #     #     observed_mask[latent_mask,:]=1
    #     #     observed_mask[:,-1,:]=0           
    #     # else:
    #     #     observed_mask = torch.zeros(observed_data.size(),device=observed_data.device)
    #     #     observed_mask[latent_mask,:]=1
    #     observed_mask = torch.zeros(observed_data.size(),device=observed_data.device)
    #     observed_mask[latent_mask,:]=1

    #     observed_mask_predicted = torch.zeros(observed_data.size(),device=observed_data.device)
    #     observed_mask_predicted[latent_mask,:]=1
    #     observed_data[observed_mask==0]=0
    #     for i in range(0,vq_embeddings.size()[0],batch_size_ode):
    #         observed_data_batch=observed_data[i:(i+batch_size_ode),:]
    #         batch_dict = {"tp_to_predict":torch.arange(0,self.timepoints,device=observed_data.device)*1.0,
    #                       "observed_data":observed_data_batch,
    #                       "observed_tp":time_points[observed_time_mask],
    #                       "observed_mask":observed_mask[i:(i+batch_size_ode),:],
    #                       "data_to_predict":observed_data_batch.clone(),
    #                       "mask_predicted_data":observed_mask_predicted[i:(i+batch_size_ode),:]
    #                       }
    #         data_object_list.append(batch_dict)

    #     return data_object_list


 # # for video-ode
    def latent_ODE_data_object(self,vq_embeddings,z_indices,observed_time_mask,batch_time_points,scale=4,batch_size_ode=1):
        # stack batch across l of different patients
        # vq_embeddings=rearrange(vq_embeddings,'t l d-> l t d')
        # latent_mask = rearrange(latent_mask.squeeze(1),'t l->l t')
        # observed_tp = torch.unique(time_points.view(-1))
        # if torch.isnan(observed_tp[-1]):
        #     observed_tp =observed_tp[:-1] 
        # data_object_list = []
        time_steps = torch.unique(batch_time_points.view(-1))
        time_steps = time_steps[~torch.isnan(time_steps)]
        time_points_flag = torch.zeros(batch_time_points.shape[0],len(time_steps))
        for b in range(0,time_points_flag.shape[0]):
            time_points_flag[b,:]=sum(time_steps==t for t in batch_time_points[b,:])
        time_points_flag = time_points_flag.bool()

        observed_mask= time_points_flag.clone().unsqueeze(-1)
        observed_data = torch.zeros(vq_embeddings.shape[0],len(time_steps),vq_embeddings.shape[2],vq_embeddings.shape[3],vq_embeddings.shape[4],vq_embeddings.shape[5]).cuda()       
        z_indices_combined = torch.zeros(z_indices.shape[0],len(time_steps),z_indices.shape[2],z_indices.shape[3],z_indices.shape[4]).cuda().long()
        if self.mode == 'interpolation':
            mask_predicted_data = observed_mask.clone()
            
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,observed_mask.squeeze(-1).bool()[b,:],:]=vq_embeddings[b,observed_time_mask[b,:].bool(),:]
                z_indices_combined[b,observed_mask.squeeze(-1).bool()[b,:],:]=z_indices[b,observed_time_mask[b,:].bool(),:]
                if not self.training: 
                    if observed_mask[b,:].sum()>2:
                        pos = torch.where(observed_mask[b,:,:]==True)[0][-1]
                        observed_mask[b,:,:]=False
                        observed_mask[b,0,:]=True
                        observed_mask[b,pos,:]=True     
            # data_to_predict[~(mask_predicted_data.squeeze(-1)),:]=0 
            data_to_predict = observed_data.clone()          
            observed_data[~(observed_mask.squeeze(-1).bool()),:]=0 
           
        elif self.mode == 'extrapolation': 
            mask_predicted_data = observed_mask.clone()
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,observed_mask.squeeze(-1).bool()[b,:],:]=vq_embeddings[b,observed_time_mask[b,:].bool(),:]
                z_indices_combined[b,observed_mask.squeeze(-1).bool()[b,:],:]=z_indices[b,observed_time_mask[b,:].bool(),:]
                # pos = torch.where(observed_mask[b,:,:]==True)[0]              
                # mask_predicted_data[b,pos[0:round(len(pos)/2)],:]=False
                # observed_mask[b,pos[round(len(pos)/2):]]=False
                mask_predicted_data[b,0:2,:]=False
                observed_mask[b,2:]=False
            data_to_predict = observed_data.clone()  
            observed_data[~(observed_mask.squeeze(-1).bool()),:]=0  

        else:
            raise NotImplementedError 
        


        batch_dict = {"tp_to_predict":torch.arange(0,self.timepoints,device=observed_data.device)/self.timepoints,
                        "observed_data":observed_data,
                        "observed_tp":time_steps/self.timepoints,
                        "observed_mask":observed_mask.long(),
                        "data_to_predict":data_to_predict,
                        "mask_predicted_data":mask_predicted_data.long(),
                        "z_indices":z_indices_combined
                        }           

        return batch_dict

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                # x = x[:, :, ::self.sample_every_n_latent_frames]
                # targets = targets[:, ::self.sample_every_n_latent_frames]
                x = x[:, :, ::self.sample_every_n_latent_frames,::self.sample_every_n_latent_frames,::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames,::self.sample_every_n_latent_frames,::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            # targets = targets.reshape(targets.shape[0], -1)
        
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, indices = self.cond_stage_model.encode(c, include_embeddings=True)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        time_points = self.get_input('observed_time_points', batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c,time_points

    # def shared_step(self, batch, batch_idx,save_nii=False,all_time_points=False):
    #     if not self.vtokens:
    #         self.first_stage_model.eval()
    #     x, c,time_points = self.get_xc(batch)
    #     lung_masks = batch['longitudianl_lung_masks']
    #     patient_IDs = batch['patientID']
    #     if self.args.vtokens_pos:
    #         cbox = batch['cbox']
    #     else:
    #         cbox = None
    #     # print('train:', x.min(), x.max(), x.shape, c)
    #     logits, target = self(x, c, batch_idx,time_points, lung_masks,cbox,save_nii,all_time_points,patient_IDs)
    #     if not all_time_points:
    #         loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    #         acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
    #         return loss, acc1, acc5

    def shared_step(self, batch, batch_idx,save_nii=False,all_time_points=False):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c,time_points = self.get_xc(batch)
        lung_masks = batch['longitudianl_lung_masks']
        patient_IDs = batch['patientID']
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        # print('train:', x.min(), x.max(), x.shape, c)
        logits, target, target_embedding,ssim_step,psnr_step,mse_step = self(x, c, batch_idx,time_points, lung_masks,cbox,save_nii,all_time_points,patient_IDs)
        if not all_time_points:
            loss = F.mse_loss(logits, target_embedding)
            flat_inputs = logits
            distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                        - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
                        + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c]          
            acc1, acc5 = accuracy(-distances.reshape(-1, distances.shape[-1]), target.reshape(-1), topk=(1, 5))

            return loss, acc1, acc5,ssim_step,psnr_step,mse_step

    def training_step(self, batch, batch_idx):
        # print(batch['patientID'])
        if self.args.optimizer=='SAM':
            optimizer = self.optimizers()
            for model in self.modules():
                enable_running_stats(model) 
            # first forward-backward pass
            loss, acc1, acc5,ssim_step,psnr_step,mse_step = self.shared_step(batch, batch_idx)
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            for model in self.modules():
                disable_running_stats(model)
            loss_2, _, _ = self.shared_step(batch, batch_idx)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)
        else:
            loss, acc1, acc5,ssim_step,psnr_step,mse_step = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/ssim', ssim_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/psnr', psnr_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/mse', mse_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5,ssim_step,psnr_step,mse_step = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/ssim', ssim_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/psnr', psnr_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/mse', mse_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        # loss, acc1, acc5 =self.shared_step(batch, batch_idx,save_nii=False,all_time_points=True)
        # print(loss)
        self.shared_step(batch, batch_idx,save_nii=False,all_time_points=True)
        

    def save_nii(self, save_dir, split, image_type,niis,
                  global_step, current_epoch, patient_id, time_point):
        root = os.path.join(save_dir, "videos", split)
        print(root)
        filename = "{}_gs-{:04}_e-{:04}_p-{:04}_t-{:03}.nii.gz".format(
            image_type,
            global_step,
            current_epoch,
            patient_id,
            time_point)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        nib.save(nib.Nifti1Image(niis, np.eye(4)),path)


    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # # separate out all parameters to those that will and won't experience regularizing weight decay
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        # # for mn, m in self.transformer.named_modules():
        # for mn, m in chain(self.transformer.named_modules(),self.latentODE_model.named_modules(),self.output_head.named_modules()):
        # # for mn, m in chain(self.transformer.named_modules(),self.output_head.named_modules()):
        #     for pn, p in m.named_parameters():               
        #         fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        #         if pn.endswith('bias'):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #         elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)
        #         else:
        #             decay.add(fpn)

        # # special case the position embedding parameter in the root GPT module as not decayed
        # # no_decay.add('pos_emb')
        # if self.args.vtokens_pos:
        #     no_decay.add('vtokens_pos_emb')

        # # validate that we considered every parameter
        # # param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        # param_dict = {pn: p for pn, p in chain(self.transformer.named_parameters(),self.latentODE_model.named_parameters(),self.output_head.named_parameters())}
        # # param_dict = {pn: p for pn, p in chain(self.transformer.named_parameters(),self.output_head.named_parameters())}    
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )

        # # create the pytorch optimizer object
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        #     {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        # ]
        # if self.args.optimizer=='SAM':
        #     base_optimizer = torch.optim.AdamW  # define an optimizer for the "sharpness-aware" update
        #     optimizer = SAM(optim_groups, base_optimizer, lr=self.learning_rate, betas=(0.9, 0.95))
        # else:
        #     optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        # # # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(chain(self.latentODE_model.parameters(),self.output_head.parameters()), lr=self.learning_rate, betas=(0.9, 0.95))
        # optim_groups = [
        #     {"params": self.transformer.parameters(), "lr": self.learning_rate},
        #     {"params": self.output_head.parameters(), "lr": self.learning_rate*10},
        # ]
        # optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))
        return optimizer


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--stft_vqvae', type=str, help='path to vqgan ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        # VideoGPT hyperparmeters
        parser.add_argument('--vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)
        parser.add_argument('--block_size', type=int, default=256)
        parser.add_argument('--n_layer', type=int, default=48)
        parser.add_argument('--n_head', type=int, default=24)
        parser.add_argument('--n_embd', type=int, default=1536)
        parser.add_argument('--n_unmasked', type=int, default=0)
        parser.add_argument('--sample_every_n_latent_frames', type=int, default=0)
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
        # latent ode hyperparameters
        parser.add_argument('--z0-encoder', type=str, default='rnn', help="Type of encoder for Latent ODE model: odernn or rnn")
        parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
        parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
        parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
        parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
        parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
        parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
        parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
        # parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
        parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
        parser.add_argument('--batch_size_ode', type=int, default=64)
        parser.add_argument('--scale', type=int, default=4)
        
        return parser

