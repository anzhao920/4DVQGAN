# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules.utils import shift_dim, accuracy, comp_getattr, ForkedPdb,enable_running_stats,disable_running_stats
from .modules.encoders import Labelator, SOSProvider, Identity
from einops import rearrange,repeat
import os
import nibabel as nib
import numpy as np
from itertools import chain
from .modules.conv_odegru import VidODE
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def load_vqgan(vqgan_ckpt, device=torch.device('cpu')):
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).to(device)
    vqgan.eval()

    return vqgan

class VQGAN_4D(pl.LightningModule):
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
        self.init_first_stage_from_ckpt(args)
        self.init_cond_stage_from_ckpt(args)

        self.embedding_dim=args.embedding_dim
        self.input_dim = self.embedding_dim*args.scale**3
        self.scale = args.scale
        self.batch_size_ode = args.batch_size_ode
        self.timepoints = args.timepoints
        temp_device = torch.device('cuda')
        self.latentODE_model = VidODE(args, args.embedding_dim, temp_device)
        
        self.output_head = nn.Sequential(
		   nn.Linear(16, int(args.embedding_dim*2)),
		   nn.Tanh(),
		   nn.Linear(int(args.embedding_dim*2), args.embedding_dim),)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        self.save_hyperparameters()
        if self.args.optimizer == 'SAM':
            self.automatic_optimization = False
        self.loss_list = []
        self.acc1_sum_list = []
        self.acc5_sum_list = []
        self.ssim_sum_list = []
        self.psnr_sum_list = []
        self.mse_sum_list = []
        self.pixelNum_sum_list = []

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
 
    def init_cond_stage_from_ckpt(self, args):
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

    def forward(self, x, c, batch_idx,time_points=None, lung_masks=None, save_nii=False,patient_IDs=None,validation_mode='only_observed',img_paths=None):
        b,t,d,h,w=x.shape
        with torch.no_grad():
            vq_embeddings, z_indices = self.encode_to_z(rearrange(x,'b t (c d) h w->(b t) c d h w',c=1))
            vq_embeddings = rearrange(vq_embeddings,'(b t) d h w c->b t c d h w',b=b)
            z_indices = rearrange(z_indices,'(b t) d h w->b t d h w',b=b)
            observed_time_mask = (~torch.isnan(time_points)).long()
            z_indices = z_indices + self.cond_stage_vocab_size
            b, t, c, d, h, w = vq_embeddings.size()
            latent_mask = torch.nn.functional.interpolate(lung_masks,size=(d,h,w),mode='trilinear',align_corners=False)
            latent_mask[latent_mask>0]=1
            
            batch_dict = self.latent_ODE_data_object(vq_embeddings,z_indices,observed_time_mask,time_points,latent_mask)
            z_indices = batch_dict['z_indices']     
            temp_mask = repeat(latent_mask.bool(),'b t d h w->b (r t) d h w',r=z_indices.shape[1])[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
            z_indices_target_flat = z_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:][temp_mask]
        
        logits,index_selected,true_intermediates,pred_intermediates = self.latentODE_model.compute_all_losses(batch_dict)
        weights = torch.linspace(1, 1, steps=self.args.timepoints)
        weights = weights[index_selected]
        b1,t1, c1, d1, h1, w1 = logits.shape
        weights = repeat(weights,'t->b t d h w c',b=b, d=d1,h=h1,w=w1,c=c1)
        weights = weights[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        logits = rearrange(logits,'b t c d h w->b t d h w c')
        logits_flat = logits[:,temp_mask,:]
        true_intermediates = rearrange(true_intermediates,'b t c d h w->b t d h w c')
        true_intermediates = true_intermediates[:,temp_mask[1:,:],:]
        true_intermediates = true_intermediates.detach()
        pred_intermediates = rearrange(pred_intermediates,'b t c d h w->b t d h w c')
        pred_intermediates = pred_intermediates[:,temp_mask[1:,:],:]  

        logits_maksed_all = logits_flat
        target_maksed_all = z_indices_target_flat.detach()
        target_maksed_embedding_all=self.first_stage_model.codebook.embeddings[target_maksed_all,:].detach()
        ssim_sum=0
        psnr_sum=0
        mse_sum = 0
        pixelNum_sum =1

        distances = (logits_flat.squeeze(0) ** 2).sum(dim=1, keepdim=True) \
                    - 2 * logits_flat.squeeze(0) @ self.first_stage_model.codebook.embeddings.t() \
                    + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c] 
        predicted_indices = torch.argmin(distances,dim=1)
    
        if validation_mode != "only_observed":
            csv_file_path = validation_mode+'_histogram_predicted.csv'
            predicted_indeices_with_outside = torch.zeros_like(z_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:])
            predicted_indeices_with_outside[temp_mask]=predicted_indices
            for i in range(0,temp_mask.shape[0]):
                temp = predicted_indeices_with_outside[i,temp_mask[i,:]]
        baseline_indices =  z_indices[:,0,:].clone()
        baseline_indices = repeat(baseline_indices.unsqueeze(1),'b t c h w->b (r t) c h w',r=z_indices.shape[1])
        baseline_indices=baseline_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        if b==1:
            baseline_indices=baseline_indices.unsqueeze(0)
        baseline_indices[:,temp_mask]=predicted_indices
        predicted_indices = baseline_indices.clone()
        input_CTs = x[observed_time_mask.bool(),:][batch_dict['mask_predicted_data'].squeeze().bool(),:]
        reconstructed_CTs = torch.zeros(input_CTs.shape).cuda()
        lung_masks = repeat(lung_masks.bool(),'b t c h w->b (r t) c h w',r=z_indices.shape[1])[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        mse_loss = nn.MSELoss(reduction='sum')
        psnr = PeakSignalNoiseRatio().cuda()
        mse_sum=0
        psnr_sum=0
        ssim_sum = 0
        pixelNum_sum = 0
        ms_ssim_sum = 0
        padding = nn.ReplicationPad3d(5)
        
        predicted_indices=predicted_indices[0,:]
        if not self.training:
            with torch.no_grad():
                for i in range(0,predicted_indices.shape[0]):
                    reconstructed_CTs[i,:]= self.first_stage_model.decode(predicted_indices[i:i+1,:])            
                    mse_sum=mse_sum+mse_loss(reconstructed_CTs[i,lung_masks[i]],input_CTs[i,lung_masks[i]]).detach()
                    psnr_sum = psnr_sum+psnr(reconstructed_CTs[i,lung_masks[i]],input_CTs[i,lung_masks[i]]).detach()*lung_masks[i].sum()
                    if predicted_indices.shape[1]==1:
                        _,ssim_map=ssim(padding(reconstructed_CTs[i:i+1]), padding(input_CTs[i:i+1]), data_range=1, size_average=False)
                        ssim_sum =ssim_sum+ssim_map[:,lung_masks[i]].sum()
                    else:
                        _,ssim_map=ssim(padding(reconstructed_CTs[i:i+1].unsqueeze(0)), padding(input_CTs[i:i+1].unsqueeze(0)), data_range=1, size_average=False)
                        ssim_sum =ssim_sum+ssim_map[:,:,lung_masks[i]].sum()
                    pixelNum_sum = pixelNum_sum+lung_masks[i].sum()

                    if save_nii:
                        niis=reconstructed_CTs[i,:]-(reconstructed_CTs[i,0,0,0]+0.5)
                        niis = torch.clamp(niis,-0.5,0.5)
                        niis = niis.squeeze().cpu().numpy()
                        niis = niis.transpose(2,1,0)
                        niis = np.flip(niis,axis=1)
                        niis = np.flip(niis,axis=0)
                        mode = self.args.mode
                        image_type = 'predicted'
                        self.save_nii(self.logger.log_dir, mode,image_type,niis,
                            patient_IDs[0], time_points[0,i])

                        niis = input_CTs[i,:]
                        niis = niis.squeeze().cpu().numpy()
                        niis = niis.transpose(2,1,0)
                        niis = np.flip(niis,axis=1)
                        niis = np.flip(niis,axis=0)
                        mode = self.args.mode
                        image_type = 'target'
                        self.save_nii(self.logger.log_dir, mode,image_type,niis,
                            patient_IDs[0], time_points[0,i])                
        
                    ms_ssim_sum = ms_ssim_sum+ms_ssim(padding(reconstructed_CTs[i:i+1].unsqueeze(0)), padding(input_CTs[i:i+1].unsqueeze(0)), data_range=1, size_average=False) 

        return logits_maksed_all, target_maksed_all,target_maksed_embedding_all,ssim_sum,psnr_sum,mse_sum,pixelNum_sum,pred_intermediates,true_intermediates 
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out


 # # for video-ode
    def latent_ODE_data_object(self,vq_embeddings,z_indices,observed_time_mask,batch_time_points,latent_mask):
        # observed_time_mask is the mask indicating which time points we have CT scans, size = [1,max_time_points]

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
                pos = torch.where(observed_mask[b,:,:]==True)[0][-1]
                observed_mask[b,:,:]=False
                observed_mask[b,0,:]=True
                observed_mask[b,pos,:]=True

            data_to_predict = observed_data.clone()          
            observed_data[~(observed_mask.squeeze(-1).bool()),:]=0

            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0 

        elif self.mode == 'extrapolation': 
            mask_predicted_data = observed_mask.clone()
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,observed_mask.squeeze(-1).bool()[b,:],:]=vq_embeddings[b,observed_time_mask[b,:].bool(),:]
                z_indices_combined[b,observed_mask.squeeze(-1).bool()[b,:],:]=z_indices[b,observed_time_mask[b,:].bool(),:]
                pos = torch.where(observed_mask[b,:,:]==True)[0]              
                observed_mask[b,:,:]=False
                observed_mask[b,pos[0:2]]=True
            data_to_predict = observed_data.clone()          
            observed_data[~(observed_mask.squeeze(-1).bool()),:]=0
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0 

        elif self.mode == 'reconstruction':
            mask_predicted_data = observed_mask.clone()
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,observed_mask.squeeze(-1).bool()[b,:],:]=vq_embeddings[b,observed_time_mask[b,:].bool(),:]
                z_indices_combined[b,observed_mask.squeeze(-1).bool()[b,:],:]=z_indices[b,observed_time_mask[b,:].bool(),:]
            data_to_predict = observed_data.clone()  
            observed_data[~(observed_mask.squeeze(-1).bool()),:]=0  
            for b in range(0,observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0    


                                                                
        else: 
            raise NotImplementedError 
        batch_dict = {"tp_to_predict":torch.arange(0,self.timepoints,device=observed_data.device)/self.timepoints,
                        "observed_data":observed_data,
                        "observed_tp":time_steps/self.timepoints,
                        "observed_mask":observed_mask.long(),
                        "data_to_predict":data_to_predict,
                        "mask_predicted_data":mask_predicted_data.long(),
                        "z_indices":z_indices_combined,
                        "latent_mask":latent_mask
                        }           

        return batch_dict

    @torch.no_grad()
    def encode_to_z(self, x):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)
        
        return x, targets


    def get_input(self, key, batch):
        x = batch[key]
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        time_points = self.get_input('observed_time_points', batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c,time_points

    def shared_step(self, batch, batch_idx,save_nii=False,validation_mode='only_observed'):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c,time_points = self.get_xc(batch)
        lung_masks = batch['longitudianl_lung_masks']
        patient_IDs = batch['patientID']
        img_paths = batch['patient_img_path_list']
        logits, target, target_embedding,ssim_sum,psnr_sum,mse_sum,pixelNum_sum,pred_intermediates,true_intermediates = self(x, c, batch_idx,time_points, lung_masks,save_nii,patient_IDs,validation_mode,img_paths)
        logits = logits.squeeze(0)
        loss = 0.2*F.mse_loss(logits, target_embedding)+F.mse_loss(pred_intermediates, true_intermediates)
        flat_inputs = logits
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
                    + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c]          
        acc1, acc5 = accuracy(-distances.reshape(-1, distances.shape[-1]), target.reshape(-1), topk=(1, 5))
        return loss, acc1, acc5,ssim_sum,psnr_sum,mse_sum,pixelNum_sum

    def training_step(self, batch, batch_idx):

        loss, acc1, acc5,ssim_sum,psnr_sum,mse_sum,pixelNum_sum = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log('train/ssim', ssim_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log('train/psnr', psnr_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log('train/mse', mse_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5,ssim_sum,psnr_sum,mse_sum,pixelNum_sum = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/ssim', ssim_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/psnr', psnr_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/mse', mse_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc1, acc5,ssim_sum,psnr_sum,mse_sum,pixelNum_sum = self.shared_step(batch, batch_idx,save_nii=True)
        self.loss_list.append(loss.detach().cpu())
        self.acc1_sum_list.append((acc1*pixelNum_sum).detach().cpu())
        self.acc5_sum_list.append((acc5*pixelNum_sum).detach().cpu())
        # self.ssim_sum_list.append(ssim_sum.detach().cpu())
        # self.psnr_sum_list.append(psnr_sum.detach().cpu())
        # self.mse_sum_list.append(mse_sum.detach().cpu())
        # self.pixelNum_sum_list.append(pixelNum_sum.detach().cpu())




        

    def save_nii(self, save_dir, split, image_type,niis,
                   patient_id, time_point):
        root = os.path.join(save_dir, "videos", split)
        print(root)
        filename = "{}_gs-_patient-{:04}_t-{:03}.nii.gz".format(
            image_type,
            patient_id,
            time_point)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        nib.save(nib.Nifti1Image(niis, np.eye(4)),path)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(chain(self.latentODE_model.parameters(),self.output_head.parameters()), lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--stft_vqvae', type=str, help='path to vqgan ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        parser.add_argument('--classification', action='store_true', default=False)
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
        parser.add_argument('--n_layers', type=int, default=3, help='A number of layer of vid ODE func')
        parser.add_argument('--n_downs', type=int, default=2)
        parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
        parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
        parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
        # parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
        parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
        parser.add_argument('--batch_size_ode', type=int, default=64)
        parser.add_argument('--scale', type=int, default=4)
        
        return parser

