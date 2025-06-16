# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules.utils import shift_dim, accuracy
from .modules.encoders import Labelator, SOSProvider, Identity
from einops import rearrange,repeat
import os
import nibabel as nib
import numpy as np
from itertools import chain
from .modules.conv_odegru import Latent_embedding_ODE
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from typing import Dict, List, Tuple, Optional

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
        self.latentODE_model = Latent_embedding_ODE(args, args.embedding_dim, temp_device)
        
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

    def visualize_predictions(self,
                             predicted_indices: torch.Tensor,
                             z_indices: torch.Tensor,
                             lung_masks: torch.Tensor,
                             time_points: torch.Tensor,
                             patient_IDs: List[int],
                             save_nii: bool = False) -> Dict[str, torch.Tensor]:
        """
        Visualize and evaluate model predictions.
        
        Args:
            predicted_indices: Predicted indices from the model
            z_indices: Target indices
            lung_masks: Binary masks for lung regions
            time_points: Time points for each scan
            patient_IDs: Patient identifiers
            save_nii: Whether to save results as NIfTI files
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Initialize metrics
        metrics = {
            'mse_sum': torch.tensor(0.0, device=self.device),
            'psnr_sum': torch.tensor(0.0, device=self.device),
            'ssim_sum': torch.tensor(0.0, device=self.device),
            'ms_ssim_sum': torch.tensor(0.0, device=self.device),
            'pixelNum_sum': torch.tensor(0.0, device=self.device)
        }
        
        # Initialize metrics calculators
        mse_loss = nn.MSELoss(reduction='sum')
        psnr = PeakSignalNoiseRatio().to(self.device)
        padding = nn.ReplicationPad3d(5)
        
        # Decode predictions
        reconstructed_CTs = torch.zeros_like(self.first_stage_model.decode(predicted_indices[0:1]))
        input_CTs = self.first_stage_model.decode(z_indices[0:1])
        
        with torch.no_grad():
            for i in range(predicted_indices.shape[0]):
                # Decode current prediction
                reconstructed_CTs[i,:] = self.first_stage_model.decode(predicted_indices[i:i+1,:])
                
                # Calculate MSE
                metrics['mse_sum'] += mse_loss(
                    reconstructed_CTs[i, lung_masks[i]], 
                    input_CTs[i, lung_masks[i]]
                )
                
                # Calculate PSNR
                metrics['psnr_sum'] += psnr(
                    reconstructed_CTs[i, lung_masks[i]], 
                    input_CTs[i, lung_masks[i]]
                ) * lung_masks[i].sum()
                
                # Calculate SSIM
                if predicted_indices.shape[1] == 1:
                    _, ssim_map = ssim(
                        padding(reconstructed_CTs[i:i+1]), 
                        padding(input_CTs[i:i+1]), 
                        data_range=1, 
                        size_average=False
                    )
                    metrics['ssim_sum'] += ssim_map[:, lung_masks[i]].sum()
                else:
                    _, ssim_map = ssim(
                        padding(reconstructed_CTs[i:i+1].unsqueeze(0)), 
                        padding(input_CTs[i:i+1].unsqueeze(0)), 
                        data_range=1, 
                        size_average=False
                    )
                    metrics['ssim_sum'] += ssim_map[:, :, lung_masks[i]].sum()
                
                metrics['pixelNum_sum'] += lung_masks[i].sum()
                
                # Calculate MS-SSIM
                metrics['ms_ssim_sum'] += ms_ssim(
                    padding(reconstructed_CTs[i:i+1].unsqueeze(0)), 
                    padding(input_CTs[i:i+1].unsqueeze(0)), 
                    data_range=1, 
                    size_average=False
                )
                
                # Save predictions if requested
                if save_nii:
                    # Save predicted scan
                    pred_nii = reconstructed_CTs[i,:] - (reconstructed_CTs[i,0,0,0] + 0.5)
                    pred_nii = torch.clamp(pred_nii, -0.5, 0.5)
                    pred_nii = pred_nii.squeeze().cpu().numpy()
                    pred_nii = pred_nii.transpose(2,1,0)
                    pred_nii = np.flip(pred_nii, axis=(0,1))
                    
                    self.save_nii(
                        self.logger.log_dir,
                        self.args.mode,
                        'predicted',
                        pred_nii,
                        patient_IDs[0],
                        time_points[0,i]
                    )
                    
                    # Save ground truth scan
                    target_nii = input_CTs[i,:]
                    target_nii = target_nii.squeeze().cpu().numpy()
                    target_nii = target_nii.transpose(2,1,0)
                    target_nii = np.flip(target_nii, axis=(0,1))
                    
                    self.save_nii(
                        self.logger.log_dir,
                        self.args.mode,
                        'target',
                        target_nii,
                        patient_IDs[0],
                        time_points[0,i]
                    )
        
        return metrics

    def forward(self, 
                x: torch.Tensor,
                c: torch.Tensor,
                batch_idx: int,
                time_points: Optional[torch.Tensor] = None,
                lung_masks: Optional[torch.Tensor] = None,
                save_nii: bool = False,
                patient_IDs: Optional[List[int]] = None,
                validation_mode: str = 'only_observed',
                img_paths: Optional[List[str]] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (b, t, d, h, w)
            c: Conditional input
            batch_idx: Batch index
            time_points: Time points for each scan
            lung_masks: Binary masks for lung regions
            save_nii: Whether to save results as NIfTI files
            patient_IDs: Patient identifiers
            validation_mode: Validation mode
            img_paths: Paths to input images
            
        Returns:
            Tuple containing model outputs and metrics
        """
        b, t, d, h, w = x.shape
        
        # Encode input to latent space
        with torch.no_grad():
            vq_embeddings, z_indices = self.encode_to_z(
                rearrange(x, 'b t (c d) h w->(b t) c d h w', c=1)
            )
            vq_embeddings = rearrange(vq_embeddings, '(b t) d h w c->b t c d h w', b=b)
            z_indices = rearrange(z_indices, '(b t) d h w->b t d h w', b=b)
            
            # Process masks and time points
            observed_time_mask = (~torch.isnan(time_points)).long()
            z_indices = z_indices + self.cond_stage_vocab_size
            b, t, c, d, h, w = vq_embeddings.size()
            
            # Resize lung masks to match latent space
            latent_mask = F.interpolate(
                lung_masks,
                size=(d, h, w),
                mode='trilinear',
                align_corners=False
            )
            latent_mask[latent_mask > 0] = 1
            
            # Prepare data for ODE model
            batch_dict = self.latent_ODE_data_object(
                vq_embeddings,
                z_indices,
                observed_time_mask,
                time_points,
                latent_mask
            )
            
            # Update z_indices from batch_dict
            z_indices = batch_dict['z_indices']
            
            # Process masks for prediction
            temp_mask = repeat(
                latent_mask.bool(),
                'b t d h w->b (r t) d h w',
                r=z_indices.shape[1]
            )[batch_dict['mask_predicted_data'].squeeze(-1).bool(), :]
            
            z_indices_target_flat = z_indices[
                batch_dict['mask_predicted_data'].squeeze(-1).bool(),
                :
            ][temp_mask]
        
        # Get predictions from ODE model
        logits, index_selected, true_intermediates, pred_intermediates = self.latentODE_model.compute_all_losses(batch_dict)
        
        # Process weights and reshape tensors
        weights = torch.linspace(1, 1, steps=self.args.timepoints)[index_selected]
        b1, t1, c1, d1, h1, w1 = logits.shape
        weights = repeat(weights, 't->b t d h w c', b=b, d=d1, h=h1, w=w1, c=c1)
        weights = weights[batch_dict['mask_predicted_data'].squeeze(-1).bool(), :]
        
        # Reshape tensors
        logits = rearrange(logits, 'b t c d h w->b t d h w c')
        logits_flat = logits[:, temp_mask, :]
        true_intermediates = rearrange(true_intermediates, 'b t c d h w->b t d h w c')
        true_intermediates = true_intermediates[:, temp_mask[1:, :], :].detach()
        pred_intermediates = rearrange(pred_intermediates, 'b t c d h w->b t d h w c')
        pred_intermediates = pred_intermediates[:, temp_mask[1:, :], :]
        
        # Process predictions
        batch_dict.update({
            'b': b, 't': t, 'c': c, 'd': d, 'h': h, 'w': w,
            'index_selected': index_selected,
            'true_intermediates': true_intermediates,
            'pred_intermediates': pred_intermediates,
            'temp_mask': temp_mask
        })
        
        # Calculate distances and get predicted indices
        distances = (logits_flat.squeeze(0) ** 2).sum(dim=1, keepdim=True) \
                   - 2 * logits_flat.squeeze(0) @ self.first_stage_model.codebook.embeddings.t() \
                   + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        predicted_indices = torch.argmin(distances, dim=1)
        
        # Handle validation mode
        if validation_mode != "only_observed":
            predicted_indices_with_outside = torch.zeros_like(z_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:])
            predicted_indices_with_outside[temp_mask] = predicted_indices
            for i in range(temp_mask.shape[0]):
                temp = predicted_indices_with_outside[i, temp_mask[i,:]]
        
        # Process baseline indices
        baseline_indices = z_indices[:,0,:].clone()
        baseline_indices = repeat(baseline_indices.unsqueeze(1), 'b t c h w->b (r t) c h w', r=z_indices.shape[1])
        baseline_indices = baseline_indices[batch_dict['mask_predicted_data'].squeeze(-1).bool(),:]
        if b == 1:
            baseline_indices = baseline_indices.unsqueeze(0)
        baseline_indices[:,temp_mask] = predicted_indices
        predicted_indices = baseline_indices.clone()
        
        # Calculate metrics if not training
        if not self.training:
            metrics = self.visualize_predictions(
                predicted_indices=predicted_indices,
                z_indices=z_indices,
                lung_masks=lung_masks,
                time_points=time_points,
                patient_IDs=patient_IDs,
                save_nii=save_nii
            )
        else:
            metrics = {
                'mse_sum': torch.tensor(0.0, device=self.device),
                'psnr_sum': torch.tensor(0.0, device=self.device),
                'ssim_sum': torch.tensor(0.0, device=self.device),
                'ms_ssim_sum': torch.tensor(0.0, device=self.device),
                'pixelNum_sum': torch.tensor(0.0, device=self.device)
            }
        
        return (
            logits_flat,
            z_indices_target_flat,
            self.first_stage_model.codebook.embeddings[z_indices_target_flat, :].detach(),
            metrics['ssim_sum'],
            metrics['psnr_sum'],
            metrics['mse_sum'],
            metrics['pixelNum_sum'],
            pred_intermediates,
            true_intermediates
        )

    def latent_ODE_data_object(self,
                             vq_embeddings: torch.Tensor,
                             z_indices: torch.Tensor,
                             observed_time_mask: torch.Tensor,
                             batch_time_points: torch.Tensor,
                             latent_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare data for the latent ODE model.
        
        Args:
            vq_embeddings: VQ embeddings tensor
            z_indices: Latent indices tensor
            observed_time_mask: Mask indicating observed time points
            batch_time_points: Time points for each scan
            latent_mask: Binary mask for latent space
            
        Returns:
            Dictionary containing processed data for ODE model
        """
        # Get unique time steps
        time_steps = torch.unique(batch_time_points.view(-1))
        time_steps = time_steps[~torch.isnan(time_steps)]
        
        # Create time points flag
        time_points_flag = torch.zeros(batch_time_points.shape[0], len(time_steps), device=batch_time_points.device)
        for b in range(time_points_flag.shape[0]):
            time_points_flag[b,:] = sum(time_steps == t for t in batch_time_points[b,:])
        time_points_flag = time_points_flag.bool()

        # Initialize tensors
        observed_mask = time_points_flag.clone().unsqueeze(-1)
        observed_data = torch.zeros(
            vq_embeddings.shape[0],
            len(time_steps),
            vq_embeddings.shape[2],
            vq_embeddings.shape[3],
            vq_embeddings.shape[4],
            vq_embeddings.shape[5],
            device=vq_embeddings.device
        )
        z_indices_combined = torch.zeros(
            z_indices.shape[0],
            len(time_steps),
            z_indices.shape[2],
            z_indices.shape[3],
            z_indices.shape[4],
            device=z_indices.device,
            dtype=torch.long
        )

        # Process data based on mode
        if self.mode == 'interpolation':
            mask_predicted_data = observed_mask.clone()
            
            for b in range(observed_time_mask.shape[0]):
                observed_data[b, observed_mask.squeeze(-1).bool()[b,:], :] = vq_embeddings[b, observed_time_mask[b,:].bool(), :]
                z_indices_combined[b, observed_mask.squeeze(-1).bool()[b,:], :] = z_indices[b, observed_time_mask[b,:].bool(), :]
                pos = torch.where(observed_mask[b,:,:] == True)[0][-1]
                observed_mask[b,:,:] = False
                observed_mask[b,0,:] = True
                observed_mask[b,pos,:] = True

            data_to_predict = observed_data.clone()
            observed_data[~(observed_mask.squeeze(-1).bool()), :] = 0

            for b in range(observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0

        elif self.mode == 'extrapolation':
            mask_predicted_data = observed_mask.clone()
            for b in range(observed_time_mask.shape[0]):
                observed_data[b, observed_mask.squeeze(-1).bool()[b,:], :] = vq_embeddings[b, observed_time_mask[b,:].bool(), :]
                z_indices_combined[b, observed_mask.squeeze(-1).bool()[b,:], :] = z_indices[b, observed_time_mask[b,:].bool(), :]
                pos = torch.where(observed_mask[b,:,:] == True)[0]
                observed_mask[b,:,:] = False
                observed_mask[b,pos[0:2]] = True
                
            data_to_predict = observed_data.clone()
            observed_data[~(observed_mask.squeeze(-1).bool()), :] = 0
            
            for b in range(observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0

        elif self.mode == 'reconstruction':
            mask_predicted_data = observed_mask.clone()
            for b in range(observed_time_mask.shape[0]):
                observed_data[b, observed_mask.squeeze(-1).bool()[b,:], :] = vq_embeddings[b, observed_time_mask[b,:].bool(), :]
                z_indices_combined[b, observed_mask.squeeze(-1).bool()[b,:], :] = z_indices[b, observed_time_mask[b,:].bool(), :]
                
            data_to_predict = observed_data.clone()
            observed_data[~(observed_mask.squeeze(-1).bool()), :] = 0
            
            for b in range(observed_time_mask.shape[0]):
                observed_data[b,:,:,~(latent_mask[b,0,:].bool())] = 0
                data_to_predict[b,:,:,~(latent_mask[b,0,:].bool())] = 0
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        # Create and return batch dictionary
        batch_dict = {
            "tp_to_predict": torch.arange(0, self.timepoints, device=observed_data.device) / self.timepoints,
            "observed_data": observed_data,
            "observed_tp": time_steps / self.timepoints,
            "observed_mask": observed_mask.long(),
            "data_to_predict": data_to_predict,
            "mask_predicted_data": mask_predicted_data.long(),
            "z_indices": z_indices_combined,
            "latent_mask": latent_mask
        }

        return batch_dict

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (embeddings, indices)
        """
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
            return x, targets
        else:
            x, targets = self.first_stage_model.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)
            return x, targets

    def get_input(self, key: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get input from batch dictionary.
        
        Args:
            key: Key to get from batch
            batch: Batch dictionary
            
        Returns:
            Input tensor
        """
        return batch[key]

    def get_xc(self, batch: Dict[str, torch.Tensor], N: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get input and conditional tensors from batch.
        
        Args:
            batch: Batch dictionary
            N: Optional number of samples to get
            
        Returns:
            Tuple of (input tensor, conditional tensor, time points)
        """
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        time_points = self.get_input('observed_time_points', batch)
        
        if N is not None:
            x = x[:N]
            c = c[:N]
            
        return x, c, time_points

    def shared_step(self,
                   batch: Dict[str, torch.Tensor],
                   batch_idx: int,
                   save_nii: bool = False,
                   validation_mode: str = 'only_observed') -> Tuple[torch.Tensor, ...]:
        """
        Shared step for training and validation.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            save_nii: Whether to save results as NIfTI files
            validation_mode: Validation mode
            
        Returns:
            Tuple of (loss, accuracy1, accuracy5, ssim, psnr, mse, pixel_num)
        """
        if not self.vtokens:
            self.first_stage_model.eval()
            
        x, c, time_points = self.get_xc(batch)
        lung_masks = batch['longitudianl_lung_masks']
        patient_IDs = batch['patientID']
        img_paths = batch['patient_img_path_list']
        
        logits, target, target_embedding, ssim_sum, psnr_sum, mse_sum, pixelNum_sum, pred_intermediates, true_intermediates = self(
            x, c, batch_idx, time_points, lung_masks, save_nii, patient_IDs, validation_mode, img_paths
        )
        
        logits = logits.squeeze(0)
        loss = 0.2 * F.mse_loss(logits, target_embedding) + F.mse_loss(pred_intermediates, true_intermediates)
        
        flat_inputs = logits
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                   - 2 * flat_inputs @ self.first_stage_model.codebook.embeddings.t() \
                   + (self.first_stage_model.codebook.embeddings.t() ** 2).sum(dim=0, keepdim=True)
                   
        acc1, acc5 = accuracy(-distances.reshape(-1, distances.shape[-1]), target.reshape(-1), topk=(1, 5))
        
        return loss, acc1, acc5, ssim_sum, psnr_sum, mse_sum, pixelNum_sum

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        loss, acc1, acc5, ssim_sum, psnr_sum, mse_sum, pixelNum_sum = self.shared_step(batch, batch_idx)
        
        self.log("train/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        loss, acc1, acc5, ssim_sum, psnr_sum, mse_sum, pixelNum_sum = self.shared_step(batch, batch_idx)
        
        self.log("val/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5.detach().item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/ssim', ssim_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/psnr', psnr_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/mse', mse_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        loss, acc1, acc5, ssim_sum, psnr_sum, mse_sum, pixelNum_sum = self.shared_step(batch, batch_idx, save_nii=True)
        
        # Store metrics for aggregation
        self.loss_list.append(loss.detach().cpu())
        self.acc1_sum_list.append((acc1*pixelNum_sum).detach().cpu())
        self.acc5_sum_list.append((acc5*pixelNum_sum).detach().cpu())
        self.ssim_sum_list.append(ssim_sum.detach().cpu())
        self.psnr_sum_list.append(psnr_sum.detach().cpu())
        self.mse_sum_list.append(mse_sum.detach().cpu())
        self.pixelNum_sum_list.append(pixelNum_sum.detach().cpu())
        
        # Log step metrics
        self.log('test/loss', loss.detach().item(), prog_bar=True, logger=True, on_step=True)
        self.log('test/acc1', acc1.detach().item(), prog_bar=True, logger=True, on_step=True)
        self.log('test/acc5', acc5.detach().item(), prog_bar=True, logger=True, on_step=True)
        self.log('test/ssim', ssim_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True)
        self.log('test/psnr', psnr_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True)
        self.log('test/mse', mse_sum/pixelNum_sum, prog_bar=True, logger=True, on_step=True)

    def test_epoch_end(self, outputs) -> None:
        """
        Aggregate test metrics at the end of testing.
        
        Args:
            outputs: List of outputs from test_step
        """
        # Calculate mean metrics
        mean_loss = torch.stack(self.loss_list).mean()
        mean_acc1 = torch.stack(self.acc1_sum_list).sum() / torch.stack(self.pixelNum_sum_list).sum()
        mean_acc5 = torch.stack(self.acc5_sum_list).sum() / torch.stack(self.pixelNum_sum_list).sum()
        mean_ssim = torch.stack(self.ssim_sum_list).sum() / torch.stack(self.pixelNum_sum_list).sum()
        mean_psnr = torch.stack(self.psnr_sum_list).sum() / torch.stack(self.pixelNum_sum_list).sum()
        mean_mse = torch.stack(self.mse_sum_list).sum() / torch.stack(self.pixelNum_sum_list).sum()
        
        # Log aggregated metrics
        self.log('test/mean_loss', mean_loss, logger=True)
        self.log('test/mean_acc1', mean_acc1, logger=True)
        self.log('test/mean_acc5', mean_acc5, logger=True)
        self.log('test/mean_ssim', mean_ssim, logger=True)
        self.log('test/mean_psnr', mean_psnr, logger=True)
        self.log('test/mean_mse', mean_mse, logger=True)
        
        # Clear metric lists
        self.loss_list.clear()
        self.acc1_sum_list.clear()
        self.acc5_sum_list.clear()
        self.ssim_sum_list.clear()
        self.psnr_sum_list.clear()
        self.mse_sum_list.clear()
        self.pixelNum_sum_list.clear()

    def save_nii(self,
                save_dir: str,
                split: str,
                image_type: str,
                niis: np.ndarray,
                patient_id: int,
                time_point: int) -> None:
        """
        Save NIfTI file.
        
        Args:
            save_dir: Directory to save in
            split: Split name (train/val/test)
            image_type: Type of image
            niis: NIfTI data
            patient_id: Patient ID
            time_point: Time point
        """
        root = os.path.join(save_dir, "videos", split)
        print(root)
        filename = "{}_gs-_patient-{:04}_t-{:03}.nii.gz".format(
            image_type,
            patient_id,
            time_point
        )
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        nib.save(nib.Nifti1Image(niis, np.eye(4)), path)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers.
        
        Returns:
            Optimizer instance
        """
        optimizer = torch.optim.AdamW(
            chain(self.latentODE_model.parameters(), self.output_head.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.95)
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--unconditional', action='store_true')
        parser.add_argument('--base_lr', type=float, default=4.5e-06)
        parser.add_argument('--classification', action='store_true', default=False)
        parser.add_argument('--first_stage_vocab_size', type=int, default=16384)
        parser.add_argument('--first_stage_key', type=str, default='video', choices=['video'])
        parser.add_argument('--cond_stage_key', type=str, default='label', choices=['label', 'text', 'stft'])
        # latent ode hyperparameters
        parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
        parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
        parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
        parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
        parser.add_argument('--n_layers', type=int, default=3, help='A number of layer of vid ODE func')
        parser.add_argument('--n_downs', type=int, default=2)
        parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
        parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
        parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
        parser.add_argument('--batch_size_ode', type=int, default=64)
        parser.add_argument('--scale', type=int, default=4)
        
        return parser

