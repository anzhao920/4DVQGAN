# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGAN, VideoData
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning import loggers as pl_loggers
import numpy as np
def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()
    args.CT_scans = True
    args.gpus = 1
    args.embedding_dim = 16
    args.n_codes = 64
    args.n_hiddens = 32
    args.downsample = (4,4,4)
    args.batch_size = 1
    args.accumulate_grad_batches = 6
    args.progress_bar_refresh_rate= 500 
    args.max_steps=50000
    args.gradient_clip_val=1.0
    args.lr = 3e-4
    args.resolution = 256
    args.sequence_length=96
    args.discriminator_iter_start=0
    args.norm_type ='batch'
    args.perceptual_weight = 4
    args.image_gan_weight = 1
    args.video_gan_weight = 1
    args.gan_feat_weight = 4
    args.image_channels = 1
    args.default_root_dir='/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/experiment15/'
    args.num_workers=0
    args.data_root = '/cluster/project7/IPFPrognosisPredictionNew/'
    args.img_path = 'Leuven_IPF_registered'
    args.mask_path ='Leuven_IPF_registered_mask'
    args.label_path = 'Leuven_data_label.csv'
    args.external_data_root = '/cluster/project9/IPFPrognosisPrediction/allData/SouthamptonExternalData/'
    args.external_img_path = 'CTscans' 
    args.external_mask_path = 'Lungmasks'
    args.external_label_path = 'MortalityDataSouthampton.csv'
    dataset_list = ['training data','external validation data','internal validation data']
    model_path_list= ["/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/experiment15/lightning_logs/version_0/checkpoints/epoch=195-step=17999-train/recon_loss=0.11.ckpt",
                      "/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/experiment16/lightning_logs/version_0/checkpoints/epoch=195-step=17999-train/recon_loss=0.13.ckpt",
                      "/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/experiment18/lightning_logs/version_0/checkpoints/epoch=195-step=17999-train/recon_loss=0.10.ckpt"
                      ]
    for dataset in dataset_list:
        if dataset == 'training data':
            args.predict_trainingdata=True
            args.external_test = False
        elif dataset == 'external validation data':
            args.predict_trainingdata=False
            args.external_test = True
        else:
            args.predict_trainingdata=False
            args.external_test = False            
        data = VideoData(args)
        args.arch = '3D-VQGAN'
        for model_path in model_path_list:           
            kwargs = dict()
            if args.gpus > 1:
                kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
            trainer = pl.Trainer.from_argparse_args(args,
                                                    precision=16,**kwargs)
            model = VQGAN.load_from_checkpoint(model_path)
            model.eval()
            trainer.predict(model, data)
            mse_loss = np.stack(model.mse_loss_list).mean()
            print('model_path: ',model_path)
            print('dataset: ',dataset)
            print('mse_loss:',mse_loss)


if __name__ == '__main__':
    main()

