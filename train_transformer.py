# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,GPUStatsMonitor
from tats import Net2NetTransformer, VideoData
import numpy as np

def main():
    

    
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()
    args.random_seed = 1234
    pl.seed_everything(args.random_seed)
    # trainer args
    args.gpus = 1
    args.batch_size = 1
    args.accumulate_grad_batches = 6
    # args.progress_bar_refresh_rate= 500 
    args.max_steps=50000

    args.log_every_n_steps=5

    # Net2NetTransformer args
    # args.vqvae = "./experiment6/lightning_logs/version_0/checkpoints/latest_checkpoint-v1.ckpt"
    args.vqvae = "./experiment15/lightning_logs/version_0/checkpoints/epoch=195-step=17999-train/recon_loss=0.11.ckpt"
    args.unconditional = True
    args.longitudinal_CT_scans = True
    args.default_root_dir='./transformer_experiment_1'
    args.base_lr = 2e-04
    args.vocab_size = 256
    args.first_stage_vocab_size=256
    args.block_size = 1537
    
    # transformer parameters
    args.n_layer=3
    args.n_head=8
    args.n_embd=1024
    args.first_stage_key = 'longitudianl_CT_scans'
    args.batch_size_ode = 1536
    args.scale = 1
    # vqgan parameters
    args.embedding_dim=16
    # args.sample_every_n_latent_frames=8
    args.optimizer = 'Adam'
    if args.optimizer != 'SAM':
        args.gradient_clip_val=1.0

    #latentODE args
    args.latents = 100
    args.gen_layers = 3 #odernn n layers of ode encoder
    args.units = 64
    args.rec_dims = 100
    args.rec_layers = 3 #odernn n layers of ode decoder
    args.gru_units = 64
    # args.n_layers = 6 #vidode n layers of ode encoder and decoder
    args.n_layers = 3
    args.n_downs = 0
    # data args
    args.resolution = 256
    args.sequence_length=96
    args.num_workers=0
    args.data_root = 'C:/My Data/Leuven/'
    args.img_path = 'Leuven_IPF_registered'
    args.mask_path ='Leuven_IPF_registered_mask'
    args.label_path = 'Leuven_data_label.csv'
    # args.external_data_root = 'C:/Users/An/OneDrive - University College London/SouthamptonExternalData/'
    # args.external_img_path = 'CTscans' 
    # args.external_mask_path = 'Lungmasks'
    # args.external_label_path = 'MortalityDataSouthampton.csv'
    args.external_data_root = 'C:/My Data/Leuven/'
    args.external_img_path = 'Leuven_IPF_registered'
    args.external_mask_path = 'Leuven_IPF_registered_mask'
    args.external_label_path = 'Leuven_data_label.csv'
    args.max_longitudinal_CT=10
    args.mode = 'generation'
    args.classification = False
    args.residual = True
    args.ode_rnn = False
    args.time_window_max = 365.25*6
    args.timepoints = round(args.time_window_max/182.5)+1
    args.downsample_latent = False
    args.flowmap = True
    args.run_backwards = True
    # args.ode_n_unit = 128
    # args.adjoint = False
    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if not args.unconditional and args.cond_stage_key=='label' else None
    model = Net2NetTransformer(args, first_stage_key=args.first_stage_key, cond_stage_key=args.cond_stage_key)

    callbacks = []
    callbacks.append(GPUStatsMonitor() )
    callbacks.append(ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=3, filename='best_checkpoint_val_loss'))
    callbacks.append(ModelCheckpoint(monitor='train/loss', mode='min', save_top_k=3, filename='best_checkpoint_train_loss'))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(gpus=args.gpus,
                      # plugins=["deepspeed_stage_2"])
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])

    # configure learning rate
    bs, base_lr = args.batch_size, args.base_lr
    ngpu = args.gpus
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

    # load the most recent checkpoint file
    # base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    # if os.path.exists(base_dir):
    #     log_folder = ckpt_file = ''
    #     version_id_used = step_used = -1
    #     for folder in os.listdir(base_dir):
    #         version_id = int(folder.split('_')[1])
    #         if version_id > version_id_used:
    #             version_id_used = version_id
    #             log_folder = folder
    #     if len(log_folder) > 0:
    #         ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
    #         for fn in os.listdir(ckpt_folder):
    #             if fn == 'latest_checkpoint.ckpt':
    #                 ckpt_file = 'latest_ch
    # eckpoint_prev.ckpt'
    #                 os.rename(os.path.join(ckpt_folder, fn), os.path.join(ckpt_folder, ckpt_file))
    #         if len(ckpt_file) > 0:
    #             args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
    #             print('will start from the recent ckpt %s'%args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps,**kwargs)
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
    #                                         max_steps=args.max_steps,**kwargs,resume_from_checkpoint='./transformer_experiment_1/lightning_logs/loss=2.58.ckpt')

    print(trainer.logger.log_dir)
    # trainer.fit(model, data)
    
    model_path = "./transformer_experiment_1/lightning_logs/loss=2.58.ckpt"
    model = Net2NetTransformer.load_from_checkpoint(model_path,args=args)
    model.eval()
    trainer.test(model, data)
    # trainer.predict(model, data)
    pixelNum_sum = np.stack(model.pixelNum_sum_list).sum()
    print('model_path: ',model_path)
    print('test mode: ',args.mode)
    print('acc1_mean: ',np.stack(model.acc1_sum_list).sum()/pixelNum_sum)
    print('acc5_mean: ',np.stack(model.acc5_sum_list).sum()/pixelNum_sum)
    print('ssim_mean: ',np.stack(model.ssim_sum_list).sum()/pixelNum_sum)
    print('psnr_mean: ',np.stack(model.psnr_sum_list).sum()/pixelNum_sum)
    print('mse_mean: ',np.stack(model.mse_sum_list).sum()/pixelNum_sum)
    

if __name__ == '__main__':
    main()

