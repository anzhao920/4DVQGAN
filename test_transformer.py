# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,GPUStatsMonitor
from tats import Net2NetTransformer, VideoData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()
    
    # trainer args
    args.gpus = 1
    args.batch_size = 1
    # args.accumulate_grad_batches = 6
    # args.progress_bar_refresh_rate= 500 
    args.max_steps=50000
    args.gradient_clip_val=1.0
    # args.log_every_n_steps=5

    # Net2NetTransformer args
    args.vqvae = "/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/experiment6/lightning_logs/version_0/checkpoints/latest_checkpoint-v1.ckpt"
    args.unconditional = True
    args.longitudinal_CT_scans = True
    args.default_root_dir='/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/transformer_experiment_1'
    args.base_lr = 4.5e-05
    args.vocab_size = 256
    args.first_stage_vocab_size=256
    args.block_size = 1537
    args.n_layer=12
    args.n_head=12
    args.n_embd=768
    args.first_stage_key = 'longitudianl_CT_scans'
    args.batch_size_ode = 1536
    args.scale = 4
    args.timepoints = 40
    # vqgan parameters
    args.embedding_dim=16
    # args.sample_every_n_latent_frames=8

    #latentODE args
    args.latents = 512
    args.gen_layers = 8
    args.units = 256
    args.rec_dims = 512
    args.rec_layers = 8
    args.gru_units = 256
    # data args
    args.resolution = 256
    args.sequence_length=96
    args.num_workers=1
    args.data_root = '/cluster/project7/IPFPrognosisPredictionNew/'
    args.img_path = 'Leuven_IPF_registered'
    args.mask_path ='Leuven_IPF_registered_mask'
    args.label_path = 'Leuven_data_label.csv'
    # args.external_data_root = 'C:/Users/An/OneDrive - University College London/SouthamptonExternalData/'
    # args.external_img_path = 'CTscans' 
    # args.external_mask_path = 'Lungmasks'
    # args.external_label_path = 'MortalityDataSouthampton.csv'
    args.external_data_root = '/cluster/project7/IPFPrognosisPredictionNew/'
    args.external_img_path = 'Leuven_IPF_registered'
    args.external_mask_path = 'Leuven_IPF_registered_mask'
    args.external_label_path = 'Leuven_data_label.csv'
    args.max_longitudinal_CT=3
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
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=3, filename='best_checkpoint'))

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

    # # load the most recent checkpoint file
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
    #                 ckpt_file = 'latest_checkpoint_prev.ckpt'
    #                 os.rename(os.path.join(ckpt_folder, fn), os.path.join(ckpt_folder, ckpt_file))
    #         if len(ckpt_file) > 0:
    #             args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
    #             print('will start from the recent ckpt %s'%args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps,**kwargs)

    # trainer.fit(model, data)
    model = Net2NetTransformer.load_from_checkpoint("/cluster/project7/IPFMortalityPredictionNewloss/TATS-main/transformer_experiment_1/lightning_logs/version_4/checkpoints/epoch=285-step=49999-train/loss=0.01.ckpt",args=args)
    model.eval()
    predictions = trainer.predict(model, data)    


if __name__ == '__main__':
    main()

