# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGAN, VideoData
from tats.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning import loggers as pl_loggers
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
    args.default_root_dir='./experiment6'
    args.num_workers=0
    args.data_root = 'C:/My Data/Leuven/'
    args.img_path = 'Leuven_IPF_registered'
    args.mask_path ='Leuven_IPF_registered_mask'
    args.label_path = 'Leuven_data_label.csv'
    args.external_data_root = 'C:/Users/An/OneDrive - University College London/SouthamptonExternalData/'
    args.external_img_path = 'CTscans' 
    args.external_mask_path = 'Lungmasks'
    args.external_label_path = 'MortalityDataSouthampton.csv'
    data = VideoData(args)
    args.arch = '3D-VQGAN'
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='train/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir+"lightning_logs/",name = args.arch,version=f"fold_{0}")
    # # load the most recent checkpoint file
    # base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    # if os.path.exists(base_dir):
    #     log_folder = ckpt_file = ''
    #     version_id_used = step_used = 0
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
                                            precision=16,**kwargs)

    trainer.fit(model, data)
    # model = VQGAN.load_from_checkpoint("./experiment6/lightning_logs/version_0/checkpoints/latest_checkpoint-v1.ckpt")
    # model.eval()
    # predictions = trainer.predict(model, data)


if __name__ == '__main__':
    main()

