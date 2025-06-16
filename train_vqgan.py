"""
Training script for 3D-VQGAN model on IPF (Idiopathic Pulmonary Fibrosis) CT scans.
This script handles the training process using PyTorch Lightning.
"""

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.vqgan_3d import VQGAN
from model.data import IPFData
from model.modules.callbacks import ImageLogger, VideoLogger
from pytorch_lightning import loggers as pl_loggers

def setup_callbacks(args):
    """
    Setup training callbacks for model checkpointing and logging.
    
    Args:
        args: Training arguments
        
    Returns:
        list: List of configured callbacks
    """
    callbacks = [
        # Save best model based on reconstruction loss
        ModelCheckpoint(
            monitor='train/recon_loss',
            save_top_k=3,
            mode='min',
            filename='latest_checkpoint'
        ),
        # Save checkpoint every 3000 steps
        ModelCheckpoint(
            every_n_train_steps=3000,
            save_top_k=-1,
            filename='{epoch}-{step}-{train/recon_loss:.2f}'
        ),
        # Log images during training
        ImageLogger(
            batch_frequency=750,
            max_images=4,
            clamp=True
        ),
        # Log videos during training
        VideoLogger(
            batch_frequency=1500,
            max_videos=4,
            clamp=True
        )
    ]
    return callbacks

def adjust_learning_rate(args):
    """
    Adjust learning rate based on batch size, number of GPUs, and gradient accumulation.
    
    Args:
        args: Training arguments
    """
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print(f"Setting learning rate to {args.lr:.2e} = {accumulate} (accumulate_grad_batches) * "
          f"{ngpu/8} (num_gpus/8) * {bs/4} (batchsize/4) * {base_lr:.2e} (base_lr)")

def main():
    """Main training function."""


    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = IPFData.add_data_specific_args(parser)
    args = parser.parse_args()
    # Set random seed for reproducibility
    pl.seed_everything(args.random_seed)
    
    # Initialize data module
    data = IPFData(args)

    # Pre-load dataloaders
    data.train_dataloader()
    data.test_dataloader()

    # Adjust learning rate based on training configuration
    adjust_learning_rate(args)

    # Initialize model
    model = VQGAN(args)

    # Setup callbacks
    callbacks = setup_callbacks(args)

    # Configure distributed training if using multiple GPUs
    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    
    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        precision=16,
        **kwargs
    )

    # Start training
    trainer.fit(model, data)

    # test the trained model
    # model = VQGAN.load_from_checkpoint("./experiment6/lightning_logs/version_0/checkpoints/latest_checkpoint-v1.ckpt")
    # model.eval()
    # predictions = trainer.validate(model, data)

if __name__ == '__main__':
    main()

