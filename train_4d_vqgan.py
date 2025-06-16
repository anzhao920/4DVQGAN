"""
Training script for 4D-VQGAN model on IPF (Idiopathic Pulmonary Fibrosis) CT scans.
This script handles the training process using PyTorch Lightning for the 4D VQGAN model,
which includes transformer-based temporal modeling.
"""

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from model import VQGAN_4D, IPFData
import numpy as np

def setup_callbacks(args):
    """
    Setup training callbacks for model checkpointing and monitoring.
    
    Args:
        args: Training arguments
        
    Returns:
        list: List of configured callbacks
    """
    callbacks = [
        # Monitor GPU usage
        GPUStatsMonitor(),
        
        # Save checkpoints every 1000 steps
        ModelCheckpoint(
            every_n_train_steps=1000,
            save_top_k=-1,
            filename='{epoch}-{step}-{train/loss:.2f}'
        ),
        
        # Save checkpoints every 5000 steps
        ModelCheckpoint(
            every_n_train_steps=5000,
            save_top_k=-1,
            filename='{epoch}-{step}-{train/loss:.2f}'
        ),
        
        # Save best model based on validation loss
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            filename='best_checkpoint_val_loss'
        ),
        
        # Save best model based on training loss
        ModelCheckpoint(
            monitor='train/loss',
            mode='min',
            save_top_k=3,
            filename='best_checkpoint_train_loss'
        )
    ]
    return callbacks

def adjust_learning_rate(model, args):
    """
    Adjust learning rate based on batch size, number of GPUs, and gradient accumulation.
    
    Args:
        model: The model to adjust learning rate for
        args: Training arguments
    """
    bs, base_lr = args.batch_size, args.base_lr
    ngpu = args.gpus
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    
    print(f"Setting learning rate to {model.learning_rate:.2e} = "
          f"{accumulate_grad_batches} (accumulate_grad_batches) * "
          f"{ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)")

def main():
    """Main training function."""
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN_4D.add_model_specific_args(parser)
    parser = IPFData.add_data_specific_args(parser)
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.random_seed)
    
    # Initialize data module
    data = IPFData(args)
    
    # Pre-load dataloaders
    data.train_dataloader()
    data.test_dataloader()
    
    # Setup conditional dimensions
    args.class_cond_dim = (data.n_classes if not args.unconditional 
                          and args.cond_stage_key == 'label' else None)
    
    # Initialize model
    model = VQGAN_4D(args, 
                     first_stage_key=args.first_stage_key,
                     cond_stage_key=args.cond_stage_key)
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    
    # Configure distributed training if using multiple GPUs
    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(
            accelerator='gpu',
            devices=args.gpus,
            strategy='ddp'
        )
    
    # Adjust learning rate
    adjust_learning_rate(model, args)
    
    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        max_steps=args.max_steps,
        **kwargs
    )
    
    print(f"Logging directory: {trainer.logger.log_dir}")
    
    # Start training
    trainer.fit(model, data)

if __name__ == '__main__':
    main() 