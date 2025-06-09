#!/bin/bash

# Training configuration
python train_vqgan.py \
    --random_seed 1234 \
    --gpus 1 \
    --embedding_dim 16 \
    --n_codes 64 \
    --n_hiddens 32 \
    --downsample 4,4,4 \
    --batch_size 1 \
    --accumulate_grad_batches 6 \
    --progress_bar_refresh_rate 500 \
    --max_steps 50000 \
    --gradient_clip_val 1.0 \
    --lr 3e-4 \
    --resolution 256 \
    --sequence_length 96 \
    --discriminator_iter_start 0 \
    --norm_type batch \
    --perceptual_weight 4 \
    --image_gan_weight 1 \
    --video_gan_weight 1 \
    --gan_feat_weight 4 \
    --image_channels 1 \
    --default_root_dir ./experiment \
    --num_workers 0 \
    --data_root "./Leuven/" \
    --img_path "Leuven_IPF_registered" \
    --mask_path "Leuven_IPF_registered_mask" \
    --label_path "Leuven_data_label.csv" \
    --external_data_root "./SouthamptonExternalData/" \
    --external_img_path "CTscans" \
    --external_mask_path "Lungmasks" \
    --external_label_path "MortalityDataSouthampton.csv" \
    --CT_scans True 