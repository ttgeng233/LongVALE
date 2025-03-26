#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29580


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT longvalellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/longvale-sft-bp-154k.json \
    --feat_folder /path/to/bp_visual_feat \
    --audio_feat_folder /path/to/bp_audio_feat \
    --asr_feat_folder /path/to/bp_speech_feat \
    --pretrain_mm_mlp_adapter ./checkpoint/vtimellm_stage1_mm_projector.bin \
    --output_dir ./checkpoints/longvale-$MODEL_VERSION-stage2-bp \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
