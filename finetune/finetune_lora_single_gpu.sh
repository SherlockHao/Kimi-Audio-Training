#!/bin/bash

MODEL="moonshotai/Kimi-Audio-7B-Instruct"
DATA="path/to/your/training/data.json"

python finetune.py \
  --model_name_or_path $MODEL \
  --model_path $MODEL \
  --data_path $DATA \
  --bf16 True \
  --output_dir output_kimia_lora \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "c_attn" "c_proj" "w1" "w2" \
  --lora_bias "none" 