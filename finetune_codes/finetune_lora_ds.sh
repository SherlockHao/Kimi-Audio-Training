#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Memory optimization settings
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Limit CPU usage per process to prevent memory bloat
ulimit -v 262144000  # Limit virtual memory to ~250GB per process

DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="moonshotai/Kimi-Audio-7B" # Set the path if you do not want to load from huggingface directly

PRETRAINED_MODEL_PATH=""

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA=""

# LoRA hyperparameters - optimized for 800 hours of data
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1

function usage() {
    echo '
Usage: bash finetune/finetune_lora_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [-r LORA_RANK] [-a LORA_ALPHA]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model_path )
            shift
            PRETRAINED_MODEL_PATH=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -r | --lora_r )
            shift
            LORA_R=$1
            ;;
        -a | --lora_alpha )
            shift
            LORA_ALPHA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# check if data exists
if [ ! -f "$DATA" ]; then
    echo "Error: DATA file does not exist"
    exit 1
fi

# check if model_path exists
if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist"
    exit 1
fi

echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA"
echo "LoRA R: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "start LoRA finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

torchrun $DISTRIBUTED_ARGS finetune_lora.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --data_path $DATA \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir output/kimiaudio_lora_ckpts \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to "none" \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune_codes/ds_config_zero2.json \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --train_whisper True \
    --train_vq_adaptor True \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2