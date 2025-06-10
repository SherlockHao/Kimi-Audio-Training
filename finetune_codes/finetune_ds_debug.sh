#!/bin/bash
# Enhanced training script with comprehensive debugging setup

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Setup debugging environment variables
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,GRAPH,INIT,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Extended NCCL timeout for large model training (2 hours)
export NCCL_TIMEOUT=7200

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True

# Disable P2P for stability (optional - uncomment if experiencing issues)
# export NCCL_P2P_DISABLE=1

# Enable NCCL tree algorithm for better performance
export NCCL_TREE_THRESHOLD=0

# Set higher socket buffer sizes for better network performance
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4

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

# Create debug output directory
DEBUG_DIR="debug_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $DEBUG_DIR

echo "Debug logs will be saved to: $DEBUG_DIR"

function usage() {
    echo '
Usage: bash finetune/finetune_ds_debug.sh [-m MODEL_PATH] [-d DATA_PATH]
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

# Log system information
echo "=== System Information ===" | tee $DEBUG_DIR/system_info.log
echo "Hostname: $(hostname)" | tee -a $DEBUG_DIR/system_info.log
echo "GPU Information:" | tee -a $DEBUG_DIR/system_info.log
nvidia-smi | tee -a $DEBUG_DIR/system_info.log
echo "CPU Information:" | tee -a $DEBUG_DIR/system_info.log
lscpu | grep -E "Model name|Socket|Core|Thread" | tee -a $DEBUG_DIR/system_info.log
echo "Memory Information:" | tee -a $DEBUG_DIR/system_info.log
free -h | tee -a $DEBUG_DIR/system_info.log

# Log environment variables
echo "=== Environment Variables ===" | tee $DEBUG_DIR/env_vars.log
env | grep -E "NCCL|CUDA|TORCH|MASTER|RANK|WORLD" | sort | tee -a $DEBUG_DIR/env_vars.log

# Function to monitor system resources in background
monitor_resources() {
    local pid=$1
    local log_file=$2
    
    echo "Time,CPU%,Memory%,GPU0_Mem%,GPU0_Util%" > $log_file
    
    while kill -0 $pid 2>/dev/null; do
        timestamp=$(date +%Y-%m-%d_%H:%M:%S)
        cpu_usage=$(ps -p $pid -o %cpu | tail -1 | xargs)
        mem_usage=$(ps -p $pid -o %mem | tail -1 | xargs)
        
        # Get GPU stats for first GPU (you can extend this for all GPUs)
        gpu_stats=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 0 | tr ',' ' ')
        gpu_mem_used=$(echo $gpu_stats | awk '{print $1}')
        gpu_mem_total=$(echo $gpu_stats | awk '{print $2}')
        gpu_util=$(echo $gpu_stats | awk '{print $3}')
        gpu_mem_percent=$(echo "scale=2; $gpu_mem_used * 100 / $gpu_mem_total" | bc)
        
        echo "$timestamp,$cpu_usage,$mem_usage,$gpu_mem_percent,$gpu_util" >> $log_file
        
        sleep 10
    done
}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "start finetune with debugging enabled"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

# Create a wrapper script to capture all outputs
cat > $DEBUG_DIR/run_training.sh << EOF
#!/bin/bash
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --model_path $PRETRAINED_MODEL_PATH \
    --data_path $DATA \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir output/kimiaudio_ckpts \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to "none" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune_codes/ds_config_zero3.json
EOF

chmod +x $DEBUG_DIR/run_training.sh

# Run training with comprehensive logging
echo "Starting training with PID monitoring..."
$DEBUG_DIR/run_training.sh 2>&1 | tee $DEBUG_DIR/training_full.log &
TRAINING_PID=$!

# Start resource monitoring in background
monitor_resources $TRAINING_PID $DEBUG_DIR/resource_usage.csv &
MONITOR_PID=$!

echo "Training PID: $TRAINING_PID"
echo "Monitor PID: $MONITOR_PID"
echo "Logs are being saved to $DEBUG_DIR/"

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo "Training completed with exit code: $TRAINING_EXIT_CODE"

# Analyze logs for common issues
echo "=== Log Analysis ===" | tee $DEBUG_DIR/analysis.txt
echo "Checking for NCCL errors..." | tee -a $DEBUG_DIR/analysis.txt
grep -i "nccl" $DEBUG_DIR/training_full.log | grep -i "error\|fail\|timeout" | tee -a $DEBUG_DIR/analysis.txt

echo "Checking for OOM errors..." | tee -a $DEBUG_DIR/analysis.txt
grep -i "out of memory\|oom" $DEBUG_DIR/training_full.log | tee -a $DEBUG_DIR/analysis.txt

echo "Checking for SIGABRT..." | tee -a $DEBUG_DIR/analysis.txt
grep -i "sigabrt\|signal 6\|abort" $DEBUG_DIR/training_full.log | tee -a $DEBUG_DIR/analysis.txt

# Save final system state
echo "=== Final System State ===" | tee $DEBUG_DIR/final_state.log
nvidia-smi | tee -a $DEBUG_DIR/final_state.log
free -h | tee -a $DEBUG_DIR/final_state.log

exit $TRAINING_EXIT_CODE