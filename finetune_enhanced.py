# Enhanced finetune.py with comprehensive debugging strategies
# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional
import signal
import sys
import time
import traceback
import psutil
import gc

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)s] - %(message)s',
    handlers=[
        logging.FileHandler(f'training_debug_rank_{os.environ.get("LOCAL_RANK", 0)}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add rank information to all log messages
class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = os.environ.get("LOCAL_RANK", 0)
        return True

logger.addFilter(RankFilter())

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# Global variables for debugging
MEMORY_LOG_INTERVAL = 100  # Log memory every N steps
CHECKPOINT_DEBUG_INTERVAL = 500  # Save debug checkpoint every N steps
DEBUG_MODE = True  # Enable comprehensive debugging

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B-Instruct")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        logger.error(f"Received signal {signum}")
        logger.error(f"Stack trace:\n{traceback.format_stack()}")
        
        # Log final memory state
        log_memory_usage("SIGNAL_HANDLER")
        
        # Cleanup distributed training
        if torch.distributed.is_initialized():
            logger.info("Destroying process group...")
            torch.distributed.destroy_process_group()
        
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def log_memory_usage(phase=""):
    """Comprehensive memory logging"""
    if not DEBUG_MODE:
        return
        
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # GPU Memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            
            logger.info(f"[{phase}] GPU {i} Memory: Allocated={allocated:.2f}GB, "
                       f"Reserved={reserved:.2f}GB, MaxAllocated={max_allocated:.2f}GB")
            
            # Log memory summary for rank 0
            if rank == 0 and i == 0:
                logger.debug(f"GPU {i} Memory Summary:\n{torch.cuda.memory_summary(i)}")
    
    # CPU Memory
    process = psutil.Process()
    cpu_mem = process.memory_info()
    logger.info(f"[{phase}] CPU Memory: RSS={cpu_mem.rss/1024**3:.2f}GB, "
               f"VMS={cpu_mem.vms/1024**3:.2f}GB")
    
    # System Memory
    sys_mem = psutil.virtual_memory()
    logger.info(f"[{phase}] System Memory: Total={sys_mem.total/1024**3:.2f}GB, "
               f"Available={sys_mem.available/1024**3:.2f}GB, "
               f"Used={sys_mem.percent}%")

def check_distributed_sync():
    """Check if all processes are synchronized"""
    if not torch.distributed.is_initialized():
        return
        
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Create a tensor with rank value
    sync_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    sync_list = [torch.zeros(1, dtype=torch.float32).cuda() for _ in range(world_size)]
    
    try:
        # All-gather to check all ranks are alive
        torch.distributed.all_gather(sync_list, sync_tensor)
        
        # Verify all ranks reported
        reported_ranks = [int(t.item()) for t in sync_list]
        expected_ranks = list(range(world_size))
        
        if sorted(reported_ranks) != expected_ranks:
            logger.error(f"Rank {rank}: Sync check failed! Expected {expected_ranks}, got {reported_ranks}")
            return False
            
        logger.debug(f"Rank {rank}: Sync check passed")
        return True
        
    except Exception as e:
        logger.error(f"Rank {rank}: Sync check failed with error: {e}")
        return False

class DebugTrainer(Trainer):
    """Enhanced Trainer with debugging capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_times = []
        self.last_step_time = time.time()
        self.memory_logs = []
        
    def training_step(self, model, inputs):
        """Override training step to add debugging"""
        step_start = time.time()
        
        # Log memory before step
        if self.state.global_step % MEMORY_LOG_INTERVAL == 0:
            log_memory_usage(f"BEFORE_STEP_{self.state.global_step}")
        
        # Check distributed sync periodically
        if self.state.global_step % 100 == 0 and self.state.global_step > 0:
            if not check_distributed_sync():
                logger.error(f"Distributed sync check failed at step {self.state.global_step}")
        
        try:
            # Original training step
            loss = super().training_step(model, inputs)
            
            # Log step time
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            
            # Detect anomalies
            if len(self.step_times) > 10:
                avg_time = sum(self.step_times[-10:]) / 10
                if step_time > avg_time * 2:
                    logger.warning(f"Step {self.state.global_step} took {step_time:.2f}s "
                                 f"(avg: {avg_time:.2f}s) - potential bottleneck")
            
            # Log memory after step
            if self.state.global_step % MEMORY_LOG_INTERVAL == 0:
                log_memory_usage(f"AFTER_STEP_{self.state.global_step}")
                
                # Force garbage collection periodically
                if self.state.global_step % (MEMORY_LOG_INTERVAL * 5) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"Forced garbage collection at step {self.state.global_step}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training step {self.state.global_step}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            log_memory_usage(f"ERROR_STEP_{self.state.global_step}")
            raise
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override checkpoint saving with error handling"""
        try:
            logger.info(f"Saving checkpoint at step {self.state.global_step}")
            log_memory_usage(f"BEFORE_CHECKPOINT_{self.state.global_step}")
            
            # Save debug info
            debug_info = {
                "step": self.state.global_step,
                "memory_logs": self.memory_logs[-100:],  # Last 100 memory logs
                "step_times": self.step_times[-100:],    # Last 100 step times
                "timestamp": time.time()
            }
            
            checkpoint_folder = super()._get_output_dir(trial=trial)
            os.makedirs(checkpoint_folder, exist_ok=True)
            
            with open(os.path.join(checkpoint_folder, "debug_info.json"), "w") as f:
                json.dump(debug_info, f, indent=2)
            
            # Original checkpoint saving
            super()._save_checkpoint(model, trial, metrics)
            
            log_memory_usage(f"AFTER_CHECKPOINT_{self.state.global_step}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    try:
        logger.info(f"Starting safe model save to {output_dir}")
        log_memory_usage("BEFORE_FINAL_SAVE")
        
        # check if zero3 mode enabled
        if deepspeed.is_deepspeed_zero3_enabled():
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = trainer.model.state_dict()
            
        if trainer.args.should_save and trainer.args.local_rank == 0:
            trainer._save(output_dir, state_dict=state_dict)
            
        log_memory_usage("AFTER_FINAL_SAVE")
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_data = all_data[:int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio):]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    # Use numpy arrays instead of Python lists to avoid memory leaks
    train_dataset = dataset_cls(train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, 
                               max_len=max_len, kimia_token_offset=kimia_token_offset)

    if eval_data:
        eval_dataset = dataset_cls(eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, 
                                  max_len=max_len, kimia_token_offset=kimia_token_offset)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def compute_loss(outputs, labels, num_items_in_batch=None):
    audio_logits, text_logits = outputs.logits
    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), 
                                                   audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), 
                                                  text_labels.view(-1), reduction="none")

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    
    loss = audio_loss + text_loss
    
    # Check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"Loss is NaN/Inf! Audio loss: {audio_loss}, Text loss: {text_loss}")
        raise ValueError("Loss is NaN or Inf")
    
    return loss

def train():
    global local_rank
    
    # Setup signal handlers first
    setup_signal_handlers()
    
    # Log environment information
    logger.info("=== Environment Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"NCCL version: {torch.cuda.nccl.version()}")
    logger.info(f"DeepSpeed version: {deepspeed.__version__}")
    
    # Set environment variables for debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "COLL,GRAPH"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_TIMEOUT"] = "7200"  # 2 hours
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Log all environment variables
    logger.info("=== NCCL Environment Variables ===")
    for key, value in os.environ.items():
        if "NCCL" in key or "CUDA" in key or "TORCH" in key:
            logger.info(f"{key}={value}")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank
    
    # Log initial memory state
    log_memory_usage("INITIAL")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info(f"Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        # local path
        cache_path = model_args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    # check if model_path exists
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")
    
    try:
        log_memory_usage("BEFORE_MODEL_LOAD")
        model = KimiAudioModel.from_pretrained(model_args.model_path, 
                                             device_map=None,
                                             **model_load_kwargs)
        log_memory_usage("AFTER_MODEL_LOAD")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    log_memory_usage("BEFORE_DATA_LOAD")
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model, text_tokenizer=text_tokenizer,
        data_args=data_args, max_len=training_args.model_max_length, 
        kimia_token_offset=model.config.kimia_token_offset
    )
    log_memory_usage("AFTER_DATA_LOAD")

    # Use enhanced trainer
    trainer = DebugTrainer(
        model=model, 
        args=training_args, 
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    # Start training with exception handling
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        log_memory_usage("TRAINING_ERROR")
        
        # Try to save emergency checkpoint
        try:
            emergency_dir = os.path.join(training_args.output_dir, "emergency_checkpoint")
            logger.info(f"Attempting to save emergency checkpoint to {emergency_dir}")
            trainer.save_model(emergency_dir)
            trainer.save_state()
        except:
            logger.error("Failed to save emergency checkpoint")
        
        raise
    
    finally:
        # Always try to save final state
        try:
            trainer.save_state()
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Final memory log
        log_memory_usage("FINAL")
        
        # Cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train()