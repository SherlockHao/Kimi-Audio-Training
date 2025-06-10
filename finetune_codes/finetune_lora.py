# LoRA fine-tuning code for Kimi-Audio model
# Based on the original finetune.py with LoRA adaptation

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional, List

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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
class LoRAArguments:
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, will use default settings."}
    )
    train_whisper: bool = field(
        default=True,
        metadata={"help": "Whether to train whisper encoder (following original finetune.py behavior)"}
    )
    train_vq_adaptor: bool = field(
        default=True,
        metadata={"help": "Whether to train VQ adaptor (recommended as it's small)"}
    )

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
    # Save the base model (with LoRA weights merged)
    if trainer.args.should_save and trainer.args.local_rank == 0:
        # Save LoRA weights
        trainer.model.save_pretrained(output_dir)
        
        # Also save the whisper model and vq_adaptor if they were trained
        if hasattr(trainer.model, 'whisper_model'):
            whisper_state_dict = trainer.model.whisper_model.state_dict()
            torch.save(whisper_state_dict, os.path.join(output_dir, "whisper_model.pt"))
        
        if hasattr(trainer.model, 'vq_adaptor'):
            # The vq_adaptor is inside the base_model
            vq_adaptor_state_dict = {}
            for name, param in trainer.model.named_parameters():
                if 'vq_adaptor' in name and param.requires_grad:
                    # Remove 'base_model.model.' prefix if exists
                    clean_name = name.replace('base_model.model.', '')
                    vq_adaptor_state_dict[clean_name] = param.data.cpu()
            if vq_adaptor_state_dict:
                torch.save(vq_adaptor_state_dict, os.path.join(output_dir, "vq_adaptor.pt"))


def freeze_base_model_except_adaptor(model):
    """Freeze base model except VQ adaptor before applying LoRA"""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze VQ adaptor
    for name, param in model.named_parameters():
        if 'vq_adaptor' in name:
            param.requires_grad = True
            logger.info(f"Unfreezing {name}")


def setup_lora_model(model, lora_args):
    """Setup LoRA for the model"""
    # Prepare model for training (handles any necessary preprocessing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Default target modules if not specified
    # Target both main layers and mimo layers attention modules
    if lora_args.lora_target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"      # MLP layers - added for better performance with 800h data
        ]
    else:
        target_modules = lora_args.lora_target_modules
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # This will apply LoRA to both self.layers and self.mimo_layers
        modules_to_save=["embed_tokens", "lm_head", "mimo_output"] if not lora_args.train_vq_adaptor else ["embed_tokens", "lm_head", "mimo_output", "vq_adaptor"],
    )
    
    # First freeze base model except VQ adaptor
    if lora_args.train_vq_adaptor:
        freeze_base_model_except_adaptor(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model


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

    train_dataset = dataset_cls(train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)

    if eval_data:
        eval_dataset = dataset_cls(eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def compute_loss(outputs, labels, num_items_in_batch=None):
    audio_logits, text_logits = outputs.logits

    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    loss = audio_loss + text_loss
    return loss


class KimiAudioLoRAModel(KimiAudioModel):
    """Wrapper to handle whisper model training separately from LoRA"""
    
    def __init__(self, config, train_whisper=True):
        super().__init__(config)
        self.train_whisper = train_whisper
        
        # Control whisper model gradient
        if not train_whisper:
            for param in self.whisper_model.parameters():
                param.requires_grad = False
        else:
            for param in self.whisper_model.parameters():
                param.requires_grad = True


def train():
    global local_rank
    
    # Set thread limits to prevent CPU oversubscription
    import os
    torch.set_num_threads(16)
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

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
    
    # Load model with whisper training control
    base_model = KimiAudioLoRAModel.init_from_pretrained(
        model_args.model_path, 
        model_load_kwargs,
    )
    base_model.train_whisper = lora_args.train_whisper
    
    # Control whisper gradient
    for param in base_model.whisper_model.parameters():
        param.requires_grad = lora_args.train_whisper
    
    # Apply LoRA to the main model (not including whisper)
    model = setup_lora_model(base_model, lora_args)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model, text_tokenizer=text_tokenizer,
        data_args=data_args, max_len=training_args.model_max_length, kimia_token_offset=model.config.kimia_token_offset
    )

    # Start trainer
    trainer = Trainer(
        model=model, args=training_args, 
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # Save training args and LoRA config for reference
    lora_config_dict = {
        "lora_r": lora_args.lora_r,
        "lora_alpha": lora_args.lora_alpha,
        "lora_dropout": lora_args.lora_dropout,
        "lora_target_modules": lora_args.lora_target_modules,
        "train_whisper": lora_args.train_whisper,
        "train_vq_adaptor": lora_args.train_vq_adaptor,
    }
    
    if trainer.args.local_rank == 0:
        with open(os.path.join(training_args.output_dir, "lora_config.json"), "w") as f:
            json.dump(lora_config_dict, f, indent=2)

if __name__ == "__main__":
    train()