# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional, List

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer, AutoConfig
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import GPTQConfig
import torch
import torch.nn.functional as F

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.lora_config import (
    LoraArguments, 
    get_optimized_lora_config, 
    merge_and_save, 
    print_lora_info, 
    save_lora_only,
    get_peft_state_maybe_zero_3,
    save_lora_weights
)

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
    use_lora: bool = field(default=False)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def safe_save_model_for_hf_trainer2(
    trainer: transformers.Trainer, 
    output_dir: str, 
    bias: str = "none",
    save_lora_only: bool = False,
    merge_lora: bool = False,
    lora_args: Optional[LoraArguments] = None
):
    """
    统一的模型保存函数，支持普通模型和LoRA模型的保存
    
    Args:
        trainer: Hugging Face Trainer对象
        output_dir: 输出目录
        bias: LoRA bias配置
        save_lora_only: 是否只保存LoRA权重
        merge_lora: 是否合并LoRA权重到基础模型
        lora_args: LoRA配置参数
    """
    
    # 判断是否使用了LoRA
    is_lora_model = hasattr(trainer.model, 'peft_config') or (
        hasattr(trainer.model, 'base_model') and hasattr(trainer.model.base_model, 'peft_config')
    )
    
    # DeepSpeed Zero3 特殊处理
    if deepspeed.is_deepspeed_zero3_enabled():
        if is_lora_model and not merge_lora:
            # Zero3 + LoRA: 需要特殊处理
            rank0_print("Saving LoRA weights under DeepSpeed Zero3...")
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            # Zero3 + 普通模型或需要合并的LoRA
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if is_lora_model:
            if save_lora_only and not merge_lora:
                # 只获取LoRA权重
                state_dict = get_peft_state_maybe_zero_3(
                    trainer.model.named_parameters(), bias
                )
            elif merge_lora:
                # 合并LoRA权重到基础模型
                rank0_print("Merging LoRA weights to base model...")
                merged_model = trainer.model.merge_and_unload()
                state_dict = merged_model.state_dict()
            else:
                # 获取完整的PEFT模型state_dict
                state_dict = trainer.model.state_dict()
        else:
            # 普通模型
            state_dict = trainer.model.state_dict()
    
    # 保存模型
    if trainer.args.should_save and trainer.args.local_rank == 0:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        if is_lora_model:
            if save_lora_only and not merge_lora:
                # 只保存LoRA权重
                rank0_print(f"Saving LoRA weights to {output_dir}")
                save_lora_weights(trainer.model, state_dict, output_dir, lora_args)
            elif merge_lora:
                # 保存合并后的完整模型
                rank0_print(f"Saving merged model to {output_dir}")
                trainer._save(output_dir, state_dict=state_dict)
                
                # 同时保存tokenizer和配置
                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(output_dir)
                
                # 保存模型配置
                if hasattr(trainer.model, 'config'):
                    trainer.model.config.save_pretrained(output_dir)
                elif hasattr(trainer.model, 'base_model') and hasattr(trainer.model.base_model, 'config'):
                    trainer.model.base_model.config.save_pretrained(output_dir)
            else:
                # 保存完整的PEFT模型（包含adapter配置）
                rank0_print(f"Saving PEFT model to {output_dir}")
                trainer.model.save_pretrained(output_dir)
                
                # 保存tokenizer
                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(output_dir)
        else:
            # 普通模型保存
            rank0_print(f"Saving model to {output_dir}")
            trainer._save(output_dir, state_dict=state_dict)

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

def compute_unlikelihood_penalty(logits, labels, mask, window_size=5, penalty_weight=0.1):
    """
    计算可微分的n-gram重复惩罚 (基于Unlikelihood Training思想)
    
    Args:
        logits (torch.Tensor): 模型的原始输出 [batch, seq_len, vocab_size]
        labels (torch.Tensor): 真实标签 [batch, seq_len]
        mask (torch.Tensor): 损失掩码 [batch, seq_len]
        window_size (int):  回顾窗口大小，即惩罚最近多少个token的重复
        penalty_weight (float): 惩罚项的权重
        
    Returns:
        torch.Tensor: 一个标量的惩罚损失
    """
    # 我们仅在batch_size=1的情况下实现，与你的compute_loss保持一致
    if logits.size(0) != 1:
        # 在多batch size下需要更复杂的padding和mask处理，这里简化
        return torch.tensor(0.0, device=logits.device)
        
    batch_size, seq_len, vocab_size = logits.shape
    
    # 1. 将logits转换为概率分布
    # 使用log_softmax更稳定，因为我们要计算log(1-p)
    log_probs = F.log_softmax(logits, dim=-1)
    
    total_penalty = torch.tensor(0.0, device=logits.device)
    num_penalized_tokens = 0

    # 2. 遍历序列，从第window_size个token开始
    for i in range(window_size, seq_len):
        if mask[0, i] == 0:  # 如果当前位置是padding，则跳过
            continue

        # 3. 确定负例集合：前window_size个真实token
        # 我们使用真实标签(labels)来确定负例，这比用模型预测更稳定
        # 且避免了惩罚模型正确预测出重复内容的情况
        # negative_candidates = labels[0, i - window_size : i]
        token_counts = {}
        for j in range(max(0, i - window_size), i):
            token = labels[0, j].item()
            token_counts[token] = token_counts.get(token, 0) + 1
        # 只将出现次数超过阈值的token作为负例
        negative_candidates = []
        for token, count in token_counts.items():
            if count >= 3:  # 可调整阈值
                negative_candidates.append(token)
        
        # 4. 获取模型在当前位置 i 对这些负例token预测的概率
        # log_probs[batch, seq_pos, token_indices]
        log_probs_of_negatives = log_probs[0, i, negative_candidates]
        
        # 5. 计算unlikelihood损失: -log(1 - p)
        # 为了数值稳定性，使用 log(1 - exp(log_p))
        # log_probs_of_negatives 是 log(p), 所以 exp(...) 就是 p
        probs_of_negatives = torch.exp(log_probs_of_negatives)
        
        # 加上一个很小的eps防止log(0)
        penalty_per_token = -torch.log(1 - probs_of_negatives + 1e-9)
        
        # 6. 累加惩罚
        total_penalty += penalty_per_token.sum()
        num_penalized_tokens += len(negative_candidates)

    # 对总惩罚进行平均，然后乘以权重
    if num_penalized_tokens > 0:
        avg_penalty = total_penalty / num_penalized_tokens
        return avg_penalty * penalty_weight
    else:
        return torch.tensor(0.0, device=logits.device)

def compute_loss(outputs, labels, num_items_in_batch=None):
    audio_logits, text_logits = outputs.logits
    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss_base = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)

    # --- 新增：可微分的重复惩罚 ---
    # 我们只对主分支（text_logits）施加这个惩罚，因为它更关键
    # text_loss_mask.view(1, -1) 是为了匹配函数内部的 batch_size=1 的假设
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")
    unlikelihood_penalty = compute_unlikelihood_penalty(
        text_logits, 
        text_labels, 
        text_loss_mask.view(1, -1), 
        window_size=5,       # 超参数1：回顾窗口
        penalty_weight=0.05  # 超参数2：惩罚权重
    )

    text_loss_total = text_loss_base + unlikelihood_penalty
    loss = audio_loss + text_loss_total

    #打印出来观察一下惩罚项的大小
    print(f"CE Loss: {text_loss_base.item():.4f}, Repetition Penalty: {unlikelihood_penalty.item():.4f}, Length of current text_logits: {text_logits.shape[1]}")
    
    return loss

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning(
                "FSDP and ZeRO3 are not supported for QLoRA."
            )

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info(f"Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        cache_path = model_args.model_name_or_path
    else:
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    # Load model with LoRA configuration
    if training_args.use_lora:
        if lora_args.q_lora:
            model = KimiAudioModel.from_pretrained(
                model_args.model_path,
                device_map=None,
                quantization_config=GPTQConfig(
                    bits=4, disable_exllama=True
                ),
                **model_load_kwargs
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        else:
            # ToDo: should use init_from_pretrained when lora ?
            model = KimiAudioModel.from_pretrained(
                model_args.model_path,
                device_map=None,
                **model_load_kwargs
            )


        if training_args.use_lora:
            if lora_args.q_lora:
                modules_to_save = None
            else:
                modules_to_save = None #["wte" "lm_head"]
        if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and modules_to_save is not None
        ):
            raise RuntimeError("Deepspeed ZeRO3 is not supported for LoRA when using modules_to_save")

        # Configure LoRA
        lora_config = get_optimized_lora_config(lora_args)
        
        model = get_peft_model(model, lora_config)
        print_lora_info(lora_config, model)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        model = KimiAudioModel.from_pretrained(
            model_args.model_path,
            device_map=None,
            **model_load_kwargs
        )

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model if hasattr(model, 'whisper_model') else model.base_model.whisper_model,
        text_tokenizer=text_tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model.config.kimia_token_offset if hasattr(model, 'config') else model.base_model.config.kimia_token_offset
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


    # 统一的模型保存逻辑
    if training_args.use_lora:
        # LoRA模型保存策略
        if lora_args.save_strategy == "lora_only":
            # 只保存LoRA权重
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "lora_weights"),
                bias=lora_args.lora_bias,
                save_lora_only=True,
                merge_lora=False,
                lora_args=lora_args
            )
        elif lora_args.save_strategy == "merged":
            # 保存合并后的模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "merged_model"),
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=True,
                lora_args=lora_args
            )
        elif lora_args.save_strategy == "both":
            # 同时保存LoRA权重和合并后的模型
            # 1. 保存LoRA权重
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "lora_weights"),
                bias=lora_args.lora_bias,
                save_lora_only=True,
                merge_lora=False,
                lora_args=lora_args
            )
            # 2. 保存合并后的模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=os.path.join(training_args.output_dir, "merged_model"),
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=True,
                lora_args=lora_args
            )
        else:
            # 默认：保存完整的PEFT模型
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=training_args.output_dir,
                bias=lora_args.lora_bias,
                save_lora_only=False,
                merge_lora=False,
                lora_args=lora_args
            )
    else:
        # 普通模型保存
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            bias="none"
        )

if __name__ == "__main__":
    train()