from dataclasses import dataclass, field
from typing import List
from peft import LoraConfig, TaskType, PeftModel
import logging
import os
import json
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    modules_to_save: List[str] = field(
        default=None
    )
    lora_save_strategy: str = field(
        default="merged",
        metadata={
            "help": "LoRA model save strategy: 'default' (save PEFT model), "
                   "'lora_only' (save only LoRA weights), "
                   "'merged' (save merged model), "
                   "'both' (save both LoRA weights and merged model)"
        }
    )
    # 新增：是否在训练结束后合并LoRA权重
    merge_lora_after_training: bool = field(
        default=False,
        metadata={"help": "Whether to merge LoRA weights after training (deprecated, use save_strategy instead)"}
    )
    # 新增：配置目标模块的参数
    num_text_layers: int = field(
        default=28, metadata={"help": "Number of text transformer layers"}
    )
    num_mimo_layers: int = field(
        default=6, metadata={"help": "Number of MIMO layers"}
    )
    # 新增：配置层数范围的参数
    text_layer_start: int = field(
        default=0, metadata={"help": "Start index of text layers to apply LoRA"}
    )
    text_layer_end: int = field(
        default=None, metadata={"help": "End index of text layers to apply LoRA (exclusive). If None, use num_text_layers"}
    )
    mimo_layer_start: int = field(
        default=0, metadata={"help": "Start index of MIMO layers to apply LoRA"}
    )
    mimo_layer_end: int = field(
        default=None, metadata={"help": "End index of MIMO layers to apply LoRA (exclusive). If None, use num_mimo_layers"}
    )
    target_text_modules: bool = field(
        default=True, metadata={"help": "Whether to apply LoRA to text-related modules"}
    )
    target_audio_modules: bool = field(
        default=True, metadata={"help": "Whether to apply LoRA to audio-related modules"}
    )
    target_mimo_modules: bool = field(
        default=True, metadata={"help": "Whether to apply LoRA to MIMO layers"}
    )
    # 新增：配置输出头的参数
    target_lm_head: bool = field(
        default=True, metadata={"help": "Whether to apply LoRA to language model head"}
    )
    target_mimo_output: bool = field(
        default=True, metadata={"help": "Whether to apply LoRA to MIMO output head"}
    )

def get_optimized_lora_config(lora_args: LoraArguments) -> LoraConfig:
    """
    Get optimized LoRA configuration for KimiAudio
    
    Args:
        lora_args: LoRA arguments containing r, alpha, dropout, etc.
    """
    target_modules = []
    
    # 设置层数范围
    text_layer_end = lora_args.text_layer_end if lora_args.text_layer_end is not None else lora_args.num_text_layers
    mimo_layer_end = lora_args.mimo_layer_end if lora_args.mimo_layer_end is not None else lora_args.num_mimo_layers
    
    # 验证层数范围的有效性
    if lora_args.text_layer_start < 0 or lora_args.text_layer_start >= lora_args.num_text_layers:
        raise ValueError(f"text_layer_start must be between 0 and {lora_args.num_text_layers-1}")
    if text_layer_end <= lora_args.text_layer_start or text_layer_end > lora_args.num_text_layers:
        raise ValueError(f"text_layer_end must be between {lora_args.text_layer_start+1} and {lora_args.num_text_layers}")
    
    if lora_args.mimo_layer_start < 0 or lora_args.mimo_layer_start >= lora_args.num_mimo_layers:
        raise ValueError(f"mimo_layer_start must be between 0 and {lora_args.num_mimo_layers-1}")
    if mimo_layer_end <= lora_args.mimo_layer_start or mimo_layer_end > lora_args.num_mimo_layers:
        raise ValueError(f"mimo_layer_end must be between {lora_args.mimo_layer_start+1} and {lora_args.num_mimo_layers}")
    
    # 如果用户指定了target_modules，优先使用用户的配置
    if lora_args.lora_target_modules:
        # 扩展用户指定的模块到所有层
        for module_suffix in lora_args.lora_target_modules:
            # Base transformer layer modules
            if lora_args.target_text_modules:
                for i in range(lora_args.text_layer_start, text_layer_end):
                    target_modules.append(f"model.layers.{i}.self_attn.{module_suffix}")
                    #if module_suffix in ["gate_proj", "up_proj", "down_proj"]:
                    #    target_modules.append(f"model.layers.{i}.mlp.{module_suffix}")
            
            # MIMO layers
            if lora_args.target_mimo_modules:
                for i in range(lora_args.mimo_layer_start, mimo_layer_end):
                    target_modules.append(f"model.mimo_layers.{i}.self_attn.{module_suffix}")
                    #if module_suffix in ["gate_proj", "up_proj", "down_proj"]:
                    #    target_modules.append(f"model.mimo_layers.{i}.mlp.{module_suffix}")
    else:
        # 使用默认的全面配置
        if lora_args.target_text_modules:
            for i in range(lora_args.text_layer_start, text_layer_end):
                target_modules.extend([
                    f"model.layers.{i}.self_attn.q_proj",
                    f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj",
                    f"model.layers.{i}.self_attn.o_proj",
                    #f"model.layers.{i}.mlp.gate_proj",
                    #f"model.layers.{i}.mlp.up_proj",
                    #f"model.layers.{i}.mlp.down_proj",
                ])
        
        if lora_args.target_mimo_modules:
            for i in range(lora_args.mimo_layer_start, mimo_layer_end):
                target_modules.extend([
                    f"model.mimo_layers.{i}.self_attn.q_proj",
                    f"model.mimo_layers.{i}.self_attn.k_proj",
                    f"model.mimo_layers.{i}.self_attn.v_proj",
                    f"model.mimo_layers.{i}.self_attn.o_proj",
                    #f"model.mimo_layers.{i}.mlp.gate_proj",
                    #f"model.mimo_layers.{i}.mlp.up_proj",
                    #f"model.mimo_layers.{i}.mlp.down_proj",
                ])
    
    # Audio adapter (always include if target_audio_modules is True)
    if lora_args.target_audio_modules:
        target_modules.extend([
            "model.vq_adaptor.layers.0",  # First linear layer
            "model.vq_adaptor.layers.3",  # Second linear layer
        ])
    
    # Output heads (根据配置决定是否包含)
    if lora_args.target_lm_head:
        target_modules.append("lm_head")
    if lora_args.target_mimo_output:
        target_modules.append("mimo_output")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha, # Typically set to 2x the rank
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None, # Don't save full modules, use LoRA for all
    )

    return lora_config 

def merge_and_save(peft_model, output_dir: str, tokenizer=None):
    """Merge LoRA weights and save the model"""
    logger.info(f"Merging LoRA weights and saving to {output_dir}")
    assert isinstance(peft_model, PeftModel), "model must be a PeftModel when merging"

    # Merge LoRA weights
    merged_model = peft_model.merge_and_unload()
    
    # Save the merged model
    merged_model.save_pretrained(output_dir)
    
    # Save tokenizer if available
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    logger.info("Model merged and saved successfully")

def save_lora_only(peft_model, lora_args, output_dir: str):
    """Save only LoRA weights"""
    logger.info(f"Saving LoRA weights to {output_dir}")
    assert isinstance(peft_model, PeftModel), "model must be a PeftModel when saving"
    peft_model.save_pretrained(output_dir)
    
    # Save LoRA config
    lora_config_path = os.path.join(output_dir, "lora_config.json")
    with open(lora_config_path, 'w') as f:
        json.dump(vars(lora_args), f, indent=2)

def print_lora_info(lora_config, peft_model):
    """Print LoRA configuration and statistics"""
    logger.info("="*50)
    logger.info("LoRA Configuration:")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    logger.info(f"  Dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    # Detailed parameter statistics
    total_params = 0
    trainable_params = 0
    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"  Trainable: {name} - {param.shape}")
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    logger.info("="*50)

def save_lora_weights(model, state_dict, output_dir, lora_args=None):
    """
    保存LoRA权重和配置
    
    Args:
        model: PEFT模型
        state_dict: LoRA权重字典
        output_dir: 输出目录
        lora_args: LoRA配置参数
    """
    # 保存LoRA权重
    torch.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))
    
    # 保存PEFT配置
    if hasattr(model, 'peft_config'):
        peft_config = model.peft_config
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
        peft_config = model.base_model.peft_config
    else:
        peft_config = None
    
    if peft_config:
        # 保存adapter配置
        for adapter_name, config in peft_config.items():
            config.save_pretrained(output_dir)
            break  # 通常只有一个adapter
    
    # 如果提供了lora_args，也保存一份用于参考
    if lora_args:
        lora_config_dict = {
            "r": lora_args.lora_r,
            "lora_alpha": lora_args.lora_alpha,
            "lora_dropout": lora_args.lora_dropout,
            "target_modules": lora_args.lora_target_modules,
            "modules_to_save": lora_args.modules_to_save,
            "bias": lora_args.lora_bias,
            "task_type": "CAUSAL_LM",
        }
        with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
            json.dump(lora_config_dict, f, indent=2)
    
    #rank0_print(f"LoRA weights saved to {output_dir}")