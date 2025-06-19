import os
import argparse
from typing import Optional, List
import shutil
import torch
from transformers import AutoModelForCausalLM, GenerationMixin
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from .modeling_kimia import MoonshotKimiaForCausalLM


class KimiAudioModel(MoonshotKimiaForCausalLM, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model
    
    @staticmethod
    def export_model(input_dir, output_dir, enable_lora=False, checkpoint=None, base_model_path=None):
        print("Loading model from {}".format(input_dir))
        
        # Determine paths for base model and LoRA weights
        if enable_lora:
            # Set LoRA path
            if checkpoint:
                # Check if checkpoint is provided as a subdirectory
                potential_lora_path = os.path.join(input_dir, checkpoint)
                if os.path.exists(potential_lora_path) and os.path.exists(os.path.join(potential_lora_path, "adapter_config.json")):
                    lora_path = potential_lora_path
                elif os.path.exists(os.path.join(input_dir, "adapter_config.json")):
                    # input_dir is already the checkpoint directory
                    lora_path = input_dir
                else:
                    raise ValueError(f"Cannot find LoRA adapter files in {input_dir} or {potential_lora_path}")
            else:
                # No specific checkpoint, use input_dir
                lora_path = input_dir
                if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
                    raise ValueError(f"Cannot find adapter_config.json in {lora_path}")
            
            # Determine base model path
            if base_model_path is None:
                # Try to read from adapter_config.json
                import json
                adapter_config_path = os.path.join(lora_path, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_path = adapter_config.get("base_model_name_or_path", "moonshotai/Kimi-Audio-7B")
                else:
                    base_model_path = "moonshotai/Kimi-Audio-7B"
            
            print(f"Loading base model from: {base_model_path}")
            print(f"Loading LoRA weights from: {lora_path}")
        else:
            # Not using LoRA, load full model directly
            base_model_path = input_dir
            lora_path = None
        
        # Load the model
        if enable_lora:
            # For LoRA, we need to load the base model using init_from_pretrained
            kimiaudio = KimiAudioModel.init_from_pretrained(base_model_path, model_load_kwargs={})
        else:
            # For full model, use from_pretrained
            kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        if enable_lora:
            from peft import PeftModel
            kimiaudio = PeftModel.from_pretrained(kimiaudio, lora_path)
            kimiaudio = kimiaudio.merge_and_unload()
            print(f"Successfully loaded and merged LoRA weights from {lora_path}")

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))

    @staticmethod
    def export_all_checkpoints(input_dir, output_dir, enable_lora=True, base_model_path=None):
        """Export all checkpoints found in the input directory."""
        checkpoints = []
        
        # Find all checkpoint directories
        for item in os.listdir(input_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(input_dir, item)):
                checkpoints.append(item)
        
        if not checkpoints:
            print(f"No checkpoints found in {input_dir}")
            return
        
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        
        for checkpoint in checkpoints:
            checkpoint_output_dir = os.path.join(output_dir, checkpoint)
            print(f"\n{'='*50}")
            print(f"Exporting {checkpoint}...")
            print(f"{'='*50}")
            
            try:
                KimiAudioModel.export_model(
                    input_dir=input_dir,
                    output_dir=checkpoint_output_dir,
                    enable_lora=enable_lora,
                    checkpoint=checkpoint,
                    base_model_path=base_model_path
                )
                print(f"Successfully exported {checkpoint}")
            except Exception as e:
                print(f"Failed to export {checkpoint}: {str(e)}")
                continue

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())
        whisper_feats = self.whisper_model(whisper_input_feats)
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model", "export_all_checkpoints"], 
                        default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    parser.add_argument("--enable_lora", action="store_true", help="Enable LoRA model loading")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Specific checkpoint to export (e.g., checkpoint-100)")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model (for LoRA). If not specified, will try to read from adapter_config.json")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":
        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})
        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(
            args.input_dir, 
            args.output_dir, 
            args.enable_lora,
            args.checkpoint,
            args.base_model_path
        )
    elif args.action == "export_all_checkpoints":
        KimiAudioModel.export_all_checkpoints(
            args.input_dir,
            args.output_dir,
            args.enable_lora,
            args.base_model_path
        )
    else:
        raise ValueError(f"Invalid action: {args.action}") 

'''
# 转换特定的 checkpoint-100（自动从adapter_config.json读取base model路径）
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model \
    --model_name "moonshotai/Kimi-Audio-7B" \
    --action "export_model" \
    --input_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7" \
    --checkpoint "checkpoint-100" \
    --output_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7/export_model_test" \
    --enable_lora

# 转换特定的 checkpoint-100（手动指定base model路径）
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model \
    --model_name "moonshotai/Kimi-Audio-7B" \
    --action "export_model" \
    --input_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7" \
    --checkpoint "checkpoint-100" \
    --output_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7/export_model_test" \
    --base_model_path "moonshotai/Kimi-Audio-7B" \
    --enable_lora

# 批量导出所有LoRA checkpoints
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model \
    --model_name "moonshotai/Kimi-Audio-7B" \
    --action "export_all_checkpoints" \
    --input_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7" \
    --output_dir "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora_7/export_model" \
    --enable_lora
'''