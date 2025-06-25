from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse
import json
from tqdm import tqdm

os.environ["HF_HOME"] = "/team/shared/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/team/shared/huggingface/hub"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    args = parser.parse_args()

    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=False,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    input_jsonl = "/team/shared/data_label/80h_v4.jsonl"
    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    for line in tqdm(lines):
        for msg in json.loads(line)["conversation"]:
            if msg['role'] == 'user' and msg['message_type'] == 'audio':
                audio_path = msg['content']
                messages = [
                    {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": "/team/shared/data_regression/dataset_phase1/acoustic/real_scene_robustness/clothing_supply/clothing_supply_case1_009_16k_mono.wav",
                    },
                ]

                wav, text = model.generate(messages, **sampling_params, output_type="text")
                if len(text) > 300:
                    output_file = "/team/haoyiya.hyy/output/single_wav_result.json"
                    output = {"audio_path" : audio_path, "Content" : text}
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(text, f, ensure_ascii=False, indent=2)
