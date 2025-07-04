from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse
import json

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

    messages = [
        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": "/team/shared/data_regression/dataset_phase1/acoustic/real_scene_robustness/clothing_supply/clothing_supply_case1_009_16k_mono.wav",
        },
    ]

    wav, text = model.generate(messages, **sampling_params, output_type="text")
    print(">>> output text: ", text)
    output_file = "/team/haoyiya.hyy/output/single_wav_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=2)

    # output_dir = "test_audios/output"
    # os.makedirs(output_dir, exist_ok=True)
    # # audio2audio
    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/qa_example.wav",
    #     }
    # ]

    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # sf.write(
    #     os.path.join(output_dir, "output.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)


    # # audio2audio multiturn
    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case1/multiturn_q1.wav",
    #     },
    #     {
    #         "role": "assistant",
    #         "message_type": "audio-text",
    #         "content": ["test_audios/multiturn/case1/multiturn_a1.wav", "当然可以，李白的诗很多，比如这句：“床前明月光，疑是地上霜。举头望明月，低头思故乡。"]
    #     },
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case1/multiturn_q2.wav",
    #     }
    # ]
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # sf.write(
    #     os.path.join(output_dir, "case_1_multiturn_a2.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)


    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case2/multiturn_q1.wav",
    #     },
    #     {
    #         "role": "assistant",
    #         "message_type": "audio-text",
    #         "content": ["test_audios/multiturn/case2/multiturn_a1.wav", "当然可以，这很简单。一二三四五六七八九十。"]
    #     },
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case2/multiturn_q2.wav",
    #     }
    # ]
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # sf.write(
    #     os.path.join(output_dir, "case_2_multiturn_a2.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)
