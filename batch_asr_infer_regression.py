from kimia_infer.api.kimia import KimiAudio
import os
import argparse
import glob

def process_audio_files(model, base_path, output_base_path, sampling_params):
    """处理指定路径下的音频文件并生成结果文件"""
    
    # 两个主要场景文件夹
    scenarios = ['meeting_room_scenario', 'real_scene_robustness']
    
    for scenario in scenarios:
        scenario_path = os.path.join(base_path, scenario)
        # output_scenario_path = os.path.join(output_base_path, scenario)
        output_scenario_path = output_base_path
        
        # 确保输出目录存在
        os.makedirs(output_scenario_path, exist_ok=True)
        
        # 获取所有子文件夹
        subdirs = [d for d in os.listdir(scenario_path) 
                  if os.path.isdir(os.path.join(scenario_path, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(scenario_path, subdir)
            output_file = os.path.join(output_scenario_path, f"{subdir}.list")
            
            print(f"处理子文件夹: {subdir_path}")
            
            # 获取该子文件夹下的所有wav文件
            wav_files = glob.glob(os.path.join(subdir_path, "*.wav"))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for wav_file in sorted(wav_files):
                    # 获取文件名（不含扩展名）
                    filename = os.path.basename(wav_file).replace('.wav', '')
                    if '_16k_mono' in filename:
                        filename = filename[:filename.find('_16k_mono')]
                    # 构建消息
                    messages = [
                        {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": wav_file,
                        },
                    ]
                    
                    try:
                        # 生成转录文本
                        wav, text = model.generate(messages, **sampling_params, output_type="text")
                        
                        # 写入结果：文件名 + 空格 + 转录结果
                        f.write(f"{filename} {text}\n")
                        print(f"  已处理: {filename}")
                        
                    except Exception as e:
                        print(f"  处理 {filename} 时出错: {str(e)}")
                        f.write(f"{filename} [ERROR: {str(e)}]\n")
            
            print(f"  结果已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/finetuned_hf_for_inference0612v1")
    args = parser.parse_args()

    # 初始化模型
    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=False,
    )

    # 采样参数
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

    # 路径配置
    base_path = "/opt/data/nvme4/kimi/data/regression/audio/"
    output_base_path = "/opt/data/nvme4/kimi/data/regression/results/kimi_0612v1/"
    
    # 处理所有音频文件
    process_audio_files(model, base_path, output_base_path, sampling_params)