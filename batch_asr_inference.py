import json
import os
from kimia_infer.api.kimia import KimiAudio
from tqdm import tqdm
import time

def load_jsonl_data(jsonl_path):
    """加载JSONL文件中的数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

def extract_audio_info(conversation):
    """从对话中提取音频路径和groundtruth"""
    audio_path = None
    groundtruth = None
    
    audio_path = conversation[1]['content']
    groundtruth = conversation[2]['content']
    
    return audio_path, groundtruth

def main():
    # 模型路径和参数配置
    model_path = "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora7/export_model"
    jsonl_path = "/team/shared/data_term_test/testset0605/audio_data.jsonl"
    output_file = "/team/shared/data_term_test/result/testset0605_kimi_06112w.json"
    
    # 只加载一次模型
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print("Model loaded successfully!")
    
    # 采样参数
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.3,
        "text_repetition_window_size": 16,
    }
    
    # 加载数据
    print(f"Loading data from {jsonl_path}...")
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} samples")
    
    # 批量推理
    results = []
    for idx, item in enumerate(tqdm(data, desc="Processing samples")):
        conversation = item['conversation']
        audio_path, groundtruth = extract_audio_info(conversation)
        term = item['term']
        
        if audio_path is None or groundtruth is None:
            print(f"Warning: Skipping sample {idx} due to missing audio path or groundtruth")
            continue
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            results.append({
                "index": idx,
                "audio_path": audio_path,
                "groundtruth": groundtruth,
                "model_output": "ERROR: Audio file not found",
                "error": True
            })
            continue
        
        try:
            # 构建消息格式
            messages = [
                {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
                {"role": "user", "message_type": "audio", "content": audio_path},
            ]
            
            # 生成结果
            start_time = time.time()
            wav, text = model.generate(messages, **sampling_params, output_type="text")
            inference_time = time.time() - start_time
            
            # 保存结果
            result = {
                "index": idx,
                "audio_path": audio_path,
                "term": term,
                "groundtruth": groundtruth,
                "model_output": text,
                "inference_time": inference_time,
                "error": False
            }
            results.append(result)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            results.append({
                "index": idx,
                "audio_path": audio_path,
                "groundtruth": groundtruth,
                "model_output": f"ERROR: {str(e)}",
                "error": True
            })
    
    # 保存结果到JSON文件
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算统计信息
    successful_results = [r for r in results if not r.get('error', False)]
    failed_results = [r for r in results if r.get('error', False)]
    
    print(f"\n========== Summary ==========")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        avg_time = sum(r['inference_time'] for r in successful_results) / len(successful_results)
        print(f"Average inference time: {avg_time:.2f}s")
    
    # 可选：计算简单的准确率（基于完全匹配）
    exact_matches = sum(1 for r in successful_results if r['model_output'].strip() == r['groundtruth'].strip())
    if successful_results:
        print(f"Exact match accuracy: {exact_matches}/{len(successful_results)} ({exact_matches/len(successful_results)*100:.2f}%)")
    
    # 可选：保存错误样本到单独文件
    if failed_results:
        with open("failed_samples.json", 'w', encoding='utf-8') as f:
            json.dump(failed_results, f, ensure_ascii=False, indent=2)
        print(f"\nFailed samples saved to failed_samples.json")
    
    print("\nBatch inference completed!")

if __name__ == "__main__":
    main()