import json
import os
from kimia_infer.api.kimia import KimiAudio
from tqdm import tqdm
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager
import torch

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

def process_batch(args):
    """在单个GPU上处理一批数据"""
    gpu_id, data_batch, model_path, sampling_params, batch_id = args
    
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # 在进程内部，设备总是0
    
    # 在每个进程中加载模型
    print(f"[GPU {gpu_id}] Loading model...")
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print(f"[GPU {gpu_id}] Model loaded successfully!")
    
    results = []
    
    # 为每个GPU创建独立的进度条
    desc = f"GPU {gpu_id}"
    for item in tqdm(data_batch, desc=desc, position=gpu_id):
        idx = item['original_index']
        conversation = item['conversation']
        audio_path, groundtruth = extract_audio_info(conversation)
        term = item['term']
        
        if audio_path is None or groundtruth is None:
            print(f"[GPU {gpu_id}] Warning: Skipping sample {idx} due to missing audio path or groundtruth")
            continue
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"[GPU {gpu_id}] Warning: Audio file not found: {audio_path}")
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
            print(f"[GPU {gpu_id}] Error processing sample {idx}: {str(e)}")
            results.append({
                "index": idx,
                "audio_path": audio_path,
                "groundtruth": groundtruth,
                "model_output": f"ERROR: {str(e)}",
                "error": True
            })
    
    return results

def main():
    # 模型路径和参数配置
    model_path = "/team/shared/kimi-audio-training-result/train_model_output/wushen/output/kimiaudio_lora7/export_model"
    jsonl_path = "/team/shared/data_term_test/testset0605/audio_data.jsonl"
    output_file = "/team/shared/data_term_test/result/testset0605_kimi_06112w_8gpu.json"
    
    # GPU数量
    num_gpus = 8
    
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
    
    # 加载数据
    print(f"Loading data from {jsonl_path}...")
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} samples")
    
    # 为每个样本添加原始索引
    for idx, item in enumerate(data):
        item['original_index'] = idx
    
    # 将数据分配给不同的GPU
    data_per_gpu = len(data) // num_gpus
    data_batches = []
    
    for i in range(num_gpus):
        start_idx = i * data_per_gpu
        if i == num_gpus - 1:
            # 最后一个GPU处理剩余的所有数据
            batch = data[start_idx:]
        else:
            batch = data[start_idx:start_idx + data_per_gpu]
        data_batches.append(batch)
    
    # 准备多进程参数
    process_args = []
    for i in range(num_gpus):
        args = (i, data_batches[i], model_path, sampling_params, i)
        process_args.append(args)
    
    # 设置spawn方法以避免CUDA问题
    mp.set_start_method('spawn', force=True)
    
    # 使用多进程池并行处理
    print(f"\nStarting parallel inference on {num_gpus} GPUs...")
    start_time = time.time()
    
    with Pool(num_gpus) as pool:
        all_results = pool.map(process_batch, process_args)
    
    # 合并所有结果
    results = []
    for gpu_results in all_results:
        results.extend(gpu_results)
    
    # 按原始索引排序
    results.sort(key=lambda x: x['index'])
    
    total_time = time.time() - start_time
    
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
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per sample: {total_time/len(results):.2f}s")
    
    if successful_results:
        avg_inference_time = sum(r['inference_time'] for r in successful_results) / len(successful_results)
        print(f"Average inference time per sample: {avg_inference_time:.2f}s")
    
    # 计算简单的准确率（基于完全匹配）
    exact_matches = sum(1 for r in successful_results if r['model_output'].strip() == r['groundtruth'].strip())
    if successful_results:
        print(f"Exact match accuracy: {exact_matches}/{len(successful_results)} ({exact_matches/len(successful_results)*100:.2f}%)")
    
    # 保存错误样本到单独文件
    if failed_results:
        error_file = output_file.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(failed_results, f, ensure_ascii=False, indent=2)
        print(f"\nFailed samples saved to {error_file}")
    
    print("\nBatch inference completed!")

if __name__ == "__main__":
    main()