import json
import os
from kimia_infer.api.kimia import KimiAudio
from tqdm import tqdm
import time
from multiprocessing import Pool, Manager, Queue, Process
import torch
import threading
from queue import Empty

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

def worker_process(gpu_id, task_queue, result_queue, model_path, model_pretrained, sampling_params):
    """工作进程：从队列中获取任务并处理"""
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 加载模型
    print(f"GPU {gpu_id}: Loading model...")
    if model_pretrained:
        model = KimiAudio(
            model_path=model_path,
            load_detokenizer=True,
        )
    else:
        model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print(f"GPU {gpu_id}: Model loaded successfully!")
    
    # 处理任务
    while True:
        try:
            # 获取任务（超时1秒）
            task = task_queue.get(timeout=1)
            
            if task is None:  # 结束信号
                break
            
            idx, item = task
            conversation = item['conversation']
            audio_path, groundtruth = extract_audio_info(conversation)
            term = item['term']
            
            if audio_path is None or groundtruth is None:
                result = {
                    "index": idx,
                    "audio_path": audio_path,
                    "groundtruth": groundtruth,
                    "model_output": "ERROR: Missing audio path or groundtruth",
                    "error": True,
                    "gpu_id": gpu_id
                }
            elif not os.path.exists(audio_path):
                result = {
                    "index": idx,
                    "audio_path": audio_path,
                    "groundtruth": groundtruth,
                    "model_output": "ERROR: Audio file not found",
                    "error": True,
                    "gpu_id": gpu_id
                }
            else:
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
                    
                    result = {
                        "index": idx,
                        "audio_path": audio_path,
                        "term": term,
                        "groundtruth": groundtruth,
                        "model_output": text,
                        "inference_time": inference_time,
                        "error": False,
                        "gpu_id": gpu_id
                    }
                except Exception as e:
                    result = {
                        "index": idx,
                        "audio_path": audio_path,
                        "groundtruth": groundtruth,
                        "model_output": f"ERROR: {str(e)}",
                        "error": True,
                        "gpu_id": gpu_id
                    }
            
            # 发送结果
            result_queue.put(result)
            
        except Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"GPU {gpu_id}: Error in worker process: {str(e)}")

def main(model_pretrained=True):
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU available! Please check your CUDA installation.")
        return
    
    print(f"Found {num_gpus} GPUs")
    
    # 模型路径和参数配置
    if model_pretrained:
        model_path = "moonshotai/Kimi-Audio-7B-Instruct"
    else:
        model_path = "output/finetuned_hf_for_inference"
    jsonl_path = "/opt/data/nvme4/kimi/data/testset0605/audio_data.jsonl"
    output_file = "testset0605_kimi_original.json"
    
    # 采样参数
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.2,
        "text_repetition_window_size": 16,
    }
    
    # 加载数据
    print(f"Loading data from {jsonl_path}...")
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} samples")
    
    # 创建任务队列和结果队列
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # 将所有任务放入队列
    for idx, item in enumerate(data):
        task_queue.put((idx, item))
    
    # 添加结束信号
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 启动工作进程
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, model_path, model_pretrained, sampling_params)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    print("\nProcessing samples on multiple GPUs...")
    results = []
    start_time = time.time()
    
    with tqdm(total=len(data), desc="Processing samples") as pbar:
        for _ in range(len(data)):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
            
            # 实时显示进度
            if len(results) % 10 == 0:
                gpu_usage = {}
                for r in results:
                    gpu_id = r.get('gpu_id', -1)
                    gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
                pbar.set_postfix({f"GPU{k}": v for k, v in sorted(gpu_usage.items())})
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # 按原始索引排序结果
    results.sort(key=lambda x: x['index'])
    
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
    print(f"Speedup: {len(results)/total_time:.2f} samples/s")
    
    if successful_results:
        avg_inference_time = sum(r['inference_time'] for r in successful_results) / len(successful_results)
        print(f"Average inference time per sample: {avg_inference_time:.2f}s")
    
    # GPU使用统计
    gpu_stats = {}
    for result in results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {'total': 0, 'successful': 0, 'failed': 0, 'total_time': 0}
        gpu_stats[gpu_id]['total'] += 1
        if result.get('error', False):
            gpu_stats[gpu_id]['failed'] += 1
        else:
            gpu_stats[gpu_id]['successful'] += 1
            gpu_stats[gpu_id]['total_time'] += result.get('inference_time', 0)
    
    print("\nGPU Usage Statistics:")
    for gpu_id in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_id]
        avg_time = stats['total_time'] / stats['successful'] if stats['successful'] > 0 else 0
        print(f"  GPU {gpu_id}: {stats['total']} total, {stats['successful']} successful, "
              f"{stats['failed']} failed, avg time: {avg_time:.2f}s")
    
    # 计算简单的准确率（基于完全匹配）
    exact_matches = sum(1 for r in successful_results if r['model_output'].strip() == r['groundtruth'].strip())
    if successful_results:
        print(f"\nExact match accuracy: {exact_matches}/{len(successful_results)} ({exact_matches/len(successful_results)*100:.2f}%)")
    
    # 保存错误样本到单独文件
    if failed_results:
        with open("failed_samples.json", 'w', encoding='utf-8') as f:
            json.dump(failed_results, f, ensure_ascii=False, indent=2)
        print(f"\nFailed samples saved to failed_samples.json")
    
    print("\nBatch inference completed!")

if __name__ == "__main__":
    import sys
    import multiprocessing
    
    # 设置启动方法为spawn，避免CUDA fork问题
    multiprocessing.set_start_method('spawn', force=True)
    
    # 支持命令行参数：python script.py [model_pretrained] [num_gpus]
    model_pretrained = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    
    main(model_pretrained)