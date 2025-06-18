from kimia_infer.api.kimia import KimiAudio
import os
import argparse
import glob
import multiprocessing as mp
from queue import Empty
import time

def worker_process(gpu_id, model_path, task_queue, result_queue, sampling_params):
    """工作进程函数，每个GPU运行一个"""
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 初始化模型（只初始化一次）
    print(f"[GPU {gpu_id}] 正在初始化模型...")
    model = KimiAudio(
        model_path=model_path,
        load_detokenizer=False,
    )
    print(f"[GPU {gpu_id}] 模型初始化完成")
    
    # 持续处理任务直到收到结束信号
    while True:
        try:
            # 获取任务（wav文件路径）
            task = task_queue.get(timeout=1)
            
            if task is None:  # 结束信号
                break
                
            wav_file, task_id = task
            
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
                result = f"{filename} {text}"
                print(f"[GPU {gpu_id}] 已处理: {filename}")
                result_queue.put((task_id, result, None))
                
            except Exception as e:
                error_msg = f"{filename} [ERROR: {str(e)}]"
                print(f"[GPU {gpu_id}] 处理 {filename} 时出错: {str(e)}")
                result_queue.put((task_id, None, error_msg))
                
        except Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] 工作进程异常: {str(e)}")
            continue
    
    print(f"[GPU {gpu_id}] 工作进程结束")

def process_audio_files_parallel(model_path, base_path, output_base_path, sampling_params, num_gpus=8):
    """并行处理音频文件"""
    
    # 两个主要场景文件夹
    scenarios = ['meeting_room_scenario', 'real_scene_robustness']
    
    # 收集所有需要处理的任务
    all_tasks = []
    task_info = {}  # 存储每个任务对应的输出文件信息
    
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
            
            # 获取该子文件夹下的所有wav文件
            wav_files = glob.glob(os.path.join(subdir_path, "*.wav"))
            
            # 为每个wav文件创建任务
            for wav_file in sorted(wav_files):
                task_id = len(all_tasks)
                all_tasks.append((wav_file, task_id))
                task_info[task_id] = output_file
    
    print(f"总共需要处理 {len(all_tasks)} 个音频文件")
    print(f"使用 {num_gpus} 个GPU进行并行处理")
    
    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将所有任务放入队列
    for task in all_tasks:
        task_queue.put(task)
    
    # 添加结束信号
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 启动工作进程
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, model_path, task_queue, result_queue, sampling_params)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    results = {}
    completed = 0
    
    while completed < len(all_tasks):
        try:
            task_id, result, error = result_queue.get(timeout=1)
            results[task_id] = (result, error)
            completed += 1
            
            if completed % 10 == 0:
                print(f"进度: {completed}/{len(all_tasks)} ({completed/len(all_tasks)*100:.1f}%)")
                
        except Empty:
            continue
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    # 整理结果并写入文件
    output_data = {}  # 按输出文件组织结果
    
    for task_id, (result, error) in results.items():
        output_file = task_info[task_id]
        
        if output_file not in output_data:
            output_data[output_file] = []
        
        if result:
            output_data[output_file].append(result)
        elif error:
            output_data[output_file].append(error)
    
    # 写入所有结果文件
    for output_file, lines in output_data.items():
        # 按文件名排序
        lines.sort(key=lambda x: x.split()[0])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        print(f"结果已保存到: {output_file}")
    
    end_time = time.time()
    print(f"\n处理完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每个文件耗时: {(end_time - start_time) / len(all_tasks):.2f} 秒")

if __name__ == "__main__":
    # 设置multiprocessing使用spawn方法，避免CUDA初始化问题
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的GPU数量")
    args = parser.parse_args()

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
    base_path = "/workplace/haoyiya.hyy/project/data/test_set/regression/audio/"
    output_base_path = "~/projects/silero-vad/.src/silero_vad/data/.bin/projects/results/regression/kimi_original_p1_test/"
    
    # 并行处理所有音频文件
    process_audio_files_parallel(
        args.model_path, 
        base_path, 
        output_base_path, 
        sampling_params,
        num_gpus=args.num_gpus
    )