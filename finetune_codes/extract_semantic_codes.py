import argparse
import os
import json
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import tqdm
from functools import partial
import numpy as np

from kimia_infer.api.prompt_manager import KimiAPromptManager


class AudioDataset(Dataset):
    """Dataset class for loading audio data"""
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return idx, self.data_list[idx]


def collate_fn(batch):
    """Custom collate function to handle batch data"""
    indices, data_items = zip(*batch)
    return list(indices), list(data_items)


def process_batch(batch_indices, batch_data, prompt_manager, device_id):
    """Process a batch of data on a specific GPU"""
    torch.cuda.set_device(device_id)
    
    results = []
    for idx, data in zip(batch_indices, batch_data):
        # Process each conversation in the data
        for msg in data["conversation"]:
            if msg["message_type"] == "audio":
                audio_path = msg["content"]
                try:
                    # Move tokenization to the specific GPU
                    with torch.cuda.device(device_id):
                        audio_tokens = prompt_manager._tokenize_audio(audio_path)
                    msg["audio_tokens"] = audio_tokens
                except Exception as e:
                    print(f"Error processing audio {audio_path}: {e}")
                    msg["audio_tokens"] = []
        
        results.append((idx, data))
    
    return results


def worker_process(rank, world_size, args, data_queue, result_queue, barrier):
    """Worker process for multi-GPU processing"""
    # Set CUDA device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"Worker {rank} starting on GPU {rank}")
    
    # Load model config
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)
    
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    
    # Initialize prompt manager on specific GPU
    with torch.cuda.device(rank):
        prompt_manager = KimiAPromptManager(
            model_path=cache_path, 
            kimia_token_offset=model_config.kimia_token_offset,
            kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
        )
    
    # Synchronize all workers
    barrier.wait()
    
    # Process data from queue
    while True:
        try:
            batch_data = data_queue.get(timeout=1)
            if batch_data is None:  # Sentinel value to stop
                break
            
            batch_indices, batch_items = batch_data
            results = process_batch(batch_indices, batch_items, prompt_manager, rank)
            result_queue.put(results)
            
        except Exception as e:
            if not data_queue.empty():
                print(f"Worker {rank} error: {e}")
            continue
    
    print(f"Worker {rank} finished")


def write_results(output_file, total_lines, result_queue, num_workers):
    """Write results to file in correct order"""
    results_dict = {}
    processed_count = 0
    workers_finished = 0
    
    with open(output_file, 'w') as f_out:
        next_write_idx = 0
        pbar = tqdm.tqdm(total=total_lines, desc="Writing results")
        
        while workers_finished < num_workers or not result_queue.empty():
            try:
                results = result_queue.get(timeout=0.1)
                
                if results is None:  # Worker finished signal
                    workers_finished += 1
                    continue
                
                # Store results in dictionary
                for idx, data in results:
                    results_dict[idx] = data
                    processed_count += 1
                
                # Write results in order
                while next_write_idx in results_dict:
                    data = results_dict.pop(next_write_idx)
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    next_write_idx += 1
                    pbar.update(1)
                    
            except:
                continue
        
        pbar.close()
        
        # Write any remaining results
        while next_write_idx in results_dict:
            data = results_dict.pop(next_write_idx)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            next_write_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--num_data_workers", type=int, default=24, help="Number of data loading workers")
    args = parser.parse_args()
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Determine number of GPUs
    if args.num_workers is None:
        args.num_workers = torch.cuda.device_count()
    else:
        args.num_workers = min(args.num_workers, torch.cuda.device_count())
    
    print(f"Using {args.num_workers} GPUs for processing")
    
    # Load data
    print("Loading data...")
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]
    
    total_lines = len(all_data)
    print(f"Total samples: {total_lines}")
    
    # Create dataset and dataloader
    dataset = AudioDataset(all_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size * args.num_workers,
        shuffle=False,
        num_workers=args.num_data_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create queues for communication
    data_queue = mp.Queue(maxsize=args.num_workers * 4)  # Limit queue size
    result_queue = mp.Queue()
    barrier = mp.Barrier(args.num_workers)
    
    # Start worker processes
    workers = []
    for rank in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(rank, args.num_workers, args, data_queue, result_queue, barrier)
        )
        p.start()
        workers.append(p)
    
    # Start result writer thread
    import threading
    writer_thread = threading.Thread(
        target=write_results,
        args=(args.output_file, total_lines, result_queue, args.num_workers)
    )
    writer_thread.start()
    
    # Distribute data to workers
    print("Distributing data to workers...")
    for batch_indices, batch_data in tqdm.tqdm(dataloader, desc="Loading batches"):
        # Split batch among workers
        batch_size_per_worker = len(batch_indices) // args.num_workers
        for i in range(args.num_workers):
            start_idx = i * batch_size_per_worker
            end_idx = start_idx + batch_size_per_worker if i < args.num_workers - 1 else len(batch_indices)
            
            worker_indices = batch_indices[start_idx:end_idx]
            worker_data = batch_data[start_idx:end_idx]
            
            if worker_indices:
                data_queue.put((worker_indices, worker_data))
    
    # Send stop signals to workers
    for _ in range(args.num_workers):
        data_queue.put(None)
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    # Send finish signals to writer
    for _ in range(args.num_workers):
        result_queue.put(None)
    
    # Wait for writer to finish
    writer_thread.join()
    
    print(f"Processing complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()