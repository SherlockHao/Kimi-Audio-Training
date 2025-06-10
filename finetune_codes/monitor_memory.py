#!/usr/bin/env python3
# monitor_memory_simple.py - Monitor GPU memory using nvidia-smi

import subprocess
import psutil
import time
import datetime
import re
import os
import json

def get_gpu_memory_nvidia_smi():
    """Get GPU memory usage using nvidia-smi command"""
    try:
        # Run nvidia-smi command
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error running nvidia-smi")
            return None
        
        gpu_info = []
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 6:
                gpu_info.append({
                    'gpu_id': int(parts[0]),
                    'name': parts[1],
                    'memory_used': float(parts[2]) / 1024,  # Convert to GB
                    'memory_total': float(parts[3]) / 1024,  # Convert to GB
                    'memory_free': float(parts[4]) / 1024,   # Convert to GB
                    'gpu_util': float(parts[5])
                })
        
        return gpu_info
    
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def get_process_gpu_memory(pid):
    """Get GPU memory usage for a specific process"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            f'--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return {}
        
        process_gpu_memory = {}
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            if line:
                parts = line.split(', ')
                if len(parts) >= 2:
                    proc_pid = int(parts[0])
                    memory_mb = float(parts[1])
                    process_gpu_memory[proc_pid] = memory_mb / 1024  # Convert to GB
        
        return process_gpu_memory.get(pid, 0)
    
    except Exception as e:
        return 0

def get_process_info(pid):
    """Get CPU and RAM info for a process"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(interval=1),
            'memory_gb': process.memory_info().rss / 1024**3,
            'status': process.status()
        }
    except:
        return None

def find_training_processes():
    """Find all training processes"""
    training_pids = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline', []))
            if 'python' in proc.info['name'] and 'finetune.py' in cmdline:
                training_pids.append(proc.info['pid'])
        except:
            pass
    
    return training_pids

def monitor_training(log_file="gpu_memory_log.csv", interval=10):
    """Main monitoring function"""
    
    print("Starting GPU memory monitor...")
    print(f"Log file: {log_file}")
    print(f"Monitoring interval: {interval} seconds")
    
    # Create log file with headers
    with open(log_file, 'w') as f:
        f.write("timestamp,pid,cpu_percent,ram_gb,gpu_memory_gb,gpu_id,gpu_used_gb,gpu_total_gb,gpu_util\n")
    
    # Also create a simple text log
    text_log = log_file.replace('.csv', '.txt')
    
    while True:
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get GPU info
            gpu_info = get_gpu_memory_nvidia_smi()
            if not gpu_info:
                print("Failed to get GPU info")
                time.sleep(interval)
                continue
            
            # Find training processes
            training_pids = find_training_processes()
            
            if not training_pids:
                print(f"{timestamp} - No training processes found")
            else:
                # Get process GPU memory usage
                process_gpu_memory = {}
                for pid in training_pids:
                    process_gpu_memory[pid] = get_process_gpu_memory(pid)
                
                # Log data
                with open(log_file, 'a') as f, open(text_log, 'a') as f_text:
                    f_text.write(f"\n{'='*60}\n")
                    f_text.write(f"Timestamp: {timestamp}\n")
                    
                    for pid in training_pids:
                        proc_info = get_process_info(pid)
                        if proc_info:
                            gpu_mem = process_gpu_memory.get(pid, 0)
                            
                            # Write summary to text log
                            f_text.write(f"\nPID {pid}:\n")
                            f_text.write(f"  CPU: {proc_info['cpu_percent']:.1f}%\n")
                            f_text.write(f"  RAM: {proc_info['memory_gb']:.2f} GB\n")
                            f_text.write(f"  GPU Memory: {gpu_mem:.2f} GB\n")
                            
                            # Write CSV data
                            for gpu in gpu_info:
                                line = f"{timestamp},{pid},{proc_info['cpu_percent']:.1f},"
                                line += f"{proc_info['memory_gb']:.2f},{gpu_mem:.2f},"
                                line += f"{gpu['gpu_id']},{gpu['memory_used']:.2f},"
                                line += f"{gpu['memory_total']:.2f},{gpu['gpu_util']:.1f}\n"
                                f.write(line)
                            
                            # Check for warnings
                            if proc_info['memory_gb'] > 40:
                                warning = f"⚠️  HIGH RAM: {proc_info['memory_gb']:.2f} GB for PID {pid}"
                                print(warning)
                                f_text.write(f"  {warning}\n")
                    
                    # GPU summary
                    f_text.write(f"\nGPU Status:\n")
                    for gpu in gpu_info:
                        usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                        f_text.write(f"  GPU {gpu['gpu_id']} ({gpu['name']}): "
                                   f"{gpu['memory_used']:.2f}/{gpu['memory_total']:.2f} GB "
                                   f"({usage_percent:.1f}%), Util: {gpu['gpu_util']:.1f}%\n")
                        
                        if usage_percent > 95:
                            warning = f"⚠️  HIGH GPU MEMORY on GPU {gpu['gpu_id']}: {usage_percent:.1f}%"
                            print(warning)
                            f_text.write(f"  {warning}\n")
            
            # Print summary
            print(f"\n{timestamp} - Monitoring {len(training_pids)} processes")
            if gpu_info:
                for gpu in gpu_info:
                    print(f"  GPU {gpu['gpu_id']}: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB "
                          f"({(gpu['memory_used']/gpu['memory_total']*100):.0f}%)")
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
        
        time.sleep(interval)

def quick_gpu_check():
    """Quick one-time GPU memory check"""
    gpu_info = get_gpu_memory_nvidia_smi()
    
    if gpu_info:
        print("\nCurrent GPU Memory Status:")
        print("-" * 60)
        for gpu in gpu_info:
            usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"GPU {gpu['gpu_id']} ({gpu['name']}):")
            print(f"  Memory: {gpu['memory_used']:.2f}/{gpu['memory_total']:.2f} GB ({usage_percent:.1f}%)")
            print(f"  GPU Utilization: {gpu['gpu_util']:.1f}%")
            print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just do a quick check
        quick_gpu_check()
    else:
        # Start continuous monitoring
        try:
            monitor_training()
        except KeyboardInterrupt:
            print("\nMonitoring stopped")