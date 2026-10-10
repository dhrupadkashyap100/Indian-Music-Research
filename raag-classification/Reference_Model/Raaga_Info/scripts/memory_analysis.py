import os
import numpy as np

def analyze_memory():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    swara_root = os.path.join(base_dir, "processed_data", "swaragrams")
    
    total_files = 0
    total_size_bytes = 0
    sample_file = None
    
    for root, _, files in os.walk(swara_root):
        for f in files:
            if f.endswith('.npy'):
                total_files += 1
                fp = os.path.join(root, f)
                total_size_bytes += os.path.getsize(fp)
                if sample_file is None:
                    sample_file = fp
                    
    avg_file_size = total_size_bytes / total_files if total_files > 0 else 0
    
    ram_required = 0
    if sample_file:
        arr = np.load(sample_file)
        # assuming float32
        ram_required = total_files * arr.size * 4
        
    import subprocess
    total_ram = 0
    available_ram = 0
    try:
        # Get total physical memory in bytes
        cmd_total = "wmic ComputerSystem get TotalPhysicalMemory"
        result_total = subprocess.check_output(cmd_total, shell=True).decode().split('\n')
        total_ram = int(result_total[1].strip())
        
        # Get free physical memory in kilobytes
        cmd_free = "wmic OS get FreePhysicalMemory"
        result_free = subprocess.check_output(cmd_free, shell=True).decode().split('\n')
        available_ram = int(result_free[1].strip()) * 1024
    except Exception as e:
        print("Could not retrieve RAM info:", e)
        
    print(f"Total Swaragram Files: {total_files}")
    print(f"Total Dataset Size (Disk): {total_size_bytes / (1024**3):.2f} GB")
    print(f"Average File Size (Disk): {avg_file_size / 1024:.2f} KB")
    print(f"Estimated RAM Required (Float32): {ram_required / (1024**3):.2f} GB")
    print(f"Total System RAM: {total_ram / (1024**3):.2f} GB")
    print(f"Available System RAM: {available_ram / (1024**3):.2f} GB")

if __name__ == "__main__":
    analyze_memory()
