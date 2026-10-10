import os
import time
import numpy as np
import torch
import concurrent.futures
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_file = os.path.join(BASE_DIR, "processed_data", "swaragrams", "swaragram_metadata.pkl")

def load_file(args):
    idx, f = args
    return idx, np.load(f)

def test_preload():
    with open(cache_file, "rb") as f:
        files, _, _ = pickle.load(f)
        
    print(f"Loaded metadata. Found {len(files)} files.")
    
    start = time.time()
    
    # 1. Sequential load into pre-allocated numpy array
    print("Testing sequential load into np.empty...")
    data_array = np.empty((len(files), 1, 12, 938), dtype=np.float32)
    t0 = time.time()
    for i, f in enumerate(files):
        data_array[i, 0] = np.load(f)
    seq_time = time.time() - t0
    print(f"Sequential load took: {seq_time:.2f}s")
    
    # 2. ThreadPoolExecutor load
    print("Testing concurrent load...")
    data_array_2 = np.empty((len(files), 1, 12, 938), dtype=np.float32)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, arr in executor.map(load_file, enumerate(files)):
            data_array_2[idx, 0] = arr
    conc_time = time.time() - t0
    print(f"Concurrent load took: {conc_time:.2f}s")

if __name__ == "__main__":
    test_preload()
