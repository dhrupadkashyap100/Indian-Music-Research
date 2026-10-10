import os
import time
import numpy as np
import torch
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_file = os.path.join(BASE_DIR, "processed_data", "swaragrams", "swaragram_metadata.pkl")

def test_torch_load():
    with open(cache_file, "rb") as f:
        files, _, _ = pickle.load(f)
        
    print(f"Loading {len(files)} files into Shared Memory Tensor...", flush=True)
    t0 = time.time()
    
    # Pre-allocate shared tensor
    data_tensor = torch.empty((len(files), 1, 12, 938), dtype=torch.float32)
    data_tensor.share_memory_()
    
    for i, f in enumerate(files):
        data_tensor[i, 0] = torch.from_numpy(np.load(f))
        if (i+1) % 10000 == 0:
            print(f"Loaded {i+1} files in {time.time()-t0:.2f}s", flush=True)
            
    print(f"Total time: {time.time()-t0:.2f}s", flush=True)

if __name__ == "__main__":
    test_torch_load()
