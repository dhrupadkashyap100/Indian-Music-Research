import os
import time
import numpy as np
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_file = os.path.join(BASE_DIR, "processed_data", "swaragrams", "swaragram_metadata.pkl")

def test_list_load():
    with open(cache_file, "rb") as f:
        files, _, _ = pickle.load(f)
        
    print(f"Loading {len(files)} files into Python List...", flush=True)
    t0 = time.time()
    
    data_cache = [None] * len(files)
    for i, f in enumerate(files):
        data_cache[i] = np.load(f)
        if (i+1) % 10000 == 0:
            print(f"Loaded {i+1} files in {time.time()-t0:.2f}s", flush=True)
            
    print(f"Total time: {time.time()-t0:.2f}s", flush=True)

if __name__ == "__main__":
    test_list_load()
