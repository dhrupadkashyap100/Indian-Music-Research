import time
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SWARA_ROOT = os.path.join(BASE_DIR, "processed_data", "swaragrams")

class FastSwaragramDataset(Dataset):
    def __init__(self):
        print("Scanning files...")
        self.files = []
        self.labels = []
        raga_folders = sorted([f for f in os.listdir(SWARA_ROOT) if os.path.isdir(os.path.join(SWARA_ROOT, f))])
        self.encoder = LabelEncoder()
        self.encoder.fit(raga_folders)
        for raga in raga_folders:
            folder = os.path.join(SWARA_ROOT, raga)
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    self.files.append(os.path.join(folder, file))
                    self.labels.append(raga)
        self.labels = self.encoder.transform(self.labels)
        
        print(f"Preloading {len(self.files)} files into RAM shared memory...")
        start_load = time.time()
        self.data_tensor = torch.empty((len(self.files), 1, 12, 938), dtype=torch.float32)
        self.label_tensor = torch.tensor(self.labels, dtype=torch.long)
        
        for i, f in enumerate(self.files):
            # Load and assign
            self.data_tensor[i, 0] = torch.from_numpy(np.load(f))
            if i % 10000 == 0 and i > 0:
                print(f"  Loaded {i} / {len(self.files)}...")
                
        self.data_tensor.share_memory_()
        self.label_tensor.share_memory_()
        print(f"Preloading finished in {time.time() - start_load:.2f}s")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]

def test_fast():
    dataset = FastSwaragramDataset()
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False, prefetch_factor=2)
    start_time = time.time()
    for i, (x, y) in enumerate(loader):
        if i == 0:
            print(f"First batch loaded in {time.time() - start_time:.4f}s")
        if i >= 10:
            break
    print(f"Completed 10 batches in {time.time() - start_time:.4f}s")

if __name__ == '__main__':
    test_fast()
