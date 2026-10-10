import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# Path to NEW swaragram dataset
# =====================================================

SWARA_ROOT = str(BASE_DIR / "processed_data" / "swaragrams")


class SwaragramDataset(Dataset):

    def __init__(self, in_memory=False):

        print("=" * 60)
        print(f"Loading Swaragram Dataset (in_memory={in_memory})...")
        print("=" * 60)
        self.in_memory = in_memory

        import pickle
        cache_file = os.path.join(SWARA_ROOT, "swaragram_metadata.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.files, self.labels, classes_ = pickle.load(f)
            self.encoder = LabelEncoder()
            self.encoder.classes_ = classes_
        else:
            self.files = []
            self.labels = []
            raga_folders = sorted([
                folder
                for folder in os.listdir(SWARA_ROOT)
                if os.path.isdir(os.path.join(SWARA_ROOT, folder))
            ])

            self.encoder = LabelEncoder()
            self.encoder.fit(raga_folders)

            for raga in raga_folders:
                folder = os.path.join(SWARA_ROOT, raga)
                for file in os.listdir(folder):
                    if file.endswith(".npy"):
                        self.files.append(
                            os.path.join(folder, file)
                        )
                        self.labels.append(raga)

            self.labels = self.encoder.transform(self.labels)
            
            with open(cache_file, "wb") as f:
                pickle.dump((self.files, self.labels, self.encoder.classes_), f)
                
        # Optional RAM Cache Initialization
        if self.in_memory:
            import time
            print(f"Preloading {len(self.files)} samples into RAM...", flush=True)
            t0 = time.time()
            self.ram_cache = [None] * len(self.files)
            for i, f in enumerate(self.files):
                # Load exactly once and convert to tensor
                swara = np.load(f)
                swara = torch.from_numpy(swara).float().unsqueeze(0)
                label = torch.tensor(self.labels[i], dtype=torch.long)
                self.ram_cache[i] = (swara, label)
            print(f"RAM Cache populated in {time.time()-t0:.2f}s", flush=True)

        print()
        print("Dataset Loaded Successfully")
        print()
        print("Number of Raagas :", len(self.encoder.classes_))
        print("Total Samples    :", len(self.files))
        print("=" * 60)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.ram_cache[idx]
            
        # Optimize loading: use mmap_mode and use torch.from_numpy to avoid copy
        swara = np.load(self.files[idx], mmap_mode='r')
        swara = torch.from_numpy(swara).float()
        
        # Shape -> (1,12,938)
        swara = swara.unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return swara, label
