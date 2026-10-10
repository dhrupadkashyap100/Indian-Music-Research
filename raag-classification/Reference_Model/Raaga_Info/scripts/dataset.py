import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# =====================================================
# Path to NEW chromagram dataset
# =====================================================

CHROMA_ROOT = str(BASE_DIR / "processed_data" / "chromagrams")


class SaragaDataset(Dataset):

    def __init__(self):

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        self.files = []
        self.labels = []

        # -----------------------------
        # Scan all raga folders
        # -----------------------------

        raga_folders = sorted([
            folder
            for folder in os.listdir(CHROMA_ROOT)
            if os.path.isdir(os.path.join(CHROMA_ROOT, folder))
        ])

        self.encoder = LabelEncoder()
        self.encoder.fit(raga_folders)

        for raga in raga_folders:

            folder = os.path.join(CHROMA_ROOT, raga)

            for file in os.listdir(folder):

                if file.endswith(".npy"):

                    self.files.append(
                        os.path.join(folder, file)
                    )

                    self.labels.append(raga)

        self.labels = self.encoder.transform(self.labels)

        print()
        print("Dataset Loaded Successfully")
        print()
        print("Number of Raagas :", len(self.encoder.classes_))
        print("Total Samples    :", len(self.files))
        print("=" * 60)

    # =====================================================

    def __len__(self):
        return len(self.files)

    # =====================================================

    def __getitem__(self, idx):

        chroma = np.load(self.files[idx])

        chroma = torch.tensor(
            chroma,
            dtype=torch.float32
        )

        # Shape -> (1,12,938)
        chroma = chroma.unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return chroma, label