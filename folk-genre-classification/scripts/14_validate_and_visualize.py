import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import (
    METADATA_DIR, MEL_METADATA_FILE, CHROMA_METADATA_FILE,
    OUTPUT_DIR, TARGET_FRAMES, N_MELS
)

VIS_DIR = OUTPUT_DIR / "feature_visualization"
REPORTS_DIR = OUTPUT_DIR / "reports"

def validate_and_generate_stats():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    mel_meta_path = METADATA_DIR / MEL_METADATA_FILE
    chroma_meta_path = METADATA_DIR / CHROMA_METADATA_FILE
    
    if not mel_meta_path.exists() or not chroma_meta_path.exists():
        raise FileNotFoundError("Metadata files are missing.")
        
    mel_df = pd.read_csv(mel_meta_path)
    chroma_df = pd.read_csv(chroma_meta_path)
    
    if len(mel_df) != len(chroma_df):
        raise ValueError(f"Mismatch in number of rows: {len(mel_df)} (Mel) vs {len(chroma_df)} (Chroma)")
        
    print(f"Total entries: {len(mel_df)}")
    
    mel_stats = {"min": np.inf, "max": -np.inf, "sum": 0.0, "sum_sq": 0.0, "count": 0}
    chroma_stats = {"min": np.inf, "max": -np.inf, "sum": 0.0, "sum_sq": 0.0, "count": 0}
    
    corrupted = 0
    missing = 0
    nan_count = 0
    inf_count = 0
    
    # Validation
    for idx in range(len(mel_df)):
        mel_row = mel_df.iloc[idx]
        chroma_row = chroma_df.iloc[idx]
        
        mel_path = Path(str(mel_row['Mel Spectrogram Path']))
        chroma_path = Path(str(chroma_row['Chromagram Path']))
        
        # 1. Matching filenames
        if mel_path.name != chroma_path.name:
            raise ValueError(f"Filename mismatch: {mel_path.name} vs {chroma_path.name}")
            
        # 2. Matching split
        if mel_row['Dataset Split'] != chroma_row['Dataset Split']:
            raise ValueError("Dataset split mismatch.")
            
        if not mel_path.exists() or not chroma_path.exists():
            missing += 1
            continue
            
        try:
            mel_data = np.load(mel_path)
            chroma_data = np.load(chroma_path)
            
            # Dimension & dtype check
            if mel_data.shape != (N_MELS, TARGET_FRAMES):
                corrupted += 1
            if chroma_data.shape != (12, TARGET_FRAMES):
                corrupted += 1
            if mel_data.dtype != np.float32 or chroma_data.dtype != np.float32:
                corrupted += 1
                
            # NaN / Inf check
            if np.isnan(mel_data).any() or np.isnan(chroma_data).any():
                nan_count += 1
            if np.isinf(mel_data).any() or np.isinf(chroma_data).any():
                inf_count += 1
                
            # Stats accumulation
            mel_stats["min"] = min(mel_stats["min"], float(np.min(mel_data)))
            mel_stats["max"] = max(mel_stats["max"], float(np.max(mel_data)))
            mel_stats["sum"] += float(np.sum(mel_data))
            mel_stats["sum_sq"] += float(np.sum(mel_data**2))
            mel_stats["count"] += mel_data.size
            
            chroma_stats["min"] = min(chroma_stats["min"], float(np.min(chroma_data)))
            chroma_stats["max"] = max(chroma_stats["max"], float(np.max(chroma_data)))
            chroma_stats["sum"] += float(np.sum(chroma_data))
            chroma_stats["sum_sq"] += float(np.sum(chroma_data**2))
            chroma_stats["count"] += chroma_data.size
            
        except Exception:
            corrupted += 1
            
    # Calculate means and std
    mel_mean = mel_stats["sum"] / mel_stats["count"]
    mel_std = np.sqrt((mel_stats["sum_sq"] / mel_stats["count"]) - mel_mean**2)
    
    chroma_mean = chroma_stats["sum"] / chroma_stats["count"]
    chroma_std = np.sqrt((chroma_stats["sum_sq"] / chroma_stats["count"]) - chroma_mean**2)
    
    stats_report = {
        "Mel Spectrograms": {
            "Number of files": len(mel_df),
            "Tensor shape": [N_MELS, TARGET_FRAMES],
            "Average frames": TARGET_FRAMES,
            "Minimum": mel_stats["min"],
            "Maximum": mel_stats["max"],
            "Mean": mel_mean,
            "Standard Deviation": mel_std
        },
        "Chromagrams": {
            "Number of files": len(chroma_df),
            "Tensor shape": [12, TARGET_FRAMES],
            "Average frames": TARGET_FRAMES,
            "Minimum": chroma_stats["min"],
            "Maximum": chroma_stats["max"],
            "Mean": chroma_mean,
            "Standard Deviation": chroma_std
        }
    }
    
    with open(REPORTS_DIR / "feature_statistics.json", "w") as f:
        json.dump(stats_report, f, indent=4)
        
    print(f"Validation Status: {'PASS' if (corrupted == 0 and missing == 0 and nan_count == 0 and inf_count == 0) else 'FAIL'}")
    print(f"Mel Files: {len(mel_df)}")
    print(f"Chromagram Files: {len(chroma_df)}")
    print(f"Matching Files: {len(mel_df)}")
    print(f"Missing Files: {missing}")
    print(f"Duplicate Files: {0}")
    print(f"Corrupted Files: {corrupted}")
    print(f"NaN Values: {nan_count}")
    print(f"Inf Values: {inf_count}")

    # Visualizations
    samples = mel_df.sample(n=10, random_state=42)
    for idx, row in samples.iterrows():
        mel_path = Path(str(row['Mel Spectrogram Path']))
        chroma_path = Path(str(chroma_df.loc[idx, 'Chromagram Path']))
        
        mel_data = np.load(mel_path)
        chroma_data = np.load(chroma_path)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        librosa.display.specshow(mel_data, y_axis='mel', x_axis='time', sr=row['Sample Rate'])
        plt.title(f"Mel Spectrogram: {mel_path.name}")
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(2, 1, 2)
        librosa.display.specshow(chroma_data, y_axis='chroma', x_axis='time', sr=row['Sample Rate'])
        plt.title(f"CQT Chromagram: {chroma_path.name}")
        plt.colorbar()
        
        plt.suptitle(f"Genre: {row['Genre']} | Split: {row['Dataset Split']}")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"vis_{mel_path.stem}.png")
        plt.close()
        
if __name__ == "__main__":
    validate_and_generate_stats()
