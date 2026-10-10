"""
=========================================================
Multi-Feature Genre Dataset

Project:
AI-Based Analysis of Indian Traditional Folk Music

Description
-----------
Loads BOTH Mel Spectrogram and CQT Chromagram (.npy) features
using their respective metadata files, and strictly verifies
pairing.

Used by:
    - Multi-Feature Research Pipeline
=========================================================
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.features_utils import load_feature, validate_feature

# ==========================================================
# METADATA COLUMN NAMES
# ==========================================================

FEATURE_NAME_COLUMN = "Feature Name"
MEL_PATH_COLUMN = "Mel Spectrogram Path"
CHROMA_PATH_COLUMN = "Chromagram Path"
GENRE_COLUMN = "Genre"
SPLIT_COLUMN = "Dataset Split"

# ==========================================================
# DATASET CLASS
# ==========================================================

class MultiFeatureDataset(Dataset):
    """
    PyTorch Dataset for Multi-Feature Folk Genre Classification.

    Each sample returns:
        mel_tensor,
        chroma_tensor,
        label_tensor,
        filename
    """

    def __init__(
        self,
        mel_dataframe: pd.DataFrame,
        chroma_dataframe: pd.DataFrame,
        genre_mapping: dict,
        enable_specaugment: bool = False
    ):
        
        self.mel_dataframe = mel_dataframe.reset_index(drop=True)
        self.chroma_dataframe = chroma_dataframe.reset_index(drop=True)
        self.genre_mapping = genre_mapping
        self.enable_specaugment = enable_specaugment
        
        if self.enable_specaugment:
            from utils.config import (
                TIME_MASK_PROBABILITY, TIME_MASK_WIDTH, NUM_TIME_MASKS,
                FREQUENCY_MASK_PROBABILITY, FREQUENCY_MASK_WIDTH, NUM_FREQUENCY_MASKS
            )
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_WIDTH)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQUENCY_MASK_WIDTH)
            self.t_prob = TIME_MASK_PROBABILITY
            self.f_prob = FREQUENCY_MASK_PROBABILITY
            self.n_t_masks = NUM_TIME_MASKS
            self.n_f_masks = NUM_FREQUENCY_MASKS

        # Integrity Checks
        if len(self.mel_dataframe) != len(self.chroma_dataframe):
            raise ValueError(
                f"Dataset length mismatch! Mel: {len(self.mel_dataframe)}, "
                f"Chroma: {len(self.chroma_dataframe)}"
            )

        print("Verifying Multi-Feature Dataset Integrity...")
        
        # Verify strict alignment
        for i in range(len(self.mel_dataframe)):
            mel_row = self.mel_dataframe.iloc[i]
            chroma_row = self.chroma_dataframe.iloc[i]

            if mel_row[GENRE_COLUMN] != chroma_row[GENRE_COLUMN]:
                raise ValueError(f"Genre mismatch at index {i}: {mel_row[GENRE_COLUMN]} vs {chroma_row[GENRE_COLUMN]}")
            
            if mel_row[SPLIT_COLUMN] != chroma_row[SPLIT_COLUMN]:
                raise ValueError(f"Split mismatch at index {i}: {mel_row[SPLIT_COLUMN]} vs {chroma_row[SPLIT_COLUMN]}")
                
            mel_path = Path(str(mel_row[MEL_PATH_COLUMN]))
            chroma_path = Path(str(chroma_row[CHROMA_PATH_COLUMN]))
            
            if mel_path.name != chroma_path.name:
                raise ValueError(f"Filename mismatch at index {i}: {mel_path.name} vs {chroma_path.name}")

        print("Integrity check passed! Datasets are perfectly aligned.")

    def __len__(self):
        return len(self.mel_dataframe)

    def _load_and_validate(self, path: Path, expected_shape: tuple, feature_type: str):
        if not path.exists():
            raise FileNotFoundError(f"{feature_type} file not found: {path}")

        feature = load_feature(path)
        
        if not validate_feature(feature):
            raise ValueError(f"Invalid {feature_type} feature detected: {path}")

        if feature.shape != expected_shape:
            raise ValueError(f"{feature_type} shape mismatch: Expected {expected_shape}, got {feature.shape} at {path}")
            
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        return feature_tensor

    def __getitem__(self, index):
        mel_row = self.mel_dataframe.iloc[index]
        chroma_row = self.chroma_dataframe.iloc[index]

        mel_path = Path(str(mel_row[MEL_PATH_COLUMN]))
        chroma_path = Path(str(chroma_row[CHROMA_PATH_COLUMN]))
        
        genre = mel_row[GENRE_COLUMN]
        filename = mel_path.name

        # Load and Validate
        mel_tensor = self._load_and_validate(mel_path, (128, 938), "Mel Spectrogram")
        chroma_tensor = self._load_and_validate(chroma_path, (12, 938), "Chromagram")
        
        # SpecAugment (Applied only to Mel Spectrogram)
        if self.enable_specaugment:
            if torch.rand(1).item() < self.f_prob:
                for _ in range(self.n_f_masks):
                    mel_tensor = self.freq_masking(mel_tensor)
            if torch.rand(1).item() < self.t_prob:
                for _ in range(self.n_t_masks):
                    mel_tensor = self.time_masking(mel_tensor)

        # Encode Label
        if genre not in self.genre_mapping:
            raise ValueError(f"Unknown Genre: {genre}")
            
        label = self.genre_mapping[genre]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return (
            mel_tensor,
            chroma_tensor,
            label_tensor,
            filename
        )
