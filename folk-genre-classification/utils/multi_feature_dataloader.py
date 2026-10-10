"""
=========================================================
Prepare Multi-Feature Dataset and DataLoaders
=========================================================

Project:
AI-Based Analysis of Indian Traditional Folk Music

Description
-----------
Loads mel_metadata.csv and chroma_metadata.csv,
builds deterministic genre mappings, constructs MultiFeatureDataset
objects, and exposes PyTorch DataLoaders.
=========================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import (
    BATCH_SIZE,
    GENRE_MAPPING_FILE,
    GENRES,
    LINE_SEPARATOR,
    MEL_METADATA_FILE,
    CHROMA_METADATA_FILE,
    METADATA_DIR,
    MODEL_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)
from utils.multi_feature_dataset import MultiFeatureDataset

REQUIRED_COLUMNS = [
    "Feature Name",
    "Genre",
    "Dataset Split"
]

def load_and_validate_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found:\n{metadata_path}")

    df = pd.read_csv(metadata_path, low_memory=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {metadata_path.name}: {missing}")

    return df

def build_and_save_genre_mapping(output_path: Path, genres: list = GENRES) -> Dict[str, int]:
    genre_mapping = {genre: idx for idx, genre in enumerate(genres)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(genre_mapping, f, indent=4)
    return genre_mapping

def print_dataset_summary(
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_genres: int,
    batch_size: int
) -> None:
    print()
    print(LINE_SEPARATOR)
    print("Multi-Feature Dataset Ready")
    print(LINE_SEPARATOR)
    print(f"Train Samples      : {train_samples}")
    print(f"Validation Samples : {validation_samples}")
    print(f"Test Samples       : {test_samples}")
    print(f"Genres             : {num_genres}")
    print(f"Batch Size         : {batch_size}")
    print(LINE_SEPARATOR)


def create_multi_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    enable_specaugment: bool = False,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    
    mel_metadata_path = METADATA_DIR / MEL_METADATA_FILE
    chroma_metadata_path = METADATA_DIR / CHROMA_METADATA_FILE
    mapping_path = MODEL_DIR / GENRE_MAPPING_FILE

    mel_df = load_and_validate_metadata(mel_metadata_path)
    chroma_df = load_and_validate_metadata(chroma_metadata_path)
    
    genre_mapping = build_and_save_genre_mapping(mapping_path, GENRES)

    # Filter splits for Mel
    train_mel_df = mel_df[mel_df["Dataset Split"] == "train"].copy()
    validation_mel_df = mel_df[mel_df["Dataset Split"] == "validation"].copy()
    test_mel_df = mel_df[mel_df["Dataset Split"] == "test"].copy()

    # Filter splits for Chroma
    train_chroma_df = chroma_df[chroma_df["Dataset Split"] == "train"].copy()
    validation_chroma_df = chroma_df[chroma_df["Dataset Split"] == "validation"].copy()
    test_chroma_df = chroma_df[chroma_df["Dataset Split"] == "test"].copy()

    train_dataset = MultiFeatureDataset(train_mel_df, train_chroma_df, genre_mapping, enable_specaugment=enable_specaugment)
    validation_dataset = MultiFeatureDataset(validation_mel_df, validation_chroma_df, genre_mapping)
    test_dataset = MultiFeatureDataset(test_mel_df, test_chroma_df, genre_mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    if verbose:
        print_dataset_summary(
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
            len(genre_mapping),
            batch_size
        )

    return train_loader, validation_loader, test_loader, genre_mapping
