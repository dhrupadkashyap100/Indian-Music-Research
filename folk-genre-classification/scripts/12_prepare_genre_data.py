"""
=========================================================
Prepare Genre Dataset and DataLoaders
=========================================================

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap

Description
-----------
Loads mel_metadata.csv, validates required feature paths,
builds deterministic genre mappings, constructs GenreDataset
objects, and exposes PyTorch DataLoaders for CNN training.

Used by:
    - 13_train_genre_classifier.py
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
    METADATA_DIR,
    MODEL_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)
from utils.dataset import GenreDataset

# ==========================================================
# REQUIRED METADATA COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [
    "Feature Name",
    "Genre",
    "Dataset Split",
    "Mel Spectrogram Path",
]


# ==========================================================
# LOAD AND VALIDATE METADATA
# ==========================================================

def load_and_validate_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load mel_metadata.csv and verify that all required columns exist.

    Parameters
    ----------
    metadata_path : Path
        Absolute path to mel_metadata.csv.

    Returns
    ----------
    pd.DataFrame
        Loaded metadata DataFrame.

    Raises
    ----------
    FileNotFoundError
        If metadata_path does not exist on disk.
    ValueError
        If required columns are missing.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    df = pd.read_csv(metadata_path, low_memory=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {metadata_path.name}: {missing}"
        )

    return df


# ==========================================================
# BUILD AND SAVE GENRE MAPPING
# ==========================================================

def build_and_save_genre_mapping(
    output_path: Path,
    genres: list = GENRES
) -> Dict[str, int]:
    """
    Create a deterministic dictionary mapping genre strings to integer IDs
    and save it as genre_mapping.json.

    Parameters
    ----------
    output_path : Path
        Destination path for saving genre_mapping.json.
    genres : list
        Ordered list of genre names from config.py.

    Returns
    ----------
    Dict[str, int]
        Mapping from genre name to label index.
    """
    genre_mapping = {genre: idx for idx, genre in enumerate(genres)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(genre_mapping, f, indent=4)

    return genre_mapping


# ==========================================================
# PRINT DATASET SUMMARY
# ==========================================================

def print_dataset_summary(
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    num_genres: int,
    batch_size: int
) -> None:
    """
    Print a clean summary table of the prepared datasets.
    """
    print()
    print(LINE_SEPARATOR)
    print("Genre Dataset Ready")
    print(LINE_SEPARATOR)
    print(f"Train Samples      : {train_samples}")
    print(f"Validation Samples : {validation_samples}")
    print(f"Test Samples       : {test_samples}")
    print(f"Genres             : {num_genres}")
    print(f"Batch Size         : {batch_size}")
    print(LINE_SEPARATOR)


# ==========================================================
# CREATE DATALOADERS
# ==========================================================

def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load metadata, build genre mapping, instantiate GenreDatasets,
    and return configured PyTorch DataLoaders for all splits.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses for data loading.
    pin_memory : bool
        If True, tensors are copied into CUDA pinned memory before return.
    verbose : bool
        If True, prints dataset summary table.

    Returns
    ----------
    tuple
        (train_loader, validation_loader, test_loader, genre_mapping)
    """
    metadata_path = METADATA_DIR / MEL_METADATA_FILE
    mapping_path = MODEL_DIR / GENRE_MAPPING_FILE

    df = load_and_validate_metadata(metadata_path)
    genre_mapping = build_and_save_genre_mapping(mapping_path, GENRES)

    train_df = df[df["Dataset Split"] == "train"].copy()
    validation_df = df[df["Dataset Split"] == "validation"].copy()
    test_df = df[df["Dataset Split"] == "test"].copy()

    train_dataset = GenreDataset(train_df, genre_mapping)
    validation_dataset = GenreDataset(validation_df, genre_mapping)
    test_dataset = GenreDataset(test_df, genre_mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
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


# ==========================================================
# MAIN (Verification)
# ==========================================================

def main() -> None:
    """
    Execute data preparation and verify DataLoader outputs.
    """
    train_loader, val_loader, test_loader, genre_mapping = create_dataloaders()

    for batch_features, batch_labels, _ in train_loader:
        print()
        print(LINE_SEPARATOR)
        print("Verification Batch Details")
        print(LINE_SEPARATOR)
        print(f"Batch shape       : {tuple(batch_features.shape)}")
        print(f"Labels shape      : {tuple(batch_labels.shape)}")
        print(f"Number of classes : {len(genre_mapping)}")
        print(LINE_SEPARATOR)
        print()
        break


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

