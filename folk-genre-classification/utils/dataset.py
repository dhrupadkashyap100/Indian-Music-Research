"""
=========================================================
PyTorch Genre Dataset

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap

Description
-----------
Loads Mel Spectrogram (.npy) features using
mel_metadata.csv and prepares them for CNN training.

Used by:
    - 12_prepare_genre_data.py
    - 13_train_genre_classifier.py
    - 14_evaluate_genre_model.py
    - 15_predict_genre.py
=========================================================
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.features_utils import (
    load_feature,
    validate_feature
)

# ==========================================================
# METADATA COLUMN NAMES
# ==========================================================

FEATURE_NAME_COLUMN = "Feature Name"

FEATURE_PATH_COLUMN = "Mel Spectrogram Path"

GENRE_COLUMN = "Genre"

SPLIT_COLUMN = "Dataset Split"

# ==========================================================
# DATASET CLASS
# ==========================================================

class GenreDataset(Dataset):
    """
    PyTorch Dataset for Folk Genre Classification.

    Each sample returns:

        feature_tensor,
        label_tensor,
        feature_name
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self,
        dataframe: pd.DataFrame,
        genre_mapping: dict,
        feature_path_column: str = FEATURE_PATH_COLUMN
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.genre_mapping = genre_mapping

        self.feature_path_column = feature_path_column

        self.feature_name_column = FEATURE_NAME_COLUMN

        self.genre_column = GENRE_COLUMN

    # ======================================================
    # TOTAL SAMPLES
    # ======================================================

    def __len__(self):

        return len(self.dataframe)

    # ======================================================
    # LOAD FEATURE
    # ======================================================

    def _load_feature(
        self,
        feature_path: str
    ):

        feature_path = Path(feature_path)

        if not feature_path.exists():

            raise FileNotFoundError(

                f"Feature file not found:\n{feature_path}"

            )

        feature = load_feature(feature_path)

        return feature

    # ======================================================
    # VALIDATE FEATURE
    # ======================================================

    def _validate_feature(
        self,
        feature,
        feature_name
    ):

        if not validate_feature(feature):

            raise ValueError(

                f"Invalid feature detected:\n{feature_name}"

            )

        return True

        # ======================================================
    # ENCODE LABEL
    # ======================================================

    def _encode_label(
        self,
        genre: str
    ):

        if genre not in self.genre_mapping:

            raise ValueError(

                f"Unknown Genre: {genre}"

            )

        return self.genre_mapping[genre]

    # ======================================================
    # PREPARE TENSOR
    # ======================================================

    def _prepare_tensor(
        self,
        feature
    ):

        feature_tensor = torch.tensor(
            feature,
            dtype=torch.float32
        )

        # CNN expects:
        # (Channels, Height, Width)
        # (1, 128, 938)

        feature_tensor = feature_tensor.unsqueeze(0)

        return feature_tensor

    # ======================================================
    # GET SINGLE SAMPLE
    # ======================================================

    def __getitem__(
        self,
        index
    ):

        row = self.dataframe.iloc[index]

        feature_name = row[self.feature_name_column]

        feature_path = row[self.feature_path_column]

        genre = row[self.genre_column]

        # ---------------------------------------------
        # Load Feature
        # ---------------------------------------------

        feature = self._load_feature(
            feature_path
        )

        # ---------------------------------------------
        # Validate Feature
        # ---------------------------------------------

        self._validate_feature(
            feature,
            feature_name
        )

        # ---------------------------------------------
        # Convert to Tensor
        # ---------------------------------------------

        feature_tensor = self._prepare_tensor(
            feature
        )

        # ---------------------------------------------
        # Encode Label
        # ---------------------------------------------

        label = self._encode_label(
            genre
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        # ---------------------------------------------
        # Return Sample
        # ---------------------------------------------

        return (
            feature_tensor,
            label_tensor,
            feature_name
        )

        # ======================================================
# PRINT DATASET SUMMARY
# ======================================================

def print_dataset_summary(dataset):

    print()

    print("=" * 60)
    print("Genre Dataset Summary")
    print("=" * 60)

    print(f"Total Samples : {len(dataset)}")

    print(f"Genres        : {len(dataset.genre_mapping)}")

    print()

    print("Genre Mapping")

    print("-" * 60)

    for genre, label in dataset.genre_mapping.items():

        print(f"{label:>2}  ->  {genre}")

    print("=" * 60)


# ======================================================
# VERIFY DATASET
# ======================================================

def verify_dataset(dataset, samples=5):

    print()

    print("=" * 60)
    print("Verifying Dataset")
    print("=" * 60)

    total = min(samples, len(dataset))

    for index in range(total):

        feature_tensor, label_tensor, feature_name = dataset[index]

        print()

        print(f"Sample : {index + 1}")

        print(f"Feature Name : {feature_name}")

        print(f"Tensor Shape : {tuple(feature_tensor.shape)}")

        print(f"Tensor Type  : {feature_tensor.dtype}")

        print(f"Label        : {label_tensor.item()}")

    print()

    print("=" * 60)
    print("Dataset Verification Successful")
    print("=" * 60)


# ======================================================
# MAIN (Testing)
# ======================================================

if __name__ == "__main__":

    print()

    print("=" * 60)
    print("GenreDataset Utility")
    print("=" * 60)

    print()

    print(
        "This module is intended to be imported by "
        "12_prepare_genre_data.py"
    )

    print()

    print(
        "Example:"
    )

    print()

    print(
        "from utils.dataset import GenreDataset"
    )

    print()

    print("=" * 60)