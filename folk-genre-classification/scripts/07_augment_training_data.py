"""
=========================================================
Augment Training Dataset

Creates two augmented versions of every
training audio segment.

Author : Dhrupad Kashyap
=========================================================
"""

from pathlib import Path
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from utils.config import (
    DATASET_DIR,
    METADATA_DIR,
    GENRES,
    SAMPLE_RATE,
    LINE_SEPARATOR
)

# ==========================================================
# PATHS
# ==========================================================

TRAIN_CSV = METADATA_DIR / "train.csv"

AUGMENTED_DIR = DATASET_DIR / "augmented"

AUGMENTED_METADATA = (
    METADATA_DIR /
    "augmented_train.csv"
)

# ==========================================================
# AUGMENTATION FUNCTIONS
# ==========================================================

def add_noise(audio):

    noise = np.random.normal(
        0,
        0.003,
        len(audio)
    )

    return audio + noise


def pitch_shift(audio):

    steps = random.choice(
        [-2, -1, 1, 2]
    )

    return librosa.effects.pitch_shift(
        audio,
        sr=SAMPLE_RATE,
        n_steps=steps
    )


def time_stretch(audio):

    rate = random.uniform(
        0.90,
        1.10
    )

    return librosa.effects.time_stretch(
        audio,
        rate=rate
    )


def gain(audio):

    factor = random.uniform(
        0.8,
        1.2
    )

    return audio * factor


AUGMENTATIONS = {

    "noise": add_noise,

    "pitch": pitch_shift,

    "stretch": time_stretch,

    "gain": gain

}

# ==========================================================
# LOAD TRAIN DATA
# ==========================================================

def load_training_dataset():

    print()
    print(LINE_SEPARATOR)
    print("Loading Training Dataset")
    print(LINE_SEPARATOR)

    df = pd.read_csv(TRAIN_CSV)

    print(f"Training Segments : {len(df)}")

    return df

    # ==========================================================
# AUGMENT TRAINING DATA
# ==========================================================

def augment_dataset(df):

    augmented_rows = []

    print()
    print(LINE_SEPARATOR)
    print("Augmenting Training Dataset")
    print(LINE_SEPARATOR)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        audio_path = Path(
            row["Segment Path"]
        )

        genre = row["Genre"]

        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE
        )

        genre_folder = (
            AUGMENTED_DIR /
            genre
        )

        genre_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Choose two different augmentations
        aug_names = random.sample(
            list(AUGMENTATIONS.keys()),
            2
        )

        for aug_name in aug_names:

            augmented_audio = AUGMENTATIONS[
                aug_name
            ](y)

            output_name = (
                audio_path.stem +
                "_" +
                aug_name +
                ".wav"
            )

            output_path = (
                genre_folder /
                output_name
            )

            sf.write(
                output_path,
                augmented_audio,
                SAMPLE_RATE
            )

            new_row = row.copy()

            new_row["Segment Path"] = str(
                output_path
            )

            new_row["Augmentation"] = aug_name

            augmented_rows.append(
                new_row
            )

    augmented_df = pd.DataFrame(
        augmented_rows
    )

    return augmented_df


# ==========================================================
# SAVE METADATA
# ==========================================================

def save_metadata(original_df, augmented_df):

    final_df = pd.concat(
        [
            original_df,
            augmented_df
        ],
        ignore_index=True
    )

    final_df.to_csv(
        AUGMENTED_METADATA,
        index=False
    )

    print()
    print(LINE_SEPARATOR)
    print("Augmented Dataset Saved")
    print(LINE_SEPARATOR)

    print(AUGMENTED_METADATA)

    print()
    print(
        f"Original Segments  : {len(original_df)}"
    )

    print(
        f"Augmented Segments : {len(augmented_df)}"
    )

    print(
        f"Final Training Set : {len(final_df)}"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    train_df = load_training_dataset()

    augmented_df = augment_dataset(
        train_df
    )

    save_metadata(
        train_df,
        augmented_df
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()