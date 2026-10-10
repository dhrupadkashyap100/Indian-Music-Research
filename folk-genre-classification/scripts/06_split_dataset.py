"""
=========================================================
Split Dataset

Performs song-level train / validation / test split.

Author : Dhrupad Kashyap
=========================================================
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from utils.config import (
    METADATA_DIR,
    GENRES,
    LINE_SEPARATOR
)

# ==========================================================
# FILE PATHS
# ==========================================================

MASTER_DATASET = METADATA_DIR / "master_dataset.csv"
SONG_GROUPS = METADATA_DIR / "song_groups.csv"

TRAIN_CSV = METADATA_DIR / "train.csv"
VALIDATION_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"


# ==========================================================
# LOAD DATASETS
# ==========================================================

def load_data():

    if not MASTER_DATASET.exists():
        raise FileNotFoundError(MASTER_DATASET)

    if not SONG_GROUPS.exists():
        raise FileNotFoundError(SONG_GROUPS)

    master_df = pd.read_csv(MASTER_DATASET)

    song_groups_df = pd.read_csv(SONG_GROUPS)

    return master_df, song_groups_df


# ==========================================================
# SPLIT A SINGLE GENRE
# ==========================================================

def split_genre(song_df):

    train_df, temp_df = train_test_split(
        song_df,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    return train_df, validation_df, test_df


# ==========================================================
# CREATE SONG-LEVEL SPLITS
# ==========================================================

def create_song_splits(song_groups_df):

    train_list = []
    validation_list = []
    test_list = []

    print()
    print(LINE_SEPARATOR)
    print("Creating Song-Level Dataset Split")
    print(LINE_SEPARATOR)

    for genre in GENRES:

        genre_df = song_groups_df[
            song_groups_df["Genre"] == genre
        ]

        train_df, validation_df, test_df = split_genre(
            genre_df
        )

        train_list.append(train_df)
        validation_list.append(validation_df)
        test_list.append(test_df)

        print(
            f"{genre:<15}"
            f" Train:{len(train_df):<3}"
            f" Validation:{len(validation_df):<3}"
            f" Test:{len(test_df):<3}"
        )

    print()

    train_songs = pd.concat(
        train_list,
        ignore_index=True
    )

    validation_songs = pd.concat(
        validation_list,
        ignore_index=True
    )

    test_songs = pd.concat(
        test_list,
        ignore_index=True
    )

    return (
        train_songs,
        validation_songs,
        test_songs
    )

    # ==========================================================
# MAP SONG SPLITS TO SEGMENTS
# ==========================================================

def map_segments(master_df, song_split_df):

    song_ids = song_split_df["Original Song ID"].unique()

    subset = master_df[
        master_df["Original Song ID"].isin(song_ids)
    ].copy()

    return subset


# ==========================================================
# SAVE DATASETS
# ==========================================================

def save_datasets(train_df, validation_df, test_df):

    train_df.to_csv(TRAIN_CSV, index=False)
    validation_df.to_csv(VALIDATION_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print()
    print(LINE_SEPARATOR)
    print("Datasets Saved Successfully")
    print(LINE_SEPARATOR)
    print(f"Train      : {TRAIN_CSV}")
    print(f"Validation : {VALIDATION_CSV}")
    print(f"Test       : {TEST_CSV}")


# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report(train_df, validation_df, test_df):

    report_path = METADATA_DIR / "dataset_split_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("DATASET SPLIT REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Training Segments   : {len(train_df)}\n")
        f.write(f"Validation Segments : {len(validation_df)}\n")
        f.write(f"Testing Segments    : {len(test_df)}\n\n")

        f.write(f"Training Songs      : {train_df['Original Song ID'].nunique()}\n")
        f.write(f"Validation Songs    : {validation_df['Original Song ID'].nunique()}\n")
        f.write(f"Testing Songs       : {test_df['Original Song ID'].nunique()}\n")


# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_summary(train_df, validation_df, test_df):

    print()
    print(LINE_SEPARATOR)
    print("DATASET SPLIT SUMMARY")
    print(LINE_SEPARATOR)

    print(f"Training Segments   : {len(train_df)}")
    print(f"Validation Segments : {len(validation_df)}")
    print(f"Testing Segments    : {len(test_df)}")

    print()

    print(f"Training Songs      : {train_df['Original Song ID'].nunique()}")
    print(f"Validation Songs    : {validation_df['Original Song ID'].nunique()}")
    print(f"Testing Songs       : {test_df['Original Song ID'].nunique()}")

    print()
    print("Genre-wise Song Distribution")
    print("-" * 60)

    for genre in GENRES:

        train_count = train_df[
            train_df["Genre"] == genre
        ]["Original Song ID"].nunique()

        validation_count = validation_df[
            validation_df["Genre"] == genre
        ]["Original Song ID"].nunique()

        test_count = test_df[
            test_df["Genre"] == genre
        ]["Original Song ID"].nunique()

        print(
            f"{genre:<15}"
            f"Train:{train_count:<3}"
            f" Val:{validation_count:<3}"
            f" Test:{test_count:<3}"
        )

    print()
    print(LINE_SEPARATOR)


# ==========================================================
# MAIN
# ==========================================================

def main():

    master_df, song_groups_df = load_data()

    train_songs, validation_songs, test_songs = create_song_splits(
        song_groups_df
    )

    train_df = map_segments(master_df, train_songs)

    validation_df = map_segments(master_df, validation_songs)

    test_df = map_segments(master_df, test_songs)

    save_datasets(
        train_df,
        validation_df,
        test_df
    )

    generate_report(
        train_df,
        validation_df,
        test_df
    )

    print_summary(
        train_df,
        validation_df,
        test_df
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()