"""
=========================================================
04_build_master_dataset.py

Build Master Dataset

Combines the generated segment_metadata.csv with the
per-genre Excel metadata files to produce one unified
master dataset used by all downstream scripts.

Author : Dhrupad Kashyap

Root Causes Fixed (2026-07-22)
-------------------------------
1. Garba double-extension filenames:
   02_standardize_audio.py wrote Garba files as
   GARBA_001.wav.wav because Processed_File_Name already
   contained .wav.  Path.stem therefore yielded
   GARBA_001.wav instead of GARBA_001, causing all 2593
   Garba segments to fail the merge.
   Fix: normalise Original Song ID in load_segment_metadata
   by stripping any trailing .wav before the merge.

2. YouTube column name fragmentation:
   Each genre xlsx uses a different column name:
     Baul      -> Source
     Bhangra / Bihu / Garba / Maand -> YouTube Link
     Dhamail / Ghoomar / Jhumur / Lavani / Sambalpuri
               -> Youtube Link  (lower-case y)
   After pd.concat these survive as three separate columns
   all renamed to YouTube Link producing a degenerate
   DataFrame with three columns of the same name.
   Fix: normalise every per-genre DataFrame individually
   inside load_genre_metadata before concatenation so each
   sheet already has exactly one YouTube Link column.

3. validate_master_dataset referenced a hardcoded
   YouTube Link column that was absent for most genres.
   Fix: validate against the canonicalised column.
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.config import (
    GENRES,
    LINE_SEPARATOR,
    METADATA_DIR,
    SMALL_SEPARATOR,
)


# ==========================================================
# FILE PATHS
# ==========================================================

SEGMENT_METADATA = METADATA_DIR / "segment_metadata.csv"

MASTER_DATASET = METADATA_DIR / "master_dataset.csv"

REPORT_FILE = METADATA_DIR / "master_dataset_report.txt"


# ==========================================================
# CONSTANTS
# ==========================================================

# All known YouTube column name variants across genre xlsx files
YOUTUBE_COLUMN_VARIANTS = {
    "youtube link",   # Bhangra / Bihu / Garba / Maand  (exact case varies)
    "youtube url",
    "youtube",
    "source",         # Baul
    "url",
}


# ==========================================================
# LOAD SEGMENT METADATA
# ==========================================================

def load_segment_metadata() -> pd.DataFrame:
    """
    Read the CSV produced by 03_segment_audio.py and normalise
    the merge key.

    Normalisation strips any trailing .wav / .WAV suffix from
    Original Song ID.  This handles the Garba double-extension
    case where the processed files were named GARBA_001.wav.wav,
    causing Path.stem to return GARBA_001.wav instead of
    GARBA_001.

    Returns
    -------
    pd.DataFrame
        Segment metadata with a clean merge key.

    Raises
    ------
    FileNotFoundError
        If segment_metadata.csv does not exist.
    """

    if not SEGMENT_METADATA.exists():
        raise FileNotFoundError(
            f"Segment metadata not found:\n{SEGMENT_METADATA}"
        )

    df = pd.read_csv(SEGMENT_METADATA)

    # Normalise merge key: strip any embedded .wav suffix
    df["Original Song ID"] = (
        df["Original Song ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.wav$", "", case=False, regex=True)
    )

    return df


# ==========================================================
# NORMALISE ONE GENRE DATAFRAME
# ==========================================================

def _normalise_genre_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise one per-genre DataFrame so that it has exactly:

    - File Name     : cleaned song identifier (no .wav suffix)
    - Song Title    : standardised from any Song Name / Title variant
    - Artist        : standardised from any Singer variant
    - YouTube Link  : single canonical URL column (coalesced from all
                      column name variants used across genre sheets)

    This normalisation is applied per-genre BEFORE pd.concat so
    that the final concatenated DataFrame never contains duplicate
    or ambiguously named columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from one genre xlsx file.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with canonical column names.
    """

    # ---- Rename well-known column aliases --------------------------------

    rename_map = {
        "Song Name":   "Song Title",
        "Title":       "Song Title",
        "Singer":      "Artist",
    }
    df = df.rename(columns=rename_map)

    # ---- Coalesce YouTube URL into one canonical column ------------------
    # Identify every YouTube-related column by lower-cased name
    yt_cols = [
        c for c in df.columns
        if c.lower().replace(" ", "").rstrip("_") in {
            "youtubelink", "youtubeurl", "youtube", "source", "url"
        }
        or c.lower() in YOUTUBE_COLUMN_VARIANTS
    ]

    if yt_cols:
        # Build a single series: first non-null value across all yt_cols
        coalesced = df[yt_cols[0]].copy()
        for col in yt_cols[1:]:
            coalesced = coalesced.combine_first(df[col])
        df["YouTube Link"] = coalesced
        # Drop all source columns except the canonical one
        drop = [c for c in yt_cols if c != "YouTube Link"]
        if drop:
            df = df.drop(columns=drop)

    # ---- Normalise File Name ---------------------------------------------
    # Fall back to Processed_File_Name when File Name is NaN (Garba)
    if "Processed_File_Name" in df.columns:
        if "File Name" not in df.columns:
            df["File Name"] = pd.NA
        df["File Name"] = df["File Name"].fillna(df["Processed_File_Name"])

    if "File Name" in df.columns:
        df["File Name"] = (
            df["File Name"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.wav$", "", case=False, regex=True)
        )

    return df


# ==========================================================
# LOAD ONE GENRE METADATA
# ==========================================================

def load_genre_metadata(genre: str) -> pd.DataFrame:
    """
    Read and normalise the per-genre Excel file from METADATA_DIR.

    Each genre sheet is normalised individually before being
    returned.  This guarantees that pd.concat never produces
    duplicate column names.

    Parameters
    ----------
    genre : str
        Genre name matching the xlsx filename prefix.

    Returns
    -------
    pd.DataFrame
        Normalised metadata for the genre with Genre column added.

    Raises
    ------
    FileNotFoundError
        If the expected xlsx file does not exist.
    """

    metadata_file = METADATA_DIR / f"{genre}_metadata.xlsx"

    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)

    df = pd.read_excel(metadata_file)
    df["Genre"] = genre

    df = _normalise_genre_df(df)

    return df


# ==========================================================
# LOAD ALL GENRES
# ==========================================================

def load_all_metadata() -> pd.DataFrame:
    """
    Load and concatenate all per-genre Excel files.

    Because each DataFrame is normalised by load_genre_metadata
    before concatenation, the result has clean, non-duplicate
    column names.

    Returns
    -------
    pd.DataFrame
        Combined, normalised metadata for all genres.
    """

    dataframes = []

    print()
    print(LINE_SEPARATOR)
    print("Loading Metadata")
    print(LINE_SEPARATOR)

    for genre in tqdm(GENRES):
        dataframe = load_genre_metadata(genre)
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows from the combined metadata.

    Parameters
    ----------
    metadata : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    before = len(metadata)
    metadata = metadata.drop_duplicates()
    after = len(metadata)

    print()
    print(f"Duplicate Rows Removed : {before - after}")

    return metadata


# ==========================================================
# VALIDATE REQUIRED COLUMNS
# ==========================================================

def validate_metadata(metadata: pd.DataFrame) -> None:
    """
    Raise ValueError if any mandatory column is absent from
    the combined metadata.

    Parameters
    ----------
    metadata : pd.DataFrame

    Raises
    ------
    ValueError
        Lists every missing column name.
    """

    required = ["Genre", "Song Title", "Artist", "File Name"]
    missing = [c for c in required if c not in metadata.columns]

    if missing:
        raise ValueError(f"Missing Columns : {missing}")


# ==========================================================
# PREPARE METADATA
# ==========================================================

def prepare_metadata() -> pd.DataFrame:
    """
    Full pipeline: load, normalise, deduplicate, and validate
    all per-genre xlsx metadata.

    Returns
    -------
    pd.DataFrame
        Clean metadata with canonical column names, ready for
        merging with segment data.
    """

    metadata = load_all_metadata()

    metadata = remove_duplicates(metadata)

    validate_metadata(metadata)

    print()
    print(LINE_SEPARATOR)
    print("Metadata Ready")
    print(LINE_SEPARATOR)
    print(f"Total Songs : {len(metadata)}")
    print(LINE_SEPARATOR)

    return metadata


# ==========================================================
# MERGE SEGMENT METADATA
# ==========================================================

def merge_metadata() -> pd.DataFrame:
    """
    Left-join segment_metadata.csv with the combined xlsx
    metadata on the normalised song identifier.

    Left key  (segment_metadata) : Original Song ID
    Right key (xlsx metadata)    : File Name

    Both keys are normalised (no .wav suffix, no whitespace)
    before the join to guarantee a clean match regardless of
    how individual xlsx sheets were populated.

    Returns
    -------
    pd.DataFrame
        Master dataset: one row per audio segment, enriched
        with per-song metadata.
    """

    metadata = prepare_metadata()

    segment_metadata = load_segment_metadata()

    print()
    print(LINE_SEPARATOR)
    print("Merging Segment Metadata")
    print(LINE_SEPARATOR)

    master = segment_metadata.merge(
        metadata,
        left_on="Original Song ID",
        right_on="File Name",
        how="left",
        suffixes=("_seg", "_meta"),
    )

    # Resolve the Genre column.
    # _seg always comes from segment_metadata (correct source of truth).
    if "Genre_seg" in master.columns and "Genre_meta" in master.columns:
        master["Genre"] = master["Genre_seg"].combine_first(master["Genre_meta"])
        master = master.drop(columns=["Genre_seg", "Genre_meta"])
    elif "Genre_x" in master.columns:
        master["Genre"] = master["Genre_x"]
        drop_cols = [c for c in ["Genre_x", "Genre_y"] if c in master.columns]
        master = master.drop(columns=drop_cols)

    print(f"Total Segments : {len(master)}")
    print()

    return master


# ==========================================================
# VALIDATE MASTER DATASET
# ==========================================================

def validate_master_dataset(master: pd.DataFrame) -> pd.DataFrame:
    """
    Report metadata completeness for the merged master dataset.

    Validates Song Title, Artist, and the canonical YouTube Link
    column for null values, and checks for duplicate segment IDs.

    Parameters
    ----------
    master : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        The unmodified master (validation is advisory only).
    """

    print()
    print(LINE_SEPARATOR)
    print("Validating Master Dataset")
    print(LINE_SEPARATOR)

    missing_titles  = master["Song Title"].isna().sum()
    missing_artist  = master["Artist"].isna().sum()

    yt_col = "YouTube Link" if "YouTube Link" in master.columns else None
    missing_youtube = int(master[yt_col].isna().sum()) if yt_col else "N/A"

    print(f"Missing Song Title : {missing_titles}")
    print(f"Missing Artist     : {missing_artist}")
    print(f"Missing YouTube    : {missing_youtube}")
    print()

    duplicate_segments = master["Segment ID"].duplicated().sum()
    print(f"Duplicate Segments : {duplicate_segments}")
    print()

    return master


# ==========================================================
# SORT MASTER DATASET
# ==========================================================

def sort_master_dataset(master: pd.DataFrame) -> pd.DataFrame:
    """
    Sort master dataset for deterministic, reproducible output.

    Sort order: Genre -> Original Song ID -> Segment ID

    Parameters
    ----------
    master : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return master.sort_values(
        by=["Genre", "Original Song ID", "Segment ID"],
        ignore_index=True,
    )


# ==========================================================
# SAVE MASTER DATASET
# ==========================================================

def save_master_dataset(master: pd.DataFrame) -> None:
    """
    Persist the master dataset as a CSV file.

    Parameters
    ----------
    master : pd.DataFrame
    """

    master.to_csv(MASTER_DATASET, index=False)

    print()
    print(LINE_SEPARATOR)
    print("Master dataset saved.")
    print(MASTER_DATASET)
    print(LINE_SEPARATOR)


# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report(master: pd.DataFrame) -> None:
    """
    Write a plain-text summary report to disk.

    Parameters
    ----------
    master : pd.DataFrame
    """

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("MASTER DATASET REPORT\n")
        report.write("=" * 60 + "\n\n")
        report.write(f"Total Segments : {len(master)}\n")
        report.write(f"Total Songs    : {master['Original Song ID'].nunique()}\n")
        report.write(f"Genres         : {master['Genre'].nunique()}\n\n")
        report.write("Genre-wise Statistics\n")
        report.write("-" * 60 + "\n")
        genre_counts = master.groupby("Genre").size()
        for genre, count in genre_counts.items():
            report.write(f"{genre:<15}: {count}\n")

    print()
    print("Report generated.")
    print(REPORT_FILE)
    print()


# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_summary(master: pd.DataFrame) -> None:
    """
    Print a concise end-of-run summary to stdout.

    Parameters
    ----------
    master : pd.DataFrame
    """

    print()
    print(LINE_SEPARATOR)
    print("MASTER DATASET SUMMARY")
    print(LINE_SEPARATOR)
    print(f"Total Segments : {len(master)}")
    print(f"Unique Songs   : {master['Original Song ID'].nunique()}")
    print(f"Genres         : {master['Genre'].nunique()}")
    print()
    print("Segments per Genre")
    print(SMALL_SEPARATOR)
    counts = master.groupby("Genre").size()
    for genre, count in counts.items():
        print(f"{genre:<15} {count}")
    print(LINE_SEPARATOR)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Entry point for the master dataset builder.

    Execution order
    ---------------
    1. Load + normalise + validate all xlsx metadata.
    2. Load + normalise segment_metadata.csv.
    3. Left-join on normalised song ID.
    4. Validate merged dataset.
    5. Sort, save, and report.
    """

    master = merge_metadata()

    master = validate_master_dataset(master)

    master = sort_master_dataset(master)

    save_master_dataset(master)

    generate_report(master)

    print_summary(master)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
