"""
=========================================================
Metadata Utility Functions

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap

=========================================================
"""

from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
import re
import pandas as pd

from utils.config import SUPPORTED_AUDIO_FORMATS


# ==========================================================
# SONG RECORD
# ==========================================================

@dataclass
class SongRecord:

    genre: str

    source_file: Path | None

    output_name: str

    song_title: str

    artist: str

    metadata_row: int


# ==========================================================
# LOAD METADATA
# ==========================================================

def load_metadata(metadata_file: Path):

    if not metadata_file.exists():

        raise FileNotFoundError(

            f"Metadata file not found:\n{metadata_file}"

        )

    dataframe = pd.read_excel(metadata_file)

    dataframe = dataframe.dropna(how="all")

    dataframe = dataframe.reset_index(drop=True)

    return dataframe


# ==========================================================
# VALIDATE METADATA
# ==========================================================

def validate_metadata(dataframe):

    required_columns = [

        "Song Title",

        "Artist"

    ]

    missing = []

    for column in required_columns:

        if column not in dataframe.columns:

            missing.append(column)

    if len(missing):

        raise ValueError(

            f"Missing columns : {missing}"

        )


# ==========================================================
# COLUMN HELPERS
# ==========================================================

def get_filename_column(dataframe):

    candidates = [

        "Processed_File_Name",

        "Processed File Name",

        "File Name",

        "Filename",

        "File_Name"

    ]

    for column in candidates:

        if column in dataframe.columns:

            return column

    raise ValueError("Output filename column not found.")


def get_original_column(dataframe):

    candidates = [

        "Original File Name",

        "Original_File_Name",

        "Original Filename"

    ]

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def get_song_title_column(dataframe):

    candidates = [

        "Song Title",

        "Song_Title",

        "Title"

    ]

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def get_artist_column(dataframe):

    candidates = [

        "Artist",

        "Singer"

    ]

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None

    # ==========================================================
# NORMALIZE FILENAMES
# ==========================================================

WORDS_TO_REMOVE = [

    "official",
    "video",
    "full",
    "song",
    "lyrics",
    "lyrical",
    "audio",
    "hd",
    "4k",
    "music",
    "feat",
    "featuring"

]


def normalize_filename(filename: str):

    if filename is None:
        return ""

    filename = str(filename)

    filename = Path(filename).stem

    filename = filename.lower()

    filename = filename.replace("&", "and")

    filename = re.sub(r"\(.*?\)", " ", filename)
    filename = re.sub(r"\[.*?\]", " ", filename)
    filename = re.sub(r"\{.*?\}", " ", filename)

    filename = filename.replace("_", " ")
    filename = filename.replace("-", " ")

    filename = re.sub(r"[^a-z0-9 ]", " ", filename)

    words = filename.split()

    cleaned = []

    for word in words:

        if word in WORDS_TO_REMOVE:
            continue

        cleaned.append(word)

    filename = " ".join(cleaned)

    filename = re.sub(r"\s+", " ", filename)

    return filename.strip()


# ==========================================================
# SIMILARITY SCORE
# ==========================================================

def similarity(a: str, b: str):

    return SequenceMatcher(
        None,
        normalize_filename(a),
        normalize_filename(b)
    ).ratio()


# ==========================================================
# CACHE AUDIO FILES
# ==========================================================

def build_audio_cache(folder: Path):

    cache = []

    if not folder.exists():

        return cache

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in [

            ext.lower()

            for ext in SUPPORTED_AUDIO_FORMATS

        ]:

            continue

        cache.append({

            "path": file,

            "name": file.name,

            "stem": file.stem,

            "normalized": normalize_filename(file.name)

        })

    return cache

    # ==========================================================
# SMART AUDIO SEARCH
# ==========================================================

def locate_audio_file(
    folder: Path,
    filename: str,
    cache=None
):

    if filename is None:
        return None

    if cache is None:
        cache = build_audio_cache(folder)

    target = normalize_filename(filename)

    if target == "":
        return None

    # ------------------------------------------------------
    # 1. Exact filename match
    # ------------------------------------------------------

    for item in cache:

        if item["name"].lower() == str(filename).lower():

            return item["path"]

    # ------------------------------------------------------
    # 2. Exact stem match
    # ------------------------------------------------------

    target_stem = Path(filename).stem.lower()

    for item in cache:

        if item["stem"].lower() == target_stem:

            return item["path"]

    # ------------------------------------------------------
    # 3. Normalized exact match
    # ------------------------------------------------------

    for item in cache:

        if item["normalized"] == target:

            return item["path"]

    # ------------------------------------------------------
    # 4. Substring match
    # ------------------------------------------------------

    for item in cache:

        current = item["normalized"]

        if target in current:

            return item["path"]

        if current in target:

            return item["path"]

    # ------------------------------------------------------
    # 5. Fuzzy match
    # ------------------------------------------------------

    best_match = None

    best_score = 0

    for item in cache:

        score = similarity(
            target,
            item["normalized"]
        )

        if score > best_score:

            best_score = score

            best_match = item

    if best_score >= 0.88:

        print()

        print("⚠ Fuzzy Match")

        print(f"Metadata : {filename}")

        print(f"Matched  : {best_match['name']}")

        print(f"Score    : {best_score:.2%}")

        print()

        return best_match["path"]

    return None

    # ==========================================================
# CREATE SONG RECORDS
# ==========================================================

def get_song_records(
    metadata_file: Path,
    raw_audio_folder: Path
):

    dataframe = load_metadata(metadata_file)

    validate_metadata(dataframe)

    output_column = get_filename_column(dataframe)

    original_column = get_original_column(dataframe)

    title_column = get_song_title_column(dataframe)

    artist_column = get_artist_column(dataframe)

    genre = metadata_file.stem

    genre = genre.replace("_metadata", "")

    genre = genre.replace("-1", "")

    genre = genre.replace("-2", "")

    cache = build_audio_cache(raw_audio_folder)

    records = []

    matched = 0

    unmatched = 0

    for index, row in dataframe.iterrows():

        output_name = str(row[output_column]).strip()

        if pd.isna(output_name) or output_name == "":
            continue

        if original_column is not None and pd.notna(row[original_column]):

            original_name = str(row[original_column]).strip()

        else:

            original_name = output_name

        source = locate_audio_file(
            raw_audio_folder,
            original_name,
            cache
        )

        if source is None:
            unmatched += 1
        else:
            matched += 1

        song_title = ""

        if title_column is not None and pd.notna(row[title_column]):

            song_title = str(row[title_column]).strip()

        artist = ""

        if artist_column is not None and pd.notna(row[artist_column]):

            artist = str(row[artist_column]).strip()

        records.append(

            SongRecord(

                genre=genre,

                source_file=source,

                output_name=output_name,

                song_title=song_title,

                artist=artist,

                metadata_row=index + 2

            )

        )

    print()

    print("=" * 60)

    print(f"{genre} Metadata Summary")

    print("=" * 60)

    print(f"Matched Songs   : {matched}")

    print(f"Unmatched Songs : {unmatched}")

    print("=" * 60)

    return records


# ==========================================================
# TOTAL SONGS
# ==========================================================

def total_songs(metadata_file: Path):

    return len(load_metadata(metadata_file))


# ==========================================================
# SAVE METADATA
# ==========================================================

def save_metadata(
    dataframe,
    metadata_file: Path
):

    dataframe.to_excel(
        metadata_file,
        index=False
    )


# ==========================================================
# VERIFY METADATA
# ==========================================================

def verify_metadata(metadata_file: Path):

    dataframe = load_metadata(metadata_file)

    validate_metadata(dataframe)

    output_column = get_filename_column(dataframe)

    duplicates = dataframe[output_column].duplicated().sum()

    return {

        "songs": len(dataframe),

        "duplicates": duplicates,

        "columns": list(dataframe.columns)

    }


# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_metadata_summary(metadata_file: Path):

    info = verify_metadata(metadata_file)

    print("=" * 60)

    print(metadata_file.name)

    print("=" * 60)

    print(f"Songs      : {info['songs']}")

    print(f"Duplicates : {info['duplicates']}")

    print("\nColumns:")

    for column in info["columns"]:

        print(f" • {column}")

    print("=" * 60)