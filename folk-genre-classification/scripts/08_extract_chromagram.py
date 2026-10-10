from __future__ import annotations

import argparse
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from utils.config import (
    AUGMENTED_TRAIN_FILE,
    FEATURE_DTYPE,
    FMAX,
    FMIN,
    GENRES,
    HOP_LENGTH,
    LINE_SEPARATOR,
    CHROMA_METADATA_FILE,
    CHROMAGRAM_DIR,
    CHROMA_LOG,
    METADATA_DIR,
    MIN_AUDIO_DURATION,
    N_FFT,
    SAMPLE_RATE,
    SMALL_SEPARATOR,
    TARGET_FRAMES,
    TARGET_SAMPLE_RATE,
    TEST_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
    WIN_LENGTH,
)
from utils.features_utils import (
    normalize_feature,
    pad_feature,
    validate_feature,
)


# ==========================================================
# CONSTANTS
# ==========================================================

#: All dataset split names and their fallback metadata files.
SPLIT_FILES: Dict[str, str] = {
    "train":      AUGMENTED_TRAIN_FILE,   # auto-falls back to TRAIN_FILE
    "validation": VALIDATION_FILE,
    "test":       TEST_FILE,
}

#: Required columns that must exist in every metadata CSV.
REQUIRED_COLUMNS: List[str] = [
    "Segment ID",
    "Original Song ID",
    "Genre",
    "Segment Path",
    "Segment Duration",
]

#: Feature file extension.
NPY_SUFFIX = ".npy"


# ==========================================================
# ARGUMENT PARSING
# ==========================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments. Currently supports ``--force`` to
        overwrite existing feature files.
    """
    parser = argparse.ArgumentParser(
        description="Extract Log Chromagrams for the folk music dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-extraction and overwrite existing .npy files.",
    )
    return parser.parse_args()


# ==========================================================
# METADATA LOADING
# ==========================================================

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Read a split metadata CSV and validate its required columns.

    Parameters
    ----------
    csv_path : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and validated metadata.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist on disk.
    ValueError
        If any required column is absent.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing}"
        )

    return df


def select_training_metadata() -> Tuple[pd.DataFrame, str]:
    """
    Choose the training metadata CSV with automatic fallback.

    Priority
    --------
    1. ``augmented_train.csv``  — used when it exists on disk.
    2. ``train.csv``            — fallback when augmented file is absent.

    Returns
    -------
    (df, source_name) : Tuple[pd.DataFrame, str]
        Loaded DataFrame and the filename actually used.
    """
    preferred = METADATA_DIR / AUGMENTED_TRAIN_FILE
    fallback  = METADATA_DIR / TRAIN_FILE

    if preferred.exists():
        return load_metadata(preferred), AUGMENTED_TRAIN_FILE

    if fallback.exists():
        return load_metadata(fallback), TRAIN_FILE

    raise FileNotFoundError(
        f"Neither {AUGMENTED_TRAIN_FILE} nor {TRAIN_FILE} found in {METADATA_DIR}"
    )


# ==========================================================
# AUDIO LOADING
# ==========================================================

def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load a WAV (or any librosa-supported) audio file.

    Always resamples to TARGET_SAMPLE_RATE and converts to mono.
    Uses librosa's native resampler for accuracy.

    Parameters
    ----------
    audio_path : Path
        Absolute path to the audio file.

    Returns
    -------
    audio : np.ndarray, shape (N,)
        Mono float32 audio samples.
    sample_rate : int
        Always equal to TARGET_SAMPLE_RATE after resampling.

    Raises
    ------
    RuntimeError
        Wraps any exception raised by librosa.load.
    """
    try:
        audio, sr = librosa.load(
            str(audio_path),
            sr=TARGET_SAMPLE_RATE,    # resample on load (most efficient)
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"librosa failed to load '{audio_path.name}': {exc}"
        ) from exc

    return audio, sr


# ==========================================================
# AUDIO VALIDATION
# ==========================================================

def validate_audio(
    audio: np.ndarray,
    sample_rate: int,
    audio_path: Path,
) -> None:
    """
    Assert audio quality before feature extraction.

    Checks
    ------
    1. Array is non-empty.
    2. Contains no NaN values.
    3. Contains no Inf values.
    4. Duration >= MIN_AUDIO_DURATION.
    5. Not all-silence (peak amplitude > 1e-6).

    Parameters
    ----------
    audio : np.ndarray
        Loaded mono audio samples.
    sample_rate : int
        Sample rate of the audio.
    audio_path : Path
        Used only in error messages.

    Raises
    ------
    ValueError
        On the first failed check with a descriptive message.
    """
    if audio.size == 0:
        raise ValueError(f"Empty audio: {audio_path.name}")

    if np.isnan(audio).any():
        raise ValueError(f"Audio contains NaN: {audio_path.name}")

    if np.isinf(audio).any():
        raise ValueError(f"Audio contains Inf: {audio_path.name}")

    duration = audio.size / sample_rate
    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio too short ({duration:.2f}s < {MIN_AUDIO_DURATION}s): "
            f"{audio_path.name}"
        )

    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError(f"Audio is effectively silent: {audio_path.name}")


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def generate_chromagram_cqt(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute a CQT Chromagram from a mono audio array.

    Processing steps
    ----------------
    1. Compute the CQT Chromagram using librosa.feature.chroma_cqt.
       This inherently maps frequencies to 12 pitch classes.
    2. CQT Chromagram values are naturally bounded in [0, 1] 
       normalized by their peak frame energy.
       However, to maintain absolute mathematical consistency 
       with the Mel Spectrogram pipeline (as requested), we will
       pass this through the explicit normalize_feature_values 
       Min-Max scalar afterwards.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono float32 audio samples.
    sample_rate : int
        Sample rate of the audio.

    Returns
    -------
    np.ndarray, shape (12, T)
        CQT Chromagram, float32.
    """
    chroma_data = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C1'),
        n_chroma=12
    )

    return chroma_data.astype(np.float32)


# ==========================================================
# FEATURE POST-PROCESSING
# ==========================================================

def normalize_feature_values(feature: np.ndarray) -> np.ndarray:
    """
    Min-max normalise feature values to the range [0, 1].

    Delegates to ``utils.features_utils.normalize_feature``,
    which handles the edge case where min == max (returns the
    array unchanged as a zero-tensor).

    Parameters
    ----------
    feature : np.ndarray
        Raw mel spectrogram array.

    Returns
    -------
    np.ndarray
        Normalised float32 array.
    """
    return normalize_feature(feature)


def pad_or_trim_feature(
    feature: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Pad (right-pad with zeros) or trim (right-trim) a spectrogram
    along the time axis so that it has exactly ``target_frames``
    columns.

    Delegates to ``utils.features_utils.pad_feature``.

    Parameters
    ----------
    feature : np.ndarray, shape (n_mels, T)
        Raw or normalised spectrogram.
    target_frames : int
        Desired number of time frames.

    Returns
    -------
    np.ndarray, shape (n_mels, target_frames)
        Padded / trimmed spectrogram.
    """
    return pad_feature(feature, target_frames)


# ==========================================================
# FEATURE VALIDATION
# ==========================================================

def validate_feature_array(
    feature: np.ndarray,
    segment_id: str,
) -> None:
    """
    Assert that a processed feature array is ready for saving.

    Checks
    ------
    1. No NaN values.
    2. No Inf values.
    3. Non-empty (size > 0).
    4. Shape is exactly (12, TARGET_FRAMES).
    5. dtype is float32.

    Parameters
    ----------
    feature : np.ndarray
        Processed feature to validate.
    segment_id : str
        Segment identifier used in error messages.

    Raises
    ------
    ValueError
        On any failed check with a descriptive message.
    """
    # Reuse library-level NaN/Inf/empty checks
    if not validate_feature(feature):
        raise ValueError(
            f"Feature for '{segment_id}' failed basic validation "
            "(NaN, Inf, or empty)."
        )

    expected_shape = (12, TARGET_FRAMES)
    if feature.shape != expected_shape:
        raise ValueError(
            f"Feature shape mismatch for '{segment_id}': "
            f"got {feature.shape}, expected {expected_shape}."
        )

    if feature.dtype != np.float32:
        raise ValueError(
            f"Feature dtype mismatch for '{segment_id}': "
            f"got {feature.dtype}, expected float32."
        )


# ==========================================================
# SAVING FEATURES
# ==========================================================

def build_output_path(
    split: str,
    genre: str,
    feature_name: str,
) -> Path:
    """
    Construct the canonical output path for a feature file.

    Layout
    ------
    CHROMAGRAM_DIR / <split> / <Genre> / <feature_name>.npy

    ``feature_name`` is derived from the **audio file stem**, not from
    the Segment ID.  This distinction is critical for augmented rows:

    - Original row  : audio stem == "BAUL_001_SEG000"
                      -> BAUL_001_SEG000.npy
    - Augmented rows: audio stem == "BAUL_001_SEG000_stretch" / "_pitch"
                      -> BAUL_001_SEG000_stretch.npy / BAUL_001_SEG000_pitch.npy

    Using the Segment ID as the filename (the previous behaviour) caused
    all three rows to resolve to the same .npy path, silently discarding
    the augmented spectrograms.

    Parameters
    ----------
    split : str
        Dataset split ("train", "validation", or "test").
    genre : str
        Genre label (e.g. "Baul").
    feature_name : str
        Stem of the source audio file
        (e.g. "BAUL_001_SEG000" or "BAUL_001_SEG000_stretch").

    Returns
    -------
    Path
        Absolute path to the output .npy file.
    """
    return CHROMAGRAM_DIR / split / genre / f"{feature_name}{NPY_SUFFIX}"


def save_feature(feature: np.ndarray, output_path: Path) -> None:
    """
    Persist a feature array to disk as a NumPy .npy file.

    Creates parent directories as needed.

    Parameters
    ----------
    feature : np.ndarray
        Float32 feature array to save.
    output_path : Path
        Destination file path (must end in .npy).

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), feature)
    except Exception as exc:
        raise IOError(
            f"Failed to write feature to '{output_path}': {exc}"
        ) from exc


# ==========================================================
# METADATA ROW CONSTRUCTION
# ==========================================================

def generate_metadata_row(
    segment_id: str,
    feature_name: str,
    original_song_id: str,
    genre: str,
    split: str,
    audio_path: Path,
    feature_path: Path,
    sample_rate: int,
    duration: float,
    feature: np.ndarray,
    timestamp: str,
) -> Dict[str, object]:
    """
    Build one metadata dictionary for a successfully processed segment.

    Parameters
    ----------
    segment_id : str
        Base segment identifier (e.g. "BAUL_001_SEG000").  Shared by the
        original row and all its augmented variants.
    feature_name : str
        Stem of the saved .npy file (e.g. "BAUL_001_SEG000_stretch").
        Unique per audio file.  Downstream scripts must use this column
        to locate the correct feature file, not the Segment ID.
    original_song_id : str
        Parent song identifier (e.g. "BAUL_001").
    genre : str
        Genre label.
    split : str
        Dataset split ("train", "validation", "test").
    audio_path : Path
        Path to the source WAV file (original or augmented).
    feature_path : Path
        Path where the .npy feature was saved.
    sample_rate : int
        Sample rate used during extraction.
    duration : float
        Audio duration in seconds.
    feature : np.ndarray
        The saved feature array (used for shape / frames).
    timestamp : str
        ISO-format processing timestamp string.

    Returns
    -------
    Dict[str, object]
        Metadata row dictionary.
    """
    return {
        "Segment ID":           segment_id,
        "Feature Name":         feature_name,
        "Original Song ID":     original_song_id,
        "Genre":                genre,
        "Dataset Split":        split,
        "Original Audio Path":  str(audio_path),
        "Chromagram Path": str(feature_path),
        "Sample Rate":          sample_rate,
        "Duration":             round(duration, 4),
        "Frames":               feature.shape[1],
        "Feature Shape":        str(feature.shape),
        "Timestamp":            timestamp,
    }


# ==========================================================
# LOG ROW CONSTRUCTION
# ==========================================================

def build_log_row(
    segment_id: str,
    genre: str,
    split: str,
    status: str,
    processing_time: float,
    error_message: str = "",
) -> Dict[str, object]:
    """
    Build one log-row dictionary for a single processed sample.

    Parameters
    ----------
    segment_id : str
        Segment identifier.
    genre : str
        Genre label.
    split : str
        Dataset split.
    status : str
        "SUCCESS", "SKIPPED", or "FAILED".
    processing_time : float
        Wall-clock seconds spent on this sample.
    error_message : str, optional
        Human-readable error reason (non-empty only for FAILED).

    Returns
    -------
    Dict[str, object]
    """
    return {
        "Timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Segment ID":      segment_id,
        "Genre":           genre,
        "Dataset":         split,
        "Status":          status,
        "Processing Time": round(processing_time, 6),
        "Error Message":   error_message,
    }


# ==========================================================
# LOG PERSISTENCE
# ==========================================================

def initialise_log(log_path: Path) -> None:
    """
    Create (or overwrite) the CSV log file with a header row.

    Parameters
    ----------
    log_path : Path
        Absolute path to the log CSV.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "Timestamp", "Segment ID", "Genre", "Dataset",
            "Status", "Processing Time", "Error Message",
        ])
        writer.writeheader()


def append_log_row(log_path: Path, row: Dict[str, object]) -> None:
    """
    Append a single log row to the CSV log file.

    Parameters
    ----------
    log_path : Path
        Absolute path to the log CSV (must already have a header).
    row : Dict[str, object]
        Log row dictionary.
    """
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writerow(row)


# ==========================================================
# METADATA PERSISTENCE
# ==========================================================

def save_metadata(
    metadata_rows: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """
    Persist all feature metadata rows to a CSV file.

    Parameters
    ----------
    metadata_rows : list
        All successfully processed segment metadata dicts.
    output_path : Path
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metadata_rows)
    df.to_csv(output_path, index=False)


# ==========================================================
# PROCESS ONE AUDIO SAMPLE
# ==========================================================

def process_audio(
    row: pd.Series,
    split: str,
    force_regenerate: bool,
    log_path: Path,
    metadata_rows: List[Dict[str, object]],
) -> str:
    """
    Full single-sample extraction pipeline.

    Performs the following in order:
    1.  Resolve output path; skip if feature already exists and
        ``force_regenerate`` is False.
    2.  Validate that the source audio file exists on disk.
    3.  Load audio.
    4.  Validate audio quality.
    5.  Extract Log Chromagram.
    6.  Normalise feature.
    7.  Pad / trim to TARGET_FRAMES.
    8.  Cast to float32.
    9.  Validate feature.
    10. Save feature to disk.
    11. Append metadata row.
    12. Append SUCCESS log row.

    Any exception at any step is caught, logged, and returns
    "FAILED" so that the pipeline continues with the next sample.

    Parameters
    ----------
    row : pd.Series
        One row from a split metadata DataFrame.
    split : str
        Dataset split name.
    force_regenerate : bool
        When True, overwrite existing .npy files.
    log_path : Path
        Path to the active log CSV.
    metadata_rows : List[Dict]
        Shared accumulator for successful metadata (mutated).

    Returns
    -------
    str
        "SUCCESS", "SKIPPED", or "FAILED".
    """
    t_start = time.perf_counter()

    segment_id    = str(row["Segment ID"])
    song_id       = str(row["Original Song ID"])
    genre         = str(row["Genre"])
    audio_path    = Path(str(row["Segment Path"]))
    duration_hint = float(row.get("Segment Duration", 0.0))

    # Derive the output filename from the audio file's stem, NOT from
    # Segment ID.  This is the critical fix: augmented audio files have
    # stems like "BAUL_001_SEG000_stretch" which must produce their own
    # .npy files distinct from the original "BAUL_001_SEG000.npy".
    feature_name = audio_path.stem
    output_path  = build_output_path(split, genre, feature_name)

    # ------------------------------------------------------------------
    # STEP 0 — Skip regeneration if already extracted, but validate
    # and record metadata so existing samples are preserved.
    # ------------------------------------------------------------------
    if output_path.exists() and not force_regenerate:
        try:
            chroma_fixed = np.load(str(output_path))
            validate_feature_array(chroma_fixed, feature_name)
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            error_msg = f"Existing feature validation failed: {exc}"
            append_log_row(log_path, build_log_row(
                feature_name, genre, split, "FAILED", elapsed, error_msg
            ))
            return "FAILED"

        sample_rate = int(row.get("Sample Rate", TARGET_SAMPLE_RATE))
        duration = float(row.get("Segment Duration", row.get("Duration", duration_hint)))
        if duration <= 0.0:
            duration = (chroma_fixed.shape[1] * HOP_LENGTH) / sample_rate

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_rows.append(generate_metadata_row(
            segment_id=segment_id,
            feature_name=feature_name,
            original_song_id=song_id,
            genre=genre,
            split=split,
            audio_path=audio_path,
            feature_path=output_path,
            sample_rate=sample_rate,
            duration=duration,
            feature=chroma_fixed,
            timestamp=timestamp,
        ))
        elapsed = time.perf_counter() - t_start
        append_log_row(log_path, build_log_row(
            feature_name, genre, split, "SKIPPED", elapsed
        ))
        return "SKIPPED"

    # ------------------------------------------------------------------
    # STEP 1 — Verify source file exists
    # ------------------------------------------------------------------
    if not audio_path.exists():
        elapsed = time.perf_counter() - t_start
        msg = f"Source file not found: {audio_path}"
        append_log_row(log_path, build_log_row(
            feature_name, genre, split, "FAILED", elapsed, msg
        ))
        return "FAILED"

    try:
        # --------------------------------------------------------------
        # STEP 2 — Load audio
        # --------------------------------------------------------------
        audio, sample_rate = load_audio(audio_path)

        # --------------------------------------------------------------
        # STEP 3 — Validate audio
        # --------------------------------------------------------------
        validate_audio(audio, sample_rate, audio_path)

        # Peak normalise
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # Duration
        duration = audio.size / sample_rate

        # --------------------------------------------------------------
        # STEP 4 — Extract mel spectrogram
        # --------------------------------------------------------------
        chroma_data = generate_chromagram_cqt(audio, sample_rate)

        # --------------------------------------------------------------
        # STEP 5 — Normalise
        # --------------------------------------------------------------
        chroma_norm = normalize_feature_values(chroma_data)

        # --------------------------------------------------------------
        # STEP 6 — Pad / Trim
        # --------------------------------------------------------------
        chroma_fixed = pad_or_trim_feature(chroma_norm, TARGET_FRAMES)

        # --------------------------------------------------------------
        # STEP 7 — Cast to float32
        # --------------------------------------------------------------
        chroma_fixed = chroma_fixed.astype(np.float32)

        # --------------------------------------------------------------
        # STEP 8 — Validate feature
        # --------------------------------------------------------------
        validate_feature_array(chroma_fixed, feature_name)

        # --------------------------------------------------------------
        # STEP 9 — Save feature
        # --------------------------------------------------------------
        save_feature(chroma_fixed, output_path)

        # --------------------------------------------------------------
        # STEP 10 — Record metadata
        # --------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_rows.append(generate_metadata_row(
            segment_id=segment_id,
            feature_name=feature_name,
            original_song_id=song_id,
            genre=genre,
            split=split,
            audio_path=audio_path,
            feature_path=output_path,
            sample_rate=sample_rate,
            duration=duration,
            feature=chroma_fixed,
            timestamp=timestamp,
        ))

        elapsed = time.perf_counter() - t_start
        append_log_row(log_path, build_log_row(
            feature_name, genre, split, "SUCCESS", elapsed
        ))
        return "SUCCESS"

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        error_msg = f"{type(exc).__name__}: {exc}"
        append_log_row(log_path, build_log_row(
            feature_name, genre, split, "FAILED", elapsed, error_msg
        ))
        return "FAILED"


# ==========================================================
# PROCESS ONE DATASET SPLIT
# ==========================================================

def process_dataset(
    df: pd.DataFrame,
    split: str,
    force_regenerate: bool,
    log_path: Path,
    metadata_rows: List[Dict[str, object]],
) -> Tuple[int, int, int]:
    """
    Iterate over every row in a metadata DataFrame and call
    ``process_audio`` for each.

    Parameters
    ----------
    df : pd.DataFrame
        Split metadata (train / validation / test).
    split : str
        Name of the dataset split.
    force_regenerate : bool
        Overwrite existing features when True.
    log_path : Path
        Active log CSV path.
    metadata_rows : List[Dict]
        Shared accumulator mutated by process_audio.

    Returns
    -------
    (success, skipped, failed) : Tuple[int, int, int]
        Counts for this split.
    """
    print()
    print(LINE_SEPARATOR)
    print(f"  Processing split : {split.upper()}")
    print(f"  Total samples    : {len(df):,}")
    print(LINE_SEPARATOR)

    success = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}", unit="seg"):
        try:
            status = process_audio(
                row, split, force_regenerate, log_path, metadata_rows
            )
        except Exception:
            # Outer safety net — should never trigger given inner try/except
            status = "FAILED"

        if status == "SUCCESS":
            success += 1
        elif status == "SKIPPED":
            skipped += 1
        else:
            failed += 1

    print()
    print(f"  ✓ Success : {success:,}")
    print(f"  ⊘ Skipped : {skipped:,}")
    print(f"  ✗ Failed  : {failed:,}")

    return success, skipped, failed


# ==========================================================
# DISK USAGE ESTIMATOR
# ==========================================================

def estimate_disk_usage(n_features: int) -> str:
    """
    Estimate total disk usage for all saved .npy files.

    Each file is a float32 array of shape (12, TARGET_FRAMES).
    Disk size ≈ 12 × TARGET_FRAMES × 4 bytes + 128 bytes header.

    Parameters
    ----------
    n_features : int
        Number of successfully saved feature files.

    Returns
    -------
    str
        Human-readable size string (e.g. "1.23 GB").
    """
    bytes_per_file = 12 * TARGET_FRAMES * 4 + 128
    total_bytes = n_features * bytes_per_file

    if total_bytes >= 1_073_741_824:
        return f"{total_bytes / 1_073_741_824:.2f} GB"
    if total_bytes >= 1_048_576:
        return f"{total_bytes / 1_048_576:.2f} MB"
    return f"{total_bytes / 1_024:.2f} KB"


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(
    split_counts: Dict[str, Tuple[int, int, int]],
    metadata_rows: List[Dict[str, object]],
    wall_time: float,
    metadata_path: Path,
    log_path: Path,
) -> None:
    """
    Print a rich, human-readable end-of-run summary.

    Parameters
    ----------
    split_counts : Dict[str, Tuple[int, int, int]]
        Per-split (success, skipped, failed) counts.
    metadata_rows : list
        All successful metadata rows (for aggregate statistics).
    wall_time : float
        Total elapsed wall-clock time in seconds.
    metadata_path : Path
        Where chroma_metadata.csv was saved.
    log_path : Path
        Where chromagram_log.csv was saved.
    """
    total_success = sum(v[0] for v in split_counts.values())
    total_skipped = sum(v[1] for v in split_counts.values())
    total_failed  = sum(v[2] for v in split_counts.values())
    total_samples = total_success + total_skipped + total_failed

    speed = total_success / wall_time if wall_time > 0 else 0.0
    disk  = estimate_disk_usage(total_success)

    print()
    print(LINE_SEPARATOR)
    print("  CHROMAGRAM EXTRACTION — SUMMARY")
    print(LINE_SEPARATOR)
    print(f"  Total Samples          : {total_samples:>10,}")
    print(f"  Successfully Processed : {total_success:>10,}")
    print(f"  Skipped (existing)     : {total_skipped:>10,}")
    print(f"  Failed                 : {total_failed:>10,}")
    print(SMALL_SEPARATOR)

    for split, (s, sk, f) in split_counts.items():
        label = f"{split.capitalize()} Features"
        print(f"  {label:<23}: {(s + sk):>10,}")

    print(SMALL_SEPARATOR)
    print(f"  Avg Processing Speed   : {speed:>9.1f} seg/s")
    print(f"  Total Processing Time  : {wall_time:>9.2f} s")
    print(f"  Estimated Disk Usage   : {disk:>10}")
    print(SMALL_SEPARATOR)
    print(f"  Metadata  -> {metadata_path}")
    print(f"  Log       -> {log_path}")

    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        print(SMALL_SEPARATOR)
        print("  Segments per Genre (successful)")
        print(SMALL_SEPARATOR)
        genre_counts = df.groupby("Genre")["Segment ID"].count()
        for genre, count in genre_counts.items():
            print(f"  {genre:<15} : {count:,}")

    print(LINE_SEPARATOR)
    print()


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Entry point for the Log Chromagram extraction pipeline.

    Execution order
    ---------------
    1.  Parse CLI arguments (--force flag).
    2.  Print configuration banner.
    3.  Initialise log file.
    4.  Load metadata for training (with auto-fallback),
        validation, and test splits.
    5.  Process each split sequentially.
    6.  Save feature metadata CSV.
    7.  Print summary.
    """
    args = parse_args()
    force: bool = args.force

    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print()
    print(LINE_SEPARATOR)
    print("  LOG CHROMAGRAM EXTRACTION PIPELINE")
    print(LINE_SEPARATOR)
    print(f"  Output Dir    : {CHROMAGRAM_DIR}")
    print(f"  Sample Rate   : {TARGET_SAMPLE_RATE} Hz")
    print(f"  N_FFT         : {N_FFT}")
    print(f"  Hop Length    : {HOP_LENGTH}")
    print(f"  Win Length    : {WIN_LENGTH}")
    print(f"  Chroma Bins   : 12")
    print(f"  FMIN / FMAX   : {FMIN} Hz / {FMAX} Hz")
    print(f"  Target Frames : {TARGET_FRAMES}")
    print(f"  Feature Shape : (12, {TARGET_FRAMES})")
    print(f"  Feature dtype : {FEATURE_DTYPE}")
    print(f"  Force Regen   : {force}")
    print(LINE_SEPARATOR)

    # ------------------------------------------------------------------
    # Ensure output split/genre directories exist
    # ------------------------------------------------------------------
    for split in ("train", "validation", "test"):
        for genre in GENRES:
            (CHROMAGRAM_DIR / split / genre).mkdir(
                parents=True, exist_ok=True
            )

    # ------------------------------------------------------------------
    # Initialise log
    # ------------------------------------------------------------------
    initialise_log(CHROMA_LOG)

    # ------------------------------------------------------------------
    # Accumulate metadata across all splits
    # ------------------------------------------------------------------
    metadata_rows: List[Dict[str, object]] = []
    split_counts:  Dict[str, Tuple[int, int, int]] = {}

    # ------------------------------------------------------------------
    # TRAINING SPLIT
    # ------------------------------------------------------------------
    try:
        train_df, train_source = select_training_metadata()
        print()
        print(f"  Training metadata : {train_source} ({len(train_df):,} rows)")
        s, sk, f = process_dataset(
            train_df, "train", force, CHROMA_LOG, metadata_rows
        )
        split_counts["train"] = (s, sk, f)
    except Exception as exc:
        print(f"\n  [ERROR] Training split failed: {exc}")
        split_counts["train"] = (0, 0, 0)

    # ------------------------------------------------------------------
    # VALIDATION SPLIT
    # ------------------------------------------------------------------
    try:
        val_df = load_metadata(METADATA_DIR / VALIDATION_FILE)
        s, sk, f = process_dataset(
            val_df, "validation", force, CHROMA_LOG, metadata_rows
        )
        split_counts["validation"] = (s, sk, f)
    except Exception as exc:
        print(f"\n  [ERROR] Validation split failed: {exc}")
        split_counts["validation"] = (0, 0, 0)

    # ------------------------------------------------------------------
    # TEST SPLIT
    # ------------------------------------------------------------------
    try:
        test_df = load_metadata(METADATA_DIR / TEST_FILE)
        s, sk, f = process_dataset(
            test_df, "test", force, CHROMA_LOG, metadata_rows
        )
        split_counts["test"] = (s, sk, f)
    except Exception as exc:
        print(f"\n  [ERROR] Test split failed: {exc}")
        split_counts["test"] = (0, 0, 0)

    # ------------------------------------------------------------------
    # Save feature metadata
    # ------------------------------------------------------------------
    metadata_path = METADATA_DIR / CHROMA_METADATA_FILE
    if metadata_rows:
        save_metadata(metadata_rows, metadata_path)
        print()
        print(f"  Feature metadata saved -> {metadata_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wall_time = time.perf_counter() - pipeline_start
    print_summary(
        split_counts, metadata_rows, wall_time, metadata_path, CHROMA_LOG
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
