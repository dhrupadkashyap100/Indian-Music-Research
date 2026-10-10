"""
=========================================================
03_segment_audio.py

Audio Segmentation Pipeline

Project : AI-Based Analysis of Indian Traditional Folk Music
Author  : Dhrupad Kashyap

Purpose
-------
Converts every standardized (processed) WAV file into
overlapping fixed-length audio segments.

Key Features
------------
Intelligent intro-skip: skips the first N seconds of
  every song (proportional to duration, capped at a max),
  avoiding long instrumental intros that degrade model
  quality.
Overlapping windows: 10-second segments with 5-second
  hop size produce rich, diverse training data.
Strict validation: per-file checks for existence,
  readability, sample rate, channel count, duration,
  and audio integrity before any I/O is performed.
Graceful error handling: one bad file never aborts
  the entire pipeline.
Rich metadata: every segment row is fully self-
  describing for seamless use by downstream scripts
  (04_build_master_dataset.py and beyond).
Structured CSV log: per-song log records for
  auditing and debugging.

Usage
-----
    python -m scripts.03_segment_audio

=========================================================
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from utils.config import (
    GENRES,
    INTRO_SKIP_PERCENT,
    LINE_SEPARATOR,
    MAX_INTRO_SKIP,
    METADATA_DIR,
    MIN_SEGMENT_DURATION,
    PROCESSED_AUDIO_DIR,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
    SEGMENTATION_LOG,
    SEGMENTED_AUDIO_DIR,
    SMALL_SEPARATOR,
    SUPPORTED_AUDIO_FORMATS,
    TARGET_EXTENSION,
    TARGET_SAMPLE_RATE,
)


# ==========================================================
# DERIVED CONSTANTS  (computed once, never re-derived)
# ==========================================================

#: Number of samples in one full segment
SEGMENT_SAMPLES: int = int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)

#: Number of samples in the overlap region
OVERLAP_SAMPLES: int = int(SEGMENT_OVERLAP * TARGET_SAMPLE_RATE)

#: How many samples to advance the window each iteration
HOP_SAMPLES: int = SEGMENT_SAMPLES - OVERLAP_SAMPLES

#: Minimum sample count a segment must have to be saved
MIN_SEGMENT_SAMPLES: int = int(MIN_SEGMENT_DURATION * TARGET_SAMPLE_RATE)


# ==========================================================
# TYPE ALIASES
# ==========================================================

MetadataRow = Dict[str, object]
LogRow = Dict[str, object]


# ==========================================================
# AUDIO LOADING
# ==========================================================

def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file from disk and return (samples, sample_rate).

    Parameters
    ----------
    audio_path : Path
        Absolute path to the WAV file.

    Returns
    -------
    audio : np.ndarray, shape (N,)
        1-D float32 array of audio samples.
    sample_rate : int
        Sample rate of the file.

    Raises
    ------
    RuntimeError
        If soundfile cannot open or decode the file.
    """
    try:
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"soundfile could not read '{audio_path.name}': {exc}"
        ) from exc

    return audio, sample_rate


# ==========================================================
# AUDIO VALIDATION
# ==========================================================

def validate_audio(
    audio_path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """
    Assert that an audio array meets every pipeline requirement.

    Checks (in order):
    1. File actually exists on disk.
    2. Supported audio format (extension in SUPPORTED_AUDIO_FORMATS).
    3. Correct sample rate (TARGET_SAMPLE_RATE).
    4. Mono channel (1-D array).
    5. Non-empty array (at least one sample).
    6. Duration is long enough to produce at least one segment
       after the intro skip.
    7. Audio is not all silence / all zeros (corrupted signal).

    Parameters
    ----------
    audio_path : Path
        Path used for error messages and extension checks.
    audio : np.ndarray
        Loaded audio samples.
    sample_rate : int
        Sample rate reported by soundfile.

    Raises
    ------
    ValueError
        On any validation failure, with a descriptive message.
    """

    # 1. Existence (defensive; caller should only pass existing paths)
    if not audio_path.exists():
        raise ValueError(f"File does not exist: {audio_path}")

    # 2. Supported format
    if audio_path.suffix.lower() not in [
        fmt.lower() for fmt in SUPPORTED_AUDIO_FORMATS
    ]:
        raise ValueError(
            f"Unsupported format '{audio_path.suffix}'. "
            f"Supported: {SUPPORTED_AUDIO_FORMATS}"
        )

    # 3. Sample rate
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate mismatch: got {sample_rate} Hz, "
            f"expected {TARGET_SAMPLE_RATE} Hz."
        )

    # 4. Mono
    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono (1-D), got shape {audio.shape}. "
            "Run 02_standardize_audio.py first."
        )

    # 5. Non-empty
    if audio.size == 0:
        raise ValueError("Audio array is empty (zero samples).")

    # 6. Duration after intro skip
    total_duration = audio.size / sample_rate
    intro_skip = calculate_intro_skip(total_duration)
    effective_duration = total_duration - intro_skip

    if effective_duration < MIN_SEGMENT_DURATION:
        raise ValueError(
            f"Effective duration after intro skip is {effective_duration:.2f}s, "
            f"which is shorter than MIN_SEGMENT_DURATION={MIN_SEGMENT_DURATION}s."
        )

    # 7. Not silent / corrupted
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError(
            "Audio appears to be all silence (max amplitude < 1e-6). "
            "File may be corrupted."
        )


# ==========================================================
# INTRO SKIP CALCULATION
# ==========================================================

def calculate_intro_skip(duration: float) -> float:
    """
    Compute how many seconds to skip at the start of a song.

    Formula:
        intro_skip = min(duration x INTRO_SKIP_PERCENT, MAX_INTRO_SKIP)

    Parameters
    ----------
    duration : float
        Total duration of the audio file in seconds.

    Returns
    -------
    float
        Number of seconds to skip before segmentation begins.
    """
    return min(duration * INTRO_SKIP_PERCENT, float(MAX_INTRO_SKIP))


# ==========================================================
# SEGMENT AUDIO
# ==========================================================

def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    intro_skip: float,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Slice a 1-D audio array into overlapping fixed-length segments,
    starting after the intro skip.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio samples (float32).
    sample_rate : int
        Sample rate (used only for time <-> sample conversion).
    intro_skip : float
        Seconds to discard from the beginning before slicing.

    Returns
    -------
    List of (segment_samples, start_time_s, end_time_s) tuples.
    Segments shorter than MIN_SEGMENT_DURATION are discarded.
    """
    intro_skip_samples: int = int(intro_skip * sample_rate)
    working_audio: np.ndarray = audio[intro_skip_samples:]

    segments: List[Tuple[np.ndarray, float, float]] = []
    cursor: int = 0
    total_working_samples: int = len(working_audio)

    while cursor < total_working_samples:
        end_cursor: int = cursor + SEGMENT_SAMPLES
        chunk: np.ndarray = working_audio[cursor:end_cursor]

        if len(chunk) < MIN_SEGMENT_SAMPLES:
            # Too short - discard the tail
            break

        # Absolute times relative to the full (un-skipped) audio
        start_time: float = (intro_skip_samples + cursor) / sample_rate
        end_time: float = (intro_skip_samples + cursor + len(chunk)) / sample_rate

        segments.append((chunk, start_time, end_time))
        cursor += HOP_SAMPLES

    return segments


# ==========================================================
# SAVE SEGMENT
# ==========================================================

def save_segment(
    segment: np.ndarray,
    output_path: Path,
    sample_rate: int,
) -> None:
    """
    Write a segment to disk as a 16-bit PCM WAV file.

    Parameters
    ----------
    segment : np.ndarray
        Mono float32 audio samples.
    output_path : Path
        Destination file path (must have a .wav extension).
    sample_rate : int
        Target sample rate.

    Raises
    ------
    IOError
        If soundfile cannot write to the specified path.
    """
    try:
        sf.write(
            str(output_path),
            segment,
            sample_rate,
            subtype="PCM_16",
        )
    except Exception as exc:
        raise IOError(
            f"Failed to write segment '{output_path.name}': {exc}"
        ) from exc


# ==========================================================
# CREATE METADATA ROW
# ==========================================================

def create_metadata(
    genre: str,
    song_id: str,
    segment_id: str,
    segment_path: Path,
    start_time: float,
    end_time: float,
    segment_duration: float,
    original_duration: float,
    intro_skip: float,
    segment_number: int,
    timestamp: str,
) -> MetadataRow:
    """
    Build one metadata dictionary for a single segment.

    The column names and values are chosen for direct compatibility
    with 04_build_master_dataset.py, which merges on 'Original Song ID'.

    Parameters
    ----------
    genre : str
        Genre label (e.g. "Baul").
    song_id : str
        Stem of the processed audio file, WITHOUT extension
        (e.g. "BAUL_001"). Must match the 'File Name' column in
        the Excel metadata sheets used by 04_build_master_dataset.py.
    segment_id : str
        Unique segment identifier (e.g. "BAUL_001_SEG000").
    segment_path : Path
        Absolute path to the saved segment WAV file.
    start_time : float
        Segment start offset in seconds (relative to full audio).
    end_time : float
        Segment end offset in seconds (relative to full audio).
    segment_duration : float
        Actual duration of this segment in seconds.
    original_duration : float
        Total duration of the source audio file in seconds.
    intro_skip : float
        Number of seconds skipped at the start of the source file.
    segment_number : int
        0-based sequential index of this segment within the song.
    timestamp : str
        ISO-format processing timestamp string.

    Returns
    -------
    MetadataRow
        Dictionary ready to be appended to the metadata list.
    """
    return {
        "Genre":                   genre,
        "Original Song ID":        song_id,
        "Segment ID":              segment_id,
        "Segment Path":            str(segment_path),
        "Start Time":              round(start_time, 4),
        "End Time":                round(end_time, 4),
        "Segment Duration":        round(segment_duration, 4),
        "Original Audio Duration": round(original_duration, 4),
        "Intro Skip":              round(intro_skip, 4),
        "Segment Number":          segment_number,
        "Processing Timestamp":    timestamp,
    }


# ==========================================================
# SAVE METADATA
# ==========================================================

def save_metadata(metadata_rows: List[MetadataRow]) -> Path:
    """
    Persist the segment metadata list to a CSV file.

    Parameters
    ----------
    metadata_rows : List[MetadataRow]
        All segment metadata accumulated during the run.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    output_path: Path = METADATA_DIR / "segment_metadata.csv"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metadata_rows)
    df.to_csv(output_path, index=False)

    return output_path


# ==========================================================
# CREATE LOG ROW
# ==========================================================

def create_log(
    song_name: str,
    genre: str,
    audio_duration: float,
    intro_skip: float,
    segments_created: int,
    processing_time: float,
    status: str,
    reason: str = "",
) -> LogRow:
    """
    Build one log-row dictionary for a single processed song.

    Parameters
    ----------
    song_name : str
        Stem of the source audio file (no extension).
    genre : str
        Genre label.
    audio_duration : float
        Total duration of the source audio in seconds.
    intro_skip : float
        Seconds skipped at the beginning.
    segments_created : int
        How many segments were written for this song.
    processing_time : float
        Wall-clock seconds spent processing this song.
    status : str
        One of "SUCCESS", "SKIPPED", or "FAILED".
    reason : str, optional
        Human-readable reason (used when status != "SUCCESS").

    Returns
    -------
    LogRow
        Dictionary ready to be saved to the segmentation log CSV.
    """
    return {
        "Song Name":        song_name,
        "Genre":            genre,
        "Audio Duration":   round(audio_duration, 4) if audio_duration else "",
        "Intro Skip":       round(intro_skip, 4) if intro_skip else "",
        "Segments Created": segments_created,
        "Processing Time":  round(processing_time, 4),
        "Status":           status,
        "Reason":           reason,
    }


# ==========================================================
# SAVE LOG
# ==========================================================

def save_log(log_rows: List[LogRow]) -> Path:
    """
    Persist the per-song log list to a CSV file.

    Parameters
    ----------
    log_rows : List[LogRow]
        All log records accumulated during the run.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    SEGMENTATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(log_rows)
    df.to_csv(SEGMENTATION_LOG, index=False)

    return SEGMENTATION_LOG


# ==========================================================
# PROCESS ONE SONG
# ==========================================================

def process_song(
    audio_path: Path,
    output_dir: Path,
    genre: str,
    metadata_rows: List[MetadataRow],
    log_rows: List[LogRow],
) -> str:
    """
    Load, validate, segment, and save one audio file.

    Parameters
    ----------
    audio_path : Path
        Full path to the processed WAV file.
    output_dir : Path
        Genre-specific directory under SEGMENTED_AUDIO_DIR.
    genre : str
        Genre label used in metadata and logging.
    metadata_rows : List[MetadataRow]
        Shared list - new segment rows are appended in-place.
    log_rows : List[LogRow]
        Shared list - a single song-level log row is appended.

    Returns
    -------
    str
        One of "SUCCESS", "SKIPPED", or "FAILED".
    """
    song_name: str = audio_path.stem
    song_start: float = time.perf_counter()

    audio_duration: float = 0.0
    intro_skip: float = 0.0
    segments_created: int = 0

    # ---- Early skip: file must exist and have a supported suffix --------
    if not audio_path.exists():
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="SKIPPED", reason="File not found on disk",
        ))
        return "SKIPPED"

    if audio_path.suffix.lower() not in [
        fmt.lower() for fmt in SUPPORTED_AUDIO_FORMATS
    ]:
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="SKIPPED",
            reason=f"Unsupported format: {audio_path.suffix}",
        ))
        return "SKIPPED"

    # ---- Load -----------------------------------------------------------
    try:
        audio, sample_rate = load_audio(audio_path)
    except RuntimeError as exc:
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="FAILED", reason=str(exc),
        ))
        return "FAILED"

    audio_duration = len(audio) / sample_rate

    # ---- Validate -------------------------------------------------------
    try:
        validate_audio(audio_path, audio, sample_rate)
    except ValueError as exc:
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="SKIPPED", reason=str(exc),
        ))
        return "SKIPPED"

    # ---- Intro skip -----------------------------------------------------
    intro_skip = calculate_intro_skip(audio_duration)

    # ---- Segment --------------------------------------------------------
    try:
        segments = segment_audio(audio, sample_rate, intro_skip)
    except Exception as exc:
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="FAILED", reason=f"Segmentation error: {exc}",
        ))
        return "FAILED"

    if not segments:
        elapsed = time.perf_counter() - song_start
        log_rows.append(create_log(
            song_name, genre, audio_duration, intro_skip,
            segments_created, elapsed,
            status="SKIPPED",
            reason="No valid segments produced after intro skip",
        ))
        return "SKIPPED"

    # ---- Write segments -------------------------------------------------
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for seg_number, (seg_samples, start_t, end_t) in enumerate(segments):

        segment_id: str = f"{song_name}_SEG{seg_number:03d}"
        segment_filename: str = f"{segment_id}{TARGET_EXTENSION}"
        output_path: Path = output_dir / segment_filename

        try:
            save_segment(seg_samples, output_path, sample_rate)
        except IOError as exc:
            # Log individually but stop processing this song
            log_rows.append(create_log(
                song_name, genre, audio_duration, intro_skip,
                segments_created, time.perf_counter() - song_start,
                status="FAILED",
                reason=f"Write error on segment {seg_number}: {exc}",
            ))
            return "FAILED"

        seg_duration: float = len(seg_samples) / sample_rate

        metadata_rows.append(create_metadata(
            genre=genre,
            song_id=song_name,            # no extension - matches 'File Name' in xlsx
            segment_id=segment_id,
            segment_path=output_path,
            start_time=start_t,
            end_time=end_t,
            segment_duration=seg_duration,
            original_duration=audio_duration,
            intro_skip=intro_skip,
            segment_number=seg_number,
            timestamp=timestamp,
        ))

        segments_created += 1

    # ---- Success --------------------------------------------------------
    elapsed = time.perf_counter() - song_start
    log_rows.append(create_log(
        song_name, genre, audio_duration, intro_skip,
        segments_created, elapsed,
        status="SUCCESS",
    ))

    return "SUCCESS"


# ==========================================================
# PROCESS ONE GENRE
# ==========================================================

def process_genre(
    genre: str,
    metadata_rows: List[MetadataRow],
    log_rows: List[LogRow],
) -> Tuple[int, int, int]:
    """
    Iterate over every WAV file in the processed folder for one genre
    and delegate each file to process_song().

    Parameters
    ----------
    genre : str
        Genre directory name (must exist under PROCESSED_AUDIO_DIR).
    metadata_rows : List[MetadataRow]
        Shared accumulator - appended to in-place by process_song().
    log_rows : List[LogRow]
        Shared accumulator - appended to in-place by process_song().

    Returns
    -------
    (successful, failed, skipped) : Tuple[int, int, int]
        Counts for this genre.
    """
    input_dir: Path = PROCESSED_AUDIO_DIR / genre
    output_dir: Path = SEGMENTED_AUDIO_DIR / genre
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(LINE_SEPARATOR)
    print(f"  Genre : {genre}")
    print(LINE_SEPARATOR)

    if not input_dir.exists():
        print(f"  [WARN] Processed folder not found: {input_dir}")
        print(f"         Run 02_standardize_audio.py first.")
        return 0, 0, 0

    # Collect only files whose extension matches supported formats
    audio_files: List[Path] = sorted([
        f for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in [fmt.lower() for fmt in SUPPORTED_AUDIO_FORMATS]
    ])

    if not audio_files:
        print(f"  [WARN] No audio files found in: {input_dir}")
        return 0, 0, 0

    successful: int = 0
    failed: int = 0
    skipped: int = 0

    for audio_path in tqdm(audio_files, desc=f"  {genre}", unit="file"):
        try:
            status = process_song(
                audio_path, output_dir, genre, metadata_rows, log_rows
            )
        except Exception as exc:
            # Absolute safety net - catch anything unforeseen
            log_rows.append(create_log(
                audio_path.stem, genre, 0.0, 0.0, 0,
                0.0, status="FAILED",
                reason=f"Unexpected exception: {traceback.format_exc(limit=3)}",
            ))
            status = "FAILED"

        if status == "SUCCESS":
            successful += 1
        elif status == "FAILED":
            failed += 1
        else:
            skipped += 1

    print()
    print(f"  ✓ Success : {successful}")
    print(f"  ✗ Failed  : {failed}")
    print(f"  ⊘ Skipped : {skipped}")

    return successful, failed, skipped


# ==========================================================
# PROCESS ENTIRE DATASET
# ==========================================================

def process_dataset() -> Tuple[int, int, int, List[MetadataRow], List[LogRow]]:
    """
    Iterate over every genre in GENRES and aggregate results.

    Returns
    -------
    total_success : int
    total_failed  : int
    total_skipped : int
    metadata_rows : List[MetadataRow]
    log_rows      : List[LogRow]
    """
    metadata_rows: List[MetadataRow] = []
    log_rows: List[LogRow] = []

    total_success: int = 0
    total_failed: int = 0
    total_skipped: int = 0

    for genre in GENRES:
        success, failed, skipped = process_genre(
            genre, metadata_rows, log_rows
        )
        total_success += success
        total_failed += failed
        total_skipped += skipped

    return total_success, total_failed, total_skipped, metadata_rows, log_rows


# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_summary(
    total_success: int,
    total_failed: int,
    total_skipped: int,
    metadata_rows: List[MetadataRow],
    log_rows: List[LogRow],
    wall_time: float,
    metadata_path: Path,
    log_path: Path,
) -> None:
    """
    Print a human-readable end-of-run summary to stdout.

    Parameters
    ----------
    total_success : int
        Songs that completed with at least one segment written.
    total_failed : int
        Songs that raised an error during processing.
    total_skipped : int
        Songs that were skipped due to validation failures.
    metadata_rows : List[MetadataRow]
        Full list of segment metadata (used to compute aggregates).
    log_rows : List[LogRow]
        Full list of per-song log records.
    wall_time : float
        Total elapsed wall-clock time in seconds.
    metadata_path : Path
        Where segment_metadata.csv was written.
    log_path : Path
        Where segmentation_log.csv was written.
    """
    total_songs: int = total_success + total_failed + total_skipped
    total_segments: int = len(metadata_rows)

    avg_segments: float = (
        total_segments / total_success if total_success > 0 else 0.0
    )

    # Compute averages from metadata DataFrame for accuracy
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        avg_intro_skip: float = float(df["Intro Skip"].mean())
        avg_audio_duration: float = float(df["Original Audio Duration"].mean())
    else:
        avg_intro_skip = 0.0
        avg_audio_duration = 0.0

    print()
    print(LINE_SEPARATOR)
    print("  SEGMENTATION SUMMARY")
    print(LINE_SEPARATOR)
    print(f"  Total Songs              : {total_songs}")
    print(f"  Successful Songs         : {total_success}")
    print(f"  Failed Songs             : {total_failed}")
    print(f"  Skipped Songs            : {total_skipped}")
    print(SMALL_SEPARATOR)
    print(f"  Total Segments Generated : {total_segments}")
    print(f"  Avg Segments per Song    : {avg_segments:.2f}")
    print(f"  Avg Intro Skip           : {avg_intro_skip:.2f} s")
    print(f"  Avg Audio Duration       : {avg_audio_duration:.2f} s")
    print(f"  Total Processing Time    : {wall_time:.2f} s")
    print(SMALL_SEPARATOR)
    print(f"  Metadata Saved  -> {metadata_path}")
    print(f"  Log Saved       -> {log_path}")

    # Genre-wise breakdown
    if metadata_rows:
        print(SMALL_SEPARATOR)
        print("  Segments per Genre")
        print(SMALL_SEPARATOR)
        genre_counts = df.groupby("Genre")["Segment ID"].count()
        for genre, count in genre_counts.items():
            print(f"  {genre:<15} : {count}")

    print(LINE_SEPARATOR)
    print()


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Entry point for the audio segmentation pipeline.

    Execution order:
    1. Print header and configuration banner.
    2. Iterate over all genres and process each song.
    3. Persist segment_metadata.csv.
    4. Persist segmentation_log.csv.
    5. Print summary.
    """
    pipeline_start: float = time.perf_counter()

    # ---- Header --------------------------------------------------------
    print()
    print(LINE_SEPARATOR)
    print("  AUDIO SEGMENTATION PIPELINE")
    print(LINE_SEPARATOR)
    print(f"  Input       : {PROCESSED_AUDIO_DIR}")
    print(f"  Output      : {SEGMENTED_AUDIO_DIR}")
    print(f"  Sample Rate : {TARGET_SAMPLE_RATE} Hz")
    print(f"  Segment Dur : {SEGMENT_DURATION} s")
    print(f"  Overlap     : {SEGMENT_OVERLAP} s")
    print(f"  Min Seg Dur : {MIN_SEGMENT_DURATION} s")
    print(f"  Intro Skip  : {int(INTRO_SKIP_PERCENT * 100)}% "
          f"(max {MAX_INTRO_SKIP} s)")
    print(LINE_SEPARATOR)

    # ---- Process -------------------------------------------------------
    (
        total_success,
        total_failed,
        total_skipped,
        metadata_rows,
        log_rows,
    ) = process_dataset()

    # ---- Persist -------------------------------------------------------
    metadata_path: Path = save_metadata(metadata_rows)
    log_path: Path = save_log(log_rows)

    wall_time: float = time.perf_counter() - pipeline_start

    # ---- Summary -------------------------------------------------------
    print_summary(
        total_success,
        total_failed,
        total_skipped,
        metadata_rows,
        log_rows,
        wall_time,
        metadata_path,
        log_path,
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
