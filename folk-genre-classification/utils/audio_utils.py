"""
=========================================================
Audio Utility Functions
=========================================================

Author : Dhrupad Kashyap

This module contains reusable audio processing
functions used throughout the project.

=========================================================
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

from utils.config import (
    TARGET_SAMPLE_RATE,
    TARGET_SUBTYPE,
)

# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(audio_file: Path):
    """
    Load audio without changing its original sample rate.
    """

    audio, sample_rate = librosa.load(
        audio_file,
        sr=None,
        mono=False
    )

    return audio, sample_rate


# ==========================================================
# SAVE AUDIO
# ==========================================================

def save_audio(
    audio,
    output_file: Path,
    sample_rate=TARGET_SAMPLE_RATE
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_file,
        audio,
        sample_rate,
        subtype=TARGET_SUBTYPE
    )


# ==========================================================
# CONVERT TO MONO
# ==========================================================

def convert_to_mono(audio):

    if audio.ndim == 1:
        return audio

    return librosa.to_mono(audio)


# ==========================================================
# RESAMPLE AUDIO
# ==========================================================

def resample_audio(
    audio,
    original_sr,
    target_sr=TARGET_SAMPLE_RATE
):

    if original_sr == target_sr:
        return audio

    return librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=target_sr
    )


# ==========================================================
# NORMALIZE AUDIO
# ==========================================================

def normalize_audio(audio):

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak


# ==========================================================
# AUDIO DURATION
# ==========================================================

def get_duration(
    audio,
    sample_rate
):

    return librosa.get_duration(
        y=audio,
        sr=sample_rate
    )


# ==========================================================
# RMS LOUDNESS
# ==========================================================

def calculate_rms(audio):

    return np.sqrt(
        np.mean(audio ** 2)
    )


# ==========================================================
# AUDIO INFORMATION
# ==========================================================

def get_audio_info(
    audio,
    sample_rate
):

    info = {

        "sample_rate": sample_rate,

        "duration": get_duration(
            audio,
            sample_rate
        ),

        "samples": len(audio),

        "rms": calculate_rms(audio)

    }

    return info

    # ==========================================================
# SILENCE DETECTION
# ==========================================================

def is_silent(
    audio,
    threshold=1e-4
):
    """
    Returns True if the audio is effectively silent.
    """

    return np.max(np.abs(audio)) < threshold


# ==========================================================
# CLIPPING DETECTION
# ==========================================================

def is_clipped(
    audio,
    threshold=0.999
):
    """
    Detects whether audio contains clipped samples.
    """

    return np.any(np.abs(audio) >= threshold)


# ==========================================================
# CHECK NaN VALUES
# ==========================================================

def has_nan(audio):

    return np.isnan(audio).any()


# ==========================================================
# CHECK INF VALUES
# ==========================================================

def has_inf(audio):

    return np.isinf(audio).any()


# ==========================================================
# AUDIO VALIDATION
# ==========================================================

def validate_audio(
    audio,
    sample_rate,
    min_duration=1.0
):
    """
    Validate processed audio.

    Returns:
        (True, "OK")
        (False, "Reason")
    """

    if audio is None:
        return False, "Audio is None"

    if len(audio) == 0:
        return False, "Empty Audio"

    if has_nan(audio):
        return False, "Contains NaN"

    if has_inf(audio):
        return False, "Contains Inf"

    duration = get_duration(audio, sample_rate)

    if duration < min_duration:
        return False, "Too Short"

    if is_silent(audio):
        return False, "Silent Audio"

    return True, "OK"


# ==========================================================
# STANDARDIZE AUDIO
# ==========================================================

def standardize_audio(audio_file: Path):
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Load audio
    2. Convert to mono
    3. Resample to target sample rate
    4. Normalize amplitude
    5. Validate audio

    Returns
    -------
    audio : np.ndarray
    info  : dict
    """

    audio, sr = load_audio(audio_file)

    audio = convert_to_mono(audio)

    audio = resample_audio(
        audio,
        sr,
        TARGET_SAMPLE_RATE
    )

    audio = normalize_audio(audio)

    valid, reason = validate_audio(
        audio,
        TARGET_SAMPLE_RATE
    )

    if not valid:

        raise ValueError(
            f"Invalid audio: {reason}"
        )

    info = get_audio_info(
        audio,
        TARGET_SAMPLE_RATE
    )

    info["clipped"] = is_clipped(audio)

    return audio, info


# ==========================================================
# PROCESS AND SAVE
# ==========================================================

def process_audio_file(
    input_file: Path,
    output_file: Path
):
    """
    Standardize and save a single audio file.

    Returns
    -------
    info : dict
    """

    audio, info = standardize_audio(input_file)

    save_audio(
        audio,
        output_file,
        TARGET_SAMPLE_RATE
    )

    return info