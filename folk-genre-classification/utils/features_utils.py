"""
=========================================================
Feature Utility Functions

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap

=========================================================
"""

from pathlib import Path

import librosa
import numpy as np

from utils.audio_utils import load_audio
from utils.config import (
    TARGET_SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    N_MFCC,
    FMIN,
    FMAX,
)


# ==========================================================
# LOAD STANDARDIZED AUDIO
# ==========================================================

def load_processed_audio(audio_file: Path):

    audio, sample_rate = load_audio(audio_file)

    if sample_rate != TARGET_SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

        sample_rate = TARGET_SAMPLE_RATE

    return audio, sample_rate


# ==========================================================
# SAVE FEATURE
# ==========================================================

def save_feature(

    feature,

    output_file: Path

):

    output_file.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    np.save(

        output_file,

        feature

    )


# ==========================================================
# LOAD FEATURE
# ==========================================================

def load_feature(

    feature_file: Path

):

    return np.load(

        feature_file,
        mmap_mode='r'

    )


# ==========================================================
# FEATURE SHAPE
# ==========================================================

def get_feature_shape(

    feature

):

    return feature.shape


# ==========================================================
# FEATURE DTYPE
# ==========================================================

def get_feature_dtype(

    feature

):

    return feature.dtype


# ==========================================================
# FEATURE INFO
# ==========================================================

def get_feature_info(

    feature

):

    return {

        "shape": feature.shape,

        "dtype": str(feature.dtype),

        "min": float(np.min(feature)),

        "max": float(np.max(feature)),

        "mean": float(np.mean(feature)),

        "std": float(np.std(feature))

    }


# ==========================================================
# FEATURE VALIDATION
# ==========================================================

def validate_feature(

    feature

):

    if feature is None:

        return False

    if np.isnan(feature).any():

        return False

    if np.isinf(feature).any():

        return False

    if feature.size == 0:

        return False

    return True

# ==========================================================
# CHROMAGRAM
# ==========================================================

def extract_chromagram(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    chromagram = librosa.feature.chroma_stft(

        y=audio,

        sr=sample_rate,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH

    )

    return chromagram

def extract_chromagram_cqt(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    chromagram = librosa.feature.chroma_cqt(

        y=audio,

        sr=sample_rate,

        hop_length=HOP_LENGTH,

        fmin=librosa.note_to_hz("C1"),
        
        n_chroma=12

    )
    
    return chromagram



# ==========================================================
# MEL SPECTROGRAM
# ==========================================================

def extract_melspectrogram(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=sample_rate,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH,

        n_mels=N_MELS,

        fmin=FMIN,

        fmax=FMAX

    )

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max

    )

    return mel_db


# ==========================================================
# MFCC
# ==========================================================

def extract_mfcc(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sample_rate,

        n_mfcc=N_MFCC,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH,

        fmin=FMIN,

        fmax=FMAX

    )

    return mfcc


# ==========================================================
# DELTA MFCC
# ==========================================================

def extract_delta_mfcc(

    mfcc

):

    return librosa.feature.delta(

        mfcc

    )


# ==========================================================
# DELTA-DELTA MFCC
# ==========================================================

def extract_delta2_mfcc(

    mfcc

):

    return librosa.feature.delta(

        mfcc,

        order=2

    )


# ==========================================================
# COMPLETE MFCC FEATURE
# ==========================================================

def extract_complete_mfcc(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    mfcc = extract_mfcc(

        audio,

        sample_rate

    )

    delta = extract_delta_mfcc(

        mfcc

    )

    delta2 = extract_delta2_mfcc(

        mfcc

    )

    return np.vstack(

        [

            mfcc,

            delta,

            delta2

        ]

    )

    # ==========================================================
# SWARAGRAM (CQT BASED)
# ==========================================================

def extract_swaragram(

    audio,

    sample_rate=TARGET_SAMPLE_RATE

):

    cqt = np.abs(

        librosa.cqt(

            y=audio,

            sr=sample_rate,

            hop_length=HOP_LENGTH,

            fmin=librosa.note_to_hz("C1")

        )

    )

    swaragram = librosa.amplitude_to_db(

        cqt,

        ref=np.max

    )

    return swaragram


# ==========================================================
# FEATURE NORMALIZATION
# ==========================================================

def normalize_feature(

    feature

):

    feature = feature.astype(np.float32)

    minimum = np.min(feature)

    maximum = np.max(feature)

    if maximum == minimum:

        return feature

    feature = (feature - minimum) / (

        maximum - minimum

    )

    return feature


# ==========================================================
# PAD FEATURE
# ==========================================================

def pad_feature(

    feature,

    target_frames=938

):

    current_frames = feature.shape[1]

    if current_frames == target_frames:

        return feature

    if current_frames > target_frames:

        return feature[:, :target_frames]

    padding = target_frames - current_frames

    return np.pad(

        feature,

        ((0, 0), (0, padding)),

        mode="constant"

    )


# ==========================================================
# PREPARE FEATURE
# ==========================================================

def prepare_feature(

    feature,

    target_frames=938

):

    feature = normalize_feature(

        feature

    )

    feature = pad_feature(

        feature,

        target_frames

    )

    return feature


# ==========================================================
# FEATURE DIMENSIONS
# ==========================================================

def get_frames(

    feature

):

    return feature.shape[1]


def get_bins(

    feature

):

    return feature.shape[0]


# ==========================================================
# FEATURE SUMMARY
# ==========================================================

def print_feature_summary(

    feature,

    name="Feature"

):

    print()

    print("=" * 60)

    print(name)

    print("=" * 60)

    print(f"Shape : {feature.shape}")

    print(f"Min   : {feature.min():.4f}")

    print(f"Max   : {feature.max():.4f}")

    print(f"Mean  : {feature.mean():.4f}")

    print(f"Std   : {feature.std():.4f}")

    print("=" * 60)

    # ==========================================================
# COMPLETE FEATURE EXTRACTION
# ==========================================================

from utils.config import TARGET_FRAMES


def extract_all_features(
    audio,
    sample_rate=TARGET_SAMPLE_RATE
):
    """
    Extract every feature required by the project.

    Returns
    -------
    dict
    """

    chromagram = prepare_feature(

        extract_chromagram(
            audio,
            sample_rate
        ),

        TARGET_FRAMES

    )

    mel = prepare_feature(

        extract_melspectrogram(
            audio,
            sample_rate
        ),

        TARGET_FRAMES

    )

    mfcc = prepare_feature(

        extract_complete_mfcc(
            audio,
            sample_rate
        ),

        TARGET_FRAMES

    )

    swaragram = prepare_feature(

        extract_swaragram(
            audio,
            sample_rate
        ),

        TARGET_FRAMES

    )

    return {

        "chromagram": chromagram,

        "melspectrogram": mel,

        "mfcc": mfcc,

        "swaragram": swaragram

    }


# ==========================================================
# SAVE ALL FEATURES
# ==========================================================

def save_all_features(
    features,
    output_folder,
    file_name
):

    output_folder = Path(output_folder)

    save_feature(

        features["chromagram"],

        output_folder /
        "chromagrams" /
        f"{file_name}.npy"

    )

    save_feature(

        features["melspectrogram"],

        output_folder /
        "melspectrograms" /
        f"{file_name}.npy"

    )

    save_feature(

        features["mfcc"],

        output_folder /
        "mfcc" /
        f"{file_name}.npy"

    )

    save_feature(

        features["swaragram"],

        output_folder /
        "swaragrams" /
        f"{file_name}.npy"

    )


# ==========================================================
# PROCESS SINGLE AUDIO
# ==========================================================

def process_audio_file(
    audio_file
):

    audio, sample_rate = load_processed_audio(
        audio_file
    )

    return extract_all_features(
        audio,
        sample_rate
    )


# ==========================================================
# VERIFY FEATURE
# ==========================================================

def verify_feature(feature):

    if not validate_feature(feature):

        raise ValueError(
            "Invalid feature."
        )

    return True


# ==========================================================
# VERIFY ALL FEATURES
# ==========================================================

def verify_all_features(features):

    for name, feature in features.items():

        verify_feature(feature)

    return True