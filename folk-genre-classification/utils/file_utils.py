"""
=========================================================
File Utility Functions
=========================================================
"""

from pathlib import Path
from typing import List

from utils.config import SUPPORTED_AUDIO_FORMATS


# ==========================================================
# Create directory if it doesn't exist
# ==========================================================

def ensure_directory(directory: Path):
    """
    Creates a directory if it does not exist.
    """
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Check whether file is audio
# ==========================================================

def is_audio_file(file: Path) -> bool:
    """
    Returns True if file is a supported audio file.
    """
    return file.suffix.lower() in SUPPORTED_AUDIO_FORMATS


# ==========================================================
# Get all audio files from folder
# ==========================================================

def get_audio_files(folder: Path) -> List[Path]:
    """
    Returns all supported audio files in a folder.
    """
    files = []

    for ext in SUPPORTED_AUDIO_FORMATS:
        files.extend(folder.glob(f"*{ext}"))

    return sorted(files)


# ==========================================================
# Find a file by filename
# ==========================================================

def find_audio_file(folder: Path, filename: str):
    """
    Finds an audio file regardless of extension.
    """

    stem = Path(filename).stem

    for ext in SUPPORTED_AUDIO_FORMATS:
        candidate = folder / f"{stem}{ext}"

        if candidate.exists():
            return candidate

    return None


# ==========================================================
# Get filename without extension
# ==========================================================

def filename_without_extension(file: Path):
    return file.stem


# ==========================================================
# Change extension
# ==========================================================

def change_extension(file: Path, extension: str):
    return file.with_suffix(extension)