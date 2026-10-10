"""
=========================================================
02_standardize_audio.py

Standardizes every audio file into

• WAV
• Mono
• 22050 Hz
• PCM 16-bit

Author:
Dhrupad Kashyap
=========================================================
"""

from pathlib import Path

from tqdm import tqdm

from utils.config import (
    METADATA_DIR,
    RAW_AUDIO_DIR,
    PROCESSED_AUDIO_DIR,
    GENRES,
    STANDARDIZATION_LOG,
    LINE_SEPARATOR
)

from utils.metadata_utils import (
    get_song_records
)

from utils.audio_utils import (
    process_audio_file
)

from utils.file_utils import (
    ensure_directory
)

from utils.logger import (
    ProjectLogger
)

logger = ProjectLogger(
    STANDARDIZATION_LOG
)


# ==========================================================
# PROCESS ONE GENRE
# ==========================================================

def process_genre(genre):

    print()

    print(LINE_SEPARATOR)

    print(f"Processing Genre : {genre}")

    print(LINE_SEPARATOR)

    metadata_file = None

    for file in METADATA_DIR.glob(f"{genre}_metadata*.xlsx"):

        metadata_file = file

        break

    if metadata_file is None:

        print("Metadata not found.")

        return 0, 0

    raw_folder = RAW_AUDIO_DIR / genre

    processed_folder = PROCESSED_AUDIO_DIR / genre

    ensure_directory(processed_folder)

    songs = get_song_records(
        metadata_file,
        raw_folder
    )

    success = 0

    failed = 0

    for song in tqdm(songs, desc=genre):

        print("\n--------------------------------")
        print(f"{song.song_title} -> {song.source_file}")
        
        try:

            if song.source_file is None:

                logger.log(
                    genre=genre,
                    song=song.song_title,
                    status="FAILED",
                    message="Source audio not found"
                )

                failed += 1
                continue

            output_file = processed_folder / f"{song.output_name}.wav"

            info = process_audio_file(
                song.source_file,
                output_file
            )

            logger.log(
                genre=genre,
                song=song.song_title,
                status="SUCCESS",
                message=f"Duration: {info['duration']:.2f}s"
            )

            success += 1

        except Exception as e:
            logger.log(
                genre=genre,
                song=song.song_title,
                status="FAILED",
                message=str(e)
            )

            failed += 1

    print()

    print(f"✓ Success : {success}")

    print(f"✗ Failed  : {failed}")

    return success, failed


# ==========================================================
# MAIN
# ==========================================================

def main():

    total_success = 0

    total_failed = 0

    print("\n" + LINE_SEPARATOR)

    print("STANDARDIZING AUDIO DATASET")

    print(LINE_SEPARATOR)

    for genre in GENRES:

        success, failed = process_genre(genre)

        total_success += success

        total_failed += failed

    print("\n" + LINE_SEPARATOR)

    print("SUMMARY")

    print(LINE_SEPARATOR)

    print(f"Successful Files : {total_success}")

    print(f"Failed Files     : {total_failed}")

    print(LINE_SEPARATOR)


if __name__ == "__main__":

    main()