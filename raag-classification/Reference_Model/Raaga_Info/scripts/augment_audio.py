import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import numpy as np
import soundfile as sf
from tqdm import tqdm

from audiomentations import Compose, PitchShift, TimeStretch

# =====================================================
# INPUT / OUTPUT
# =====================================================

INPUT_ROOT = str(BASE_DIR / "processed_data" / "segmented_audio")

OUTPUT_ROOT = str(BASE_DIR / "processed_data" / "augmented_audio")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# AUGMENTATIONS
# =====================================================

pitch_up = Compose([
    PitchShift(
        min_semitones=1,
        max_semitones=1,
        p=1.0
    )
])

pitch_down = Compose([
    PitchShift(
        min_semitones=-1,
        max_semitones=-1,
        p=1.0
    )
])

speed_up = Compose([
    TimeStretch(
        min_rate=1.1,
        max_rate=1.1,
        p=1.0
    )
])

speed_down = Compose([
    TimeStretch(
        min_rate=0.9,
        max_rate=0.9,
        p=1.0
    )
])

print("=" * 60)
print("Creating augmented dataset...")
print("=" * 60)

total_processed = 0
total_skipped = 0

# =====================================================
# PROCESS ALL FILES
# =====================================================

for root, dirs, files in os.walk(INPUT_ROOT):

    relative = os.path.relpath(root, INPUT_ROOT)

    output_folder = os.path.join(
        OUTPUT_ROOT,
        relative
    )

    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(files):

        if not file.lower().endswith(".wav"):
            continue

        input_file = os.path.join(root, file)

        # -------------------------
        # Read audio safely
        # -------------------------

        try:
            audio, sr = sf.read(input_file)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = audio.astype(np.float32)

        except Exception as e:
            print("\n--------------------------------------")
            print("Skipping corrupted file:")
            print(input_file)
            print(e)
            print("--------------------------------------")
            total_skipped += 1
            continue

        filename = os.path.splitext(file)[0]

        # -------------------------
        # Save original
        # -------------------------

        sf.write(
            os.path.join(output_folder, filename + "_orig.wav"),
            audio,
            sr
        )

        # -------------------------
        # Pitch Up
        # -------------------------

        aug = pitch_up(
            samples=audio,
            sample_rate=sr
        )

        sf.write(
            os.path.join(output_folder, filename + "_pitchup.wav"),
            aug,
            sr
        )

        # -------------------------
        # Pitch Down
        # -------------------------

        aug = pitch_down(
            samples=audio,
            sample_rate=sr
        )

        sf.write(
            os.path.join(output_folder, filename + "_pitchdown.wav"),
            aug,
            sr
        )

        # -------------------------
        # Speed Up
        # -------------------------

        aug = speed_up(
            samples=audio,
            sample_rate=sr
        )

        sf.write(
            os.path.join(output_folder, filename + "_speedup.wav"),
            aug,
            sr
        )

        # -------------------------
        # Speed Down
        # -------------------------

        aug = speed_down(
            samples=audio,
            sample_rate=sr
        )

        sf.write(
            os.path.join(output_folder, filename + "_speeddown.wav"),
            aug,
            sr
        )

        total_processed += 1

print("\n" + "=" * 60)
print("Finished!")
print("=" * 60)
print(f"Processed files : {total_processed}")
print(f"Skipped files   : {total_skipped}")
print(f"Saved to        : {OUTPUT_ROOT}")
print("=" * 60)