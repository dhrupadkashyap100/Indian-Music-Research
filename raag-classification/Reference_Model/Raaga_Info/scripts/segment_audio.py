import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import subprocess
import pandas as pd
from tqdm import tqdm

# ===================================================
# INPUT DATASET
# ===================================================

INPUT_ROOT = str(BASE_DIR / "reference_dataset" / "Saraga" / "saraga1.5_hindustani")
CSV_PATH = str(BASE_DIR / "reference_dataset" / "metadata" / "saraga_metadata.csv")

# ===================================================
# OUTPUT DATASET
# ===================================================

OUTPUT_ROOT = str(BASE_DIR / "processed_data" / "segmented_audio")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEGMENT_LENGTH = 10   # seconds

print("=" * 60)
print("Creating 10-second audio clips...")
print("=" * 60)

# Load valid metadata
df = pd.read_csv(CSV_PATH)
df = df.head(5)  # SUBSET TEST
print(f"Loaded {len(df)} valid recordings from metadata (SUBSET TEST).")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
    input_file = os.path.join(str(row['folder']), str(row['audio_file']))
    canonical_raga = str(row['canonical_raga'])
    unique_id = str(row['unique_id'])

    if not os.path.exists(input_file):
        print(f"Warning: Audio file not found: {input_file}")
        continue

    output_folder = os.path.join(OUTPUT_ROOT, canonical_raga)
    os.makedirs(output_folder, exist_ok=True)

    output_pattern = os.path.join(output_folder, f"{unique_id}_segment_%03d.wav")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_file,
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_LENGTH),
        "-c",
        "copy",
        output_pattern
    ]

    subprocess.run(command)

print("\nFinished!")
print("Dataset saved to:")
print(OUTPUT_ROOT)