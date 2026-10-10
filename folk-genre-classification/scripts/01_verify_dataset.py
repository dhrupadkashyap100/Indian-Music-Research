from pathlib import Path
import pandas as pd
import soundfile as sf

# =====================================================
# DATASET PATHS
# =====================================================

BASE = Path("datasets")
RAW_AUDIO = BASE / "raw_audio"
METADATA = BASE / "metadata"

genres = [
    "Baul",
    "Bhangra",
    "Bihu",
    "Dhamail",
    "Garba",
    "Ghoomar",
    "Jhumur",
    "Lavani",
    "Maand",
    "Sambalpuri"
]

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

# =====================================================
# VERIFY EACH GENRE
# =====================================================

for genre in genres:

    print(f"\nChecking {genre}...")

    metadata_file = METADATA / f"{genre}_metadata.xlsx"
    audio_folder = RAW_AUDIO / genre

    if not metadata_file.exists():
        print("❌ Metadata file missing!")
        continue

    if not audio_folder.exists():
        print("❌ Audio folder missing!")
        continue

    # Read metadata
    df = pd.read_excel(metadata_file)

    # Find all WAV and MP3 files
    wav_files = list(audio_folder.glob("*.wav"))
    mp3_files = list(audio_folder.glob("*.mp3"))

    audio_files = wav_files + mp3_files

    print(f"Metadata entries : {len(df)}")
    print(f"Audio files      : {len(audio_files)}")

    if len(df) == len(audio_files):
        print("Count            : ✅ Match")
    else:
        print("Count            : ❌ Mismatch")

    corrupted = 0

    for audio in audio_files:
        try:
            sf.info(audio)
        except Exception:
            corrupted += 1

    print(f"Corrupted files  : {corrupted}")

print("\n" + "=" * 60)
print("Verification completed.")
print("=" * 60)