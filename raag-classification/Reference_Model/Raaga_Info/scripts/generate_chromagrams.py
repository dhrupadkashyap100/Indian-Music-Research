import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import numpy as np
import librosa
from tqdm import tqdm

# =====================================================
# INPUT / OUTPUT
# =====================================================

INPUT_ROOT = str(BASE_DIR / "processed_data" / "augmented_audio")

OUTPUT_ROOT = str(BASE_DIR / "processed_data" / "chromagrams")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGET_FRAMES = 938

print("=" * 60)
print("Generating Chromagrams...")
print("=" * 60)

processed = 0
skipped = 0

for root, dirs, files in os.walk(INPUT_ROOT):

    relative = os.path.relpath(root, INPUT_ROOT)

    output_folder = os.path.join(OUTPUT_ROOT, relative)

    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(files):

        if not file.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(root, file)

        try:

            y, sr = librosa.load(
                audio_path,
                sr=22050,
                mono=True
            )

            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
                n_chroma=12
            )

            if chroma.shape[1] > TARGET_FRAMES:
                chroma = chroma[:, :TARGET_FRAMES]

            elif chroma.shape[1] < TARGET_FRAMES:

                pad = TARGET_FRAMES - chroma.shape[1]

                chroma = np.pad(
                    chroma,
                    ((0, 0), (0, pad)),
                    mode="constant"
                )

            output_file = os.path.join(
                output_folder,
                os.path.splitext(file)[0] + ".npy"
            )

            np.save(output_file, chroma)

            processed += 1

        except Exception as e:

            print("\nSkipping:", audio_path)
            print(e)

            skipped += 1

print("\n" + "=" * 60)
print("Finished!")
print("=" * 60)
print("Processed :", processed)
print("Skipped   :", skipped)
print("Saved to  :", OUTPUT_ROOT)
print("=" * 60)