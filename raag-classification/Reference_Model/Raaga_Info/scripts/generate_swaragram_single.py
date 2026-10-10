import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from swaragram_wrapper import (
    load_audio, compute_stft, compute_ref_pitch, compute_swaragram, apply_log_compression
)
from tonic_utils import adjust_tonic_for_augmentation

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_FRAMES = 938

def get_base_tonic_from_csv(audio_path):
    csv_path = BASE_DIR / "reference_dataset" / "metadata" / "saraga_metadata.csv"
    df = pd.read_csv(csv_path)
    
    # Extract the base folder name (Raag name)
    parent_folder = Path(audio_path).parent.name
    
    # Find matching row in metadata
    # The folder column in CSV ends with the Raag name
    matched_row = df[df['folder'].str.endswith(parent_folder)]
    if len(matched_row) > 0:
        return float(matched_row.iloc[0]['tonic'])
    raise ValueError(f"Tonic not found for folder {parent_folder}")

def generate_swaragram(audio_path):
    print(f"Processing: {audio_path}")
    
    # 1. Load Audio
    x, sr = load_audio(audio_path, sr=22050)
    
    # 2. Get Tonic
    base_tonic = get_base_tonic_from_csv(audio_path)
    filename = Path(audio_path).name
    adjusted_tonic = adjust_tonic_for_augmentation(base_tonic, filename)
    print(f"Base Tonic: {base_tonic}, Adjusted Tonic: {adjusted_tonic}")
    
    # 3. Compute STFT
    stft = compute_stft(x)
    
    # 4. Compute Ref Pitch & Swaragram
    ref_pitch, ref_freq = compute_ref_pitch(adjusted_tonic)
    swara = compute_swaragram(stft, ref_pitch, ref_freq)
    
    # 5. Log Compression
    swara = apply_log_compression(swara)
    
    # 6. Pad/Truncate
    if swara.shape[1] > TARGET_FRAMES:
        swara = swara[:, :TARGET_FRAMES]
    elif swara.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - swara.shape[1]
        swara = np.pad(swara, ((0, 0), (0, pad)), mode="constant")
        
    print(f"Final Swaragram Shape: {swara.shape}")
    
    # 7. Save output
    filename = Path(audio_path).name
    parent_folder = Path(audio_path).parent.name
    output_dir = BASE_DIR / "processed_data" / "swaragrams" / parent_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / Path(audio_path).with_suffix('.npy').name
    np.save(output_path, swara)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="")
    args = parser.parse_args()
    
    if args.audio:
        audio_file = args.audio
    else:
        # Default test file
        audio_file = str(BASE_DIR / "processed_data" / "augmented_audio" / "Aahir Bhairon" / "Aahir Bhairon.mp3_000_pitchup.wav")
        
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} does not exist.")
        sys.exit(1)
        
    generate_swaragram(audio_file)
