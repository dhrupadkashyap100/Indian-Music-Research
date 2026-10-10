import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

from swaragram_wrapper import (
    load_audio, compute_stft, compute_ref_pitch, compute_swaragram, apply_log_compression
)
from tonic_utils import adjust_tonic_for_augmentation

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_ROOT = BASE_DIR / "processed_data" / "augmented_audio"
OUTPUT_ROOT = BASE_DIR / "processed_data" / "swaragrams"
CSV_PATH = BASE_DIR / "reference_dataset" / "metadata" / "saraga_metadata.csv"
TARGET_FRAMES = 938

# Global lookup dictionary for tonics to avoid reading CSV repeatedly
TONIC_LOOKUP = {}

def load_metadata():
    df = pd.read_csv(CSV_PATH)
    # Map unique_id to Tonic
    for _, row in df.iterrows():
        unique_id = str(row['unique_id'])
        TONIC_LOOKUP[unique_id] = float(row['tonic'])

# Initialize lookup globally so Windows worker processes have it populated
load_metadata()

def process_file(audio_path_str):
    audio_path = Path(audio_path_str)
    filename = audio_path.name
    parent_folder = audio_path.parent.name
    
    output_dir = OUTPUT_ROOT / parent_folder
    output_path = output_dir / audio_path.with_suffix('.npy').name
    
    # Checkpoint/Resume: skip if valid output exists
    if output_path.exists():
        try:
            # Quick check if it's not corrupted
            np.load(output_path, mmap_mode='r')
            return True, "skipped"
        except Exception:
            pass # Recompute if corrupted
            
    try:
        # Extract unique_id from filename (e.g. REC_0007_segment_001.wav -> REC_0007)
        unique_id = "_".join(filename.split("_")[:2])
        if not unique_id.startswith("REC_"):
            raise ValueError(f"Invalid filename format, could not find REC_ prefix: {filename}")
            
        # 1. Look up base tonic
        if unique_id not in TONIC_LOOKUP:
            raise ValueError(f"Tonic not found for unique_id {unique_id}")
        base_tonic = TONIC_LOOKUP[unique_id]
        
        # 2. Adjust tonic for augmentation
        adjusted_tonic = adjust_tonic_for_augmentation(base_tonic, filename)
        
        # 3. Load Audio
        x, sr = load_audio(str(audio_path), sr=22050)
        
        # 4. Compute STFT
        stft = compute_stft(x)
        
        # 5. Compute Ref Pitch & Swaragram
        ref_pitch, ref_freq = compute_ref_pitch(adjusted_tonic)
        swara = compute_swaragram(stft, ref_pitch, ref_freq)
        
        # 6. Log Compression
        swara = apply_log_compression(swara)
        
        # 7. Pad/Truncate
        if swara.shape[1] > TARGET_FRAMES:
            swara = swara[:, :TARGET_FRAMES]
        elif swara.shape[1] < TARGET_FRAMES:
            pad = TARGET_FRAMES - swara.shape[1]
            swara = np.pad(swara, ((0, 0), (0, pad)), mode="constant")
            
        # 8. Save output
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_path, swara)
        return True, None
        
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Initializing Batch Swaragram Generation...")
    print("=" * 60)
    
    print(f"Loaded tonics for {len(TONIC_LOOKUP)} unique Raagas.")
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Gather all wav files
    all_files = []
    for root, _, files in os.walk(INPUT_ROOT):
        for f in files:
            if f.lower().endswith(".wav"):
                all_files.append(os.path.join(root, f))
                
    total_files = len(all_files)
    print(f"Found {total_files} audio files to process.")
    print("=" * 60)
    
    processed = 0
    skipped = 0
    failed = 0
    failed_files = []
    
    # Using ProcessPoolExecutor for efficiency on 59k files
    # Limit max_workers to avoid memory/CPU overloading
    max_workers = min(16, (os.cpu_count() or 1) + 4)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Processing Swaragrams"):
            f = futures[future]
            try:
                success, error = future.result()
                if success and error is None:
                    processed += 1
                elif success and error == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    failed_files.append((f, error))
            except Exception as e:
                failed += 1
                failed_files.append((f, str(e)))
                
    print("\n" + "=" * 60)
    print("Batch Processing Finished!")
    print("=" * 60)
    print(f"Total Audio Processed : {total_files}")
    print(f"Successfully Generated: {processed}")
    print(f"Failed              : {failed}")
    if failed > 0:
        print("\nFailed Files Example:")
        for f, err in failed_files[:5]:
            print(f" - {Path(f).name}: {err}")
    print("=" * 60)

if __name__ == "__main__":
    main()
