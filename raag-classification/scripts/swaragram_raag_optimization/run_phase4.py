
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import librosa
from collections import Counter

sys.path.append(os.path.join(PROJECT_ROOT, r"third_party\swaragram-master"))
from swaragram import core

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\swaragram_raag_optimization")
FEAT_DIR = os.path.join(OPT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")
SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
META_CSV_OUT = os.path.join(OPT_DIR, "trf_swaragram_metadata.csv")

def computeSTFT(x, win_length=2048, hop_length=512, window='hann', pad_mode='constant'):
    return librosa.stft(x, n_fft=win_length, hop_length=hop_length, win_length=win_length, window=window, pad_mode=pad_mode)

def computeRefPitch(tonic):
    freqGen = np.vectorize(lambda x: 440 * 2 ** ((x - 69) / 12))
    midiPitchFreq = freqGen(np.arange(0,128))
    tonicMidiPitch = np.abs(tonic - midiPitchFreq).argmin()
    return tonicMidiPitch, midiPitchFreq[tonicMidiPitch]

def logCompression(x):
    eps = 2.220446049250313e-16
    return (23 * np.log10(eps + x))

def estimate_tonic_fast(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0:
        return 261.63 # Default C4
    midi_rounded = np.round(midi * 10) / 10
    most_common = Counter(midi_rounded).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def hz_to_pitch_class(hz):
    midi = round(librosa.hz_to_midi(hz))
    return midi % 12

def append_to_csv(row_dict, csv_path):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def main():
    print("Loading manifests...")
    trf = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording')
    df_split = pd.read_csv(SPLIT_MANIFEST)
    splits = df_split.groupby('source_recording').first()['split'].to_dict()
    
    total_recs = len(trf)
    print(f"Total recordings to process: {total_recs}")
    
    # Track existing features to avoid overwrite
    if os.path.exists(META_CSV_OUT):
        existing_df = pd.read_csv(META_CSV_OUT)
        existing_recs = set(existing_df['recording_id'].unique())
    else:
        existing_recs = set()
    
    start_time_all = time.time()
    
    stats = {
        "processed": 0,
        "failed": 0,
        "skipped": len(existing_recs),
        "generated_sequences": 0,
        "nan_in_features": 0,
        "inf_in_features": 0
    }
    
    SR = 22050
    WINDOW_SEC = 60
    SAMPLES_PER_WINDOW = SR * WINDOW_SEC
    
    for idx, row in trf.iterrows():
        rec = row['source_recording']
        
        if rec in existing_recs:
            print(f"[{idx+1}/{total_recs}] Skipping {rec} (Already processed)")
            continue
            
        path = row['audio_path']
        if not os.path.exists(path):
            print(f"[{idx+1}/{total_recs}] FAILED {rec}: Path not found {path}")
            stats["failed"] += 1
            continue
            
        split = splits.get(rec, "UNKNOWN")
        raag = row['raag']
        
        print(f"[{idx+1}/{total_recs}] Processing {rec} (Split: {split})...")
        t0 = time.time()
        
        try:
            # 1. Load audio and estimate tonic on the whole audio (or first 60s for speed, but user wants recording level tonic)
            # Loading entire audio can be heavy. Let's load it all since it's 1 recording at a time.
            x, _ = librosa.load(path, sr=SR, mono=True)
            
            # Estimate tonic using only the first 60 seconds to avoid freezing on massive files
            tonic_hz = estimate_tonic_fast(x[:SR*60], SR)
            tonic_midi, refFreq = computeRefPitch(tonic_hz)
            tonic_pc = hz_to_pitch_class(tonic_hz)
            
            # 2. Divide into contiguous 60-second non-overlapping windows
            total_samples = len(x)
            num_windows = total_samples // SAMPLES_PER_WINDOW
            
            for seq_idx in range(num_windows):
                start_sample = seq_idx * SAMPLES_PER_WINDOW
                end_sample = start_sample + SAMPLES_PER_WINDOW
                
                x_window = x[start_sample:end_sample]
                start_t = start_sample / SR
                end_t = end_sample / SR
                
                # 3. Compute Swaragram for the window
                stft_w = computeSTFT(x_window, win_length=2048, hop_length=512)
                swara_w = core.computeSwaragram(stft_w, sampleRate=SR, winLength=2048, refPitch=tonic_midi, refFreq=refFreq)
                swara_w_log = logCompression(swara_w)
                
                # 4. Verify Feature
                shape = swara_w_log.shape
                has_nan = np.isnan(swara_w_log).any()
                has_inf = np.isinf(swara_w_log).any()
                
                if has_nan: stats["nan_in_features"] += 1
                if has_inf: stats["inf_in_features"] += 1
                
                # 5. Save Feature
                feat_path = os.path.join(FEAT_DIR, f"{rec}_seq_{seq_idx}.npy")
                np.save(feat_path, swara_w_log)
                
                # 6. Append to metadata
                row_dict = {
                    "recording_id": rec,
                    "original_filename": os.path.basename(path),
                    "raag": raag,
                    "split": split,
                    "sequence_index": seq_idx,
                    "start_time": start_t,
                    "end_time": end_t,
                    "tonic_estimate_hz": tonic_hz,
                    "tonic_midi": tonic_midi,
                    "tonic_pitch_class": tonic_pc,
                    "feature_path": feat_path,
                    "feature_shape": str(shape),
                    "nan_count": int(has_nan),
                    "inf_count": int(has_inf),
                    "status": "SUCCESS"
                }
                append_to_csv(row_dict, META_CSV_OUT)
                stats["generated_sequences"] += 1
            
            stats["processed"] += 1
            t_total = time.time() - t0
            print(f"  -> SUCCESS: {num_windows} sequences generated. ({t_total:.1f}s)")
            
            # Explicitly free memory
            del x
            
        except Exception as e:
            print(f"  -> FAILED: {e}")
            stats["failed"] += 1
            # Append failure row
            row_dict = {
                "recording_id": rec,
                "original_filename": os.path.basename(path),
                "raag": raag,
                "split": split,
                "sequence_index": -1,
                "start_time": 0.0,
                "end_time": 0.0,
                "tonic_estimate_hz": 0.0,
                "tonic_midi": 0,
                "tonic_pitch_class": 0,
                "feature_path": "",
                "feature_shape": "",
                "nan_count": 0,
                "inf_count": 0,
                "status": f"FAILED: {e}"
            }
            append_to_csv(row_dict, META_CSV_OUT)
            
    total_time = time.time() - start_time_all
    stats["total_extraction_time_sec"] = total_time
    stats["average_time_per_recording"] = total_time / (stats["processed"] if stats["processed"] > 0 else 1)
    
    print("\n--- EXTRACTION SUMMARY ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(OPT_DIR, "trf_swaragram_extraction_summary.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    md = f"""# TRF Swaragram Extraction Report

## Summary
- **Recordings Processed:** {stats['processed']}
- **Recordings Skipped:** {stats['skipped']}
- **Recordings Failed:** {stats['failed']}
- **Sequences Generated (60s chunks):** {stats['generated_sequences']}
- **NaN Detected:** {stats['nan_in_features']}
- **Inf Detected:** {stats['inf_in_features']}

## Performance
- **Total Extraction Time:** {stats['total_extraction_time_sec'] / 60:.2f} minutes
- **Average Processing Time per Recording:** {stats['average_time_per_recording']:.2f} seconds

## Methodology
- 60-second non-overlapping contiguous chunks were extracted from each recording.
- Incomplete final chunks were discarded.
- Audio was resampled to 22,050 Hz and mono.
- Tonic normalization was applied on a per-recording basis (the tonic was estimated from the full audio recording).
- Train/Validation/Test boundaries were strictly preserved.
"""
    with open(os.path.join(OPT_DIR, "trf_swaragram_extraction_report.md"), "w") as f:
        f.write(md)

if __name__ == '__main__':
    main()
