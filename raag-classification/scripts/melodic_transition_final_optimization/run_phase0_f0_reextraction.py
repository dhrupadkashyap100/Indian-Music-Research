
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import time
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_META = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

FEAT_DIR = os.path.join(OPT_DIR, "features_f0")
os.makedirs(FEAT_DIR, exist_ok=True)

def estimate_tonic(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0:
        return 261.63 # Default C4
    midi_rounded = np.round(midi * 10) / 10
    most_common = Counter(midi_rounded).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def extract_f0_contour(y, sr, tonic_hz):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
    voiced_mask = (f0 > 0) & (rms_interp > 0.05 * np.max(rms_interp))
    
    # Store continuous cents sequence, with NaN for unvoiced
    cents = np.full(len(f0), np.nan, dtype=np.float32)
    cents[voiced_mask] = 1200 * np.log2(f0[voiced_mask] / tonic_hz)
    return cents

def process_recording(task):
    rec_type = task['type']
    rec_id = task['id']
    audio_path = task['audio_path']
    tonic_gt = task.get('tonic_gt', None)
    
    # Skip if already extracted
    out_f0 = os.path.join(FEAT_DIR, f"{rec_id}_f0.npy")
    out_est_f0 = os.path.join(FEAT_DIR, f"{rec_id}_EST_f0.npy")
    if rec_type == 'TRF' and os.path.exists(out_f0): return True
    if rec_type == 'SARAGA' and os.path.exists(out_est_f0): return True

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        chunk_samples = 60 * sr
        n_chunks = len(y) // chunk_samples
        
        if rec_type == 'TRF':
            f0_seqs = []
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                tonic_est = estimate_tonic(y_chunk, sr)
                cents = extract_f0_contour(y_chunk, sr, tonic_est)
                f0_seqs.append(cents)
            if f0_seqs:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_f0.npy"), np.array(f0_seqs))
                
        else: # SARAGA
            f0_est, f0_gt = [], []
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                
                # EST
                tonic_est = estimate_tonic(y_chunk, sr)
                f0_est.append(extract_f0_contour(y_chunk, sr, tonic_est))
                
                # GT
                f0_gt.append(extract_f0_contour(y_chunk, sr, tonic_gt))
                
            if f0_est:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_f0.npy"), np.array(f0_est))
            if f0_gt:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_f0.npy"), np.array(f0_gt))
                
        return True
    except Exception as e:
        print(f"Error {rec_id}: {e}")
        return False

def main():
    print("Preparing tasks...")
    trf = pd.read_csv(TRF_MANIFEST)
    trf_recs = trf.drop_duplicates('source_recording')
    
    sar_meta = pd.read_csv(SARAGA_META)
    sar_map = pd.read_csv(SARAGA_MAPPING)
    if 'evaluation_eligibility' in sar_map.columns:
        sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
    else:
        sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]
    sar_joined = pd.merge(sar_mapped, sar_meta, left_on='saraga_recording_id', right_on='unique_id')
    
    tasks = []
    for _, row in trf_recs.iterrows():
        tasks.append({
            'type': 'TRF', 'id': row['source_recording'], 'audio_path': row['audio_path']
        })
    for _, row in sar_joined.iterrows():
        tasks.append({
            'type': 'SARAGA', 'id': row['saraga_recording_id'],
            'audio_path': os.path.join(row['folder'], row['audio_file']),
            'tonic_gt': row['tonic']
        })
        
    print(f"Total tasks: {len(tasks)}")
    
    workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_recording, t): t['id'] for t in tasks}
        
        for i, fut in enumerate(as_completed(futures)):
            fut.result()
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(tasks)}")
                
    print("Done extraction.")

if __name__ == '__main__':
    main()
