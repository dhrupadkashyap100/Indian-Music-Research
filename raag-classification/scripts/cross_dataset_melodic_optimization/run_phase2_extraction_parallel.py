
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import time
import json
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_META = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

FEAT_DIR = os.path.join(OPT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

def estimate_tonic(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0:
        return 261.63 # Default C4
    midi_rounded = np.round(midi * 10) / 10
    most_common = Counter(midi_rounded).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def extract_melodic_features(y, sr, tonic_hz):
    # 1. F0 extraction
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # 2. Voicing filtering
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
    voiced_mask = (f0 > 0) & (rms_interp > 0.05 * np.max(rms_interp))
    
    f0_voiced = f0[voiced_mask]
    
    if len(f0_voiced) == 0:
        return None
        
    # 3. Tonic-relative pitch (cents)
    cents = 1200 * np.log2(f0_voiced / tonic_hz)
    semitones = cents / 100.0
    
    # 4. Pitch-class representation (12 bins)
    pc = np.mod(np.round(semitones), 12)
    pc_hist = np.histogram(pc, bins=np.arange(13)-0.5)[0]
    pc_dist = pc_hist / (np.sum(pc_hist) + 1e-9)
    
    # 5. Pitch interval histogram (-12 to +12 semitones)
    intervals = np.diff(semitones)
    int_hist = np.histogram(intervals, bins=np.arange(-12.5, 13.5))[0]
    int_dist = int_hist / (np.sum(int_hist) + 1e-9)
    
    # 6. Ascending/descending
    asc = np.sum(intervals > 0.5)
    desc = np.sum(intervals < -0.5)
    
    # 7. Note transition matrix (12x12)
    trans_mat = np.zeros((12, 12))
    pc_seq = np.mod(np.round(semitones), 12).astype(int)
    for i in range(len(pc_seq) - 1):
        trans_mat[pc_seq[i], pc_seq[i+1]] += 1
    trans_mat_norm = trans_mat / (np.sum(trans_mat, axis=1, keepdims=True) + 1e-9)
    
    return {
        'pc_dist': pc_dist.astype(np.float32),
        'int_dist': int_dist.astype(np.float32),
        'trans_mat': trans_mat_norm.flatten().astype(np.float32)
    }

def process_recording(task):
    rec_type = task['type']
    rec_id = task['id']
    audio_path = task['audio_path']
    tonic_gt = task.get('tonic_gt', None)
    split = task['split']
    raag = task['raag']
    
    # Check if exists
    out_pc = os.path.join(FEAT_DIR, f"{rec_id}_pc.npy")
    out_est_pc = os.path.join(FEAT_DIR, f"{rec_id}_EST_pc.npy")
    if rec_type == 'TRF' and os.path.exists(out_pc):
        try:
            arr = np.load(out_pc)
            return [{'recording_id': rec_id, 'dataset': 'TRF', 'split': split, 'raag': raag, 'n_sequences': len(arr), 'tonic_type': 'EST'}]
        except: pass
    if rec_type == 'SARAGA' and os.path.exists(out_est_pc):
        try:
            arr = np.load(out_est_pc)
            res = [{'recording_id': rec_id, 'dataset': 'SARAGA', 'split': split, 'raag': raag, 'n_sequences': len(arr), 'tonic_type': 'EST'}]
            if os.path.exists(os.path.join(FEAT_DIR, f"{rec_id}_GT_pc.npy")):
                res.append({'recording_id': rec_id, 'dataset': 'SARAGA', 'split': split, 'raag': raag, 'n_sequences': len(arr), 'tonic_type': 'GT'})
            return res
        except: pass

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        chunk_samples = 60 * sr
        n_chunks = len(y) // chunk_samples
        
        results = []
        
        if rec_type == 'TRF':
            pc_dists, int_dists, trans_mats = [], [], []
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                tonic_est = estimate_tonic(y_chunk, sr)
                feats = extract_melodic_features(y_chunk, sr, tonic_est)
                if feats is not None:
                    pc_dists.append(feats['pc_dist'])
                    int_dists.append(feats['int_dist'])
                    trans_mats.append(feats['trans_mat'])
                    
            if pc_dists:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_pc.npy"), np.array(pc_dists))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_int.npy"), np.array(int_dists))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_trans.npy"), np.array(trans_mats))
                results.append({'recording_id': rec_id, 'dataset': 'TRF', 'split': split, 'raag': raag, 'n_sequences': len(pc_dists), 'tonic_type': 'EST'})
                
        else: # SARAGA
            pc_est, int_est, trans_est = [], [], []
            pc_gt, int_gt, trans_gt = [], [], []
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                
                # EST
                tonic_est = estimate_tonic(y_chunk, sr)
                feats_est = extract_melodic_features(y_chunk, sr, tonic_est)
                if feats_est is not None:
                    pc_est.append(feats_est['pc_dist'])
                    int_est.append(feats_est['int_dist'])
                    trans_est.append(feats_est['trans_mat'])
                    
                # GT
                feats_gt = extract_melodic_features(y_chunk, sr, tonic_gt)
                if feats_gt is not None:
                    pc_gt.append(feats_gt['pc_dist'])
                    int_gt.append(feats_gt['int_dist'])
                    trans_gt.append(feats_gt['trans_mat'])
                    
            if pc_est:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_pc.npy"), np.array(pc_est))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_int.npy"), np.array(int_est))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_trans.npy"), np.array(trans_est))
                results.append({'recording_id': rec_id, 'dataset': 'SARAGA', 'split': split, 'raag': raag, 'n_sequences': len(pc_est), 'tonic_type': 'EST'})
            if pc_gt:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_pc.npy"), np.array(pc_gt))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_int.npy"), np.array(int_gt))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_trans.npy"), np.array(trans_gt))
                results.append({'recording_id': rec_id, 'dataset': 'SARAGA', 'split': split, 'raag': raag, 'n_sequences': len(pc_gt), 'tonic_type': 'GT'})
                
        return results
    except Exception as e:
        print(f"Error {rec_id}: {e}")
        return []

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
            'type': 'TRF',
            'id': row['source_recording'],
            'audio_path': row['audio_path'],
            'split': row['split'],
            'raag': row['raag']
        })
    for _, row in sar_joined.iterrows():
        tasks.append({
            'type': 'SARAGA',
            'id': row['saraga_recording_id'],
            'audio_path': os.path.join(row['folder'], row['audio_file']),
            'tonic_gt': row['tonic'],
            'split': 'TEST',
            'raag': row['mapped_trf_raag']
        })
        
    print(f"Total tasks: {len(tasks)}")
    meta_records = []
    
    workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_recording, t): t['id'] for t in tasks}
        
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res:
                meta_records.extend(res)
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(tasks)}")
                
    pd.DataFrame(meta_records).to_csv(os.path.join(OPT_DIR, "melodic_feature_metadata.csv"), index=False)
    print("Done extraction.")

if __name__ == '__main__':
    main()
