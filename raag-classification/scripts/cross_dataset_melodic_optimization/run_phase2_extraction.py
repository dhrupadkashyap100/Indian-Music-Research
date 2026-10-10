
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
    start_time = time.time()
    
    # 1. F0 extraction
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # 2. Voicing filtering
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
    voiced_mask = (f0 > 0) & (rms_interp > 0.05 * np.max(rms_interp))
    
    f0_voiced = f0[voiced_mask]
    voiced_ratio = len(f0_voiced) / len(f0) if len(f0) > 0 else 0
    
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
    total_moves = asc + desc + 1e-9
    asc_ratio = asc / total_moves
    desc_ratio = desc / total_moves
    
    # 7. Note transition matrix (12x12)
    trans_mat = np.zeros((12, 12))
    pc_seq = np.mod(np.round(semitones), 12).astype(int)
    for i in range(len(pc_seq) - 1):
        trans_mat[pc_seq[i], pc_seq[i+1]] += 1
    trans_mat_norm = trans_mat / (np.sum(trans_mat, axis=1, keepdims=True) + 1e-9)
    
    ext_time = time.time() - start_time
    
    return {
        'voiced_ratio': voiced_ratio,
        'cents': cents.astype(np.float32),
        'pc_dist': pc_dist.astype(np.float32),
        'int_dist': int_dist.astype(np.float32),
        'asc_ratio': float(asc_ratio),
        'desc_ratio': float(desc_ratio),
        'trans_mat': trans_mat_norm.flatten().astype(np.float32),
        'ext_time': ext_time
    }

def main():
    print("Loading manifests...")
    trf = pd.read_csv(TRF_MANIFEST)
    trf_recs = trf.drop_duplicates('source_recording')
    
    sar_meta = pd.read_csv(SARAGA_META)
    sar_map = pd.read_csv(SARAGA_MAPPING)
    if 'evaluation_eligibility' in sar_map.columns:
        sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
    else:
        sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]
    
    sar_joined = pd.merge(sar_mapped, sar_meta, left_on='saraga_recording_id', right_on='unique_id')
    
    meta_records = []
    
    # TRF Processing
    print(f"Processing {len(trf_recs)} TRF recordings...")
    for idx, row in trf_recs.iterrows():
        rec_id = row['source_recording']
        audio_path = row['audio_path']
        
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            chunk_samples = 60 * sr
            n_chunks = len(y) // chunk_samples
            
            pc_dists = []
            int_dists = []
            trans_mats = []
            
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                tonic_est = estimate_tonic(y_chunk, sr)
                feats = extract_melodic_features(y_chunk, sr, tonic_est)
                
                if feats is not None:
                    pc_dists.append(feats['pc_dist'])
                    int_dists.append(feats['int_dist'])
                    trans_mats.append(feats['trans_mat'])
                    
            if len(pc_dists) > 0:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_pc.npy"), np.array(pc_dists))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_int.npy"), np.array(int_dists))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_trans.npy"), np.array(trans_mats))
                
                meta_records.append({
                    'recording_id': rec_id,
                    'dataset': 'TRF',
                    'split': row['split'],
                    'raag': row['raag'],
                    'n_sequences': len(pc_dists),
                    'tonic_type': 'EST'
                })
        except Exception as e:
            print(f"Failed TRF {rec_id}: {e}")
            
    # SARAGA Processing
    print(f"Processing {len(sar_joined)} SARAGA recordings...")
    for idx, row in sar_joined.iterrows():
        rec_id = row['saraga_recording_id']
        audio_path = os.path.join(row['folder'], row['audio_file'])
        tonic_gt = row['tonic']
        
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            chunk_samples = 60 * sr
            n_chunks = len(y) // chunk_samples
            
            pc_est, int_est, trans_est = [], [], []
            pc_gt, int_gt, trans_gt = [], [], []
            
            for i in range(n_chunks):
                y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
                
                tonic_est = estimate_tonic(y_chunk, sr)
                feats_est = extract_melodic_features(y_chunk, sr, tonic_est)
                if feats_est is not None:
                    pc_est.append(feats_est['pc_dist'])
                    int_est.append(feats_est['int_dist'])
                    trans_est.append(feats_est['trans_mat'])
                    
                feats_gt = extract_melodic_features(y_chunk, sr, tonic_gt)
                if feats_gt is not None:
                    pc_gt.append(feats_gt['pc_dist'])
                    int_gt.append(feats_gt['int_dist'])
                    trans_gt.append(feats_gt['trans_mat'])
                    
            if len(pc_est) > 0:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_pc.npy"), np.array(pc_est))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_int.npy"), np.array(int_est))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_EST_trans.npy"), np.array(trans_est))
                
                meta_records.append({
                    'recording_id': rec_id,
                    'dataset': 'SARAGA',
                    'split': 'TEST',
                    'raag': row['mapped_trf_raag'],
                    'n_sequences': len(pc_est),
                    'tonic_type': 'EST'
                })
                
            if len(pc_gt) > 0:
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_pc.npy"), np.array(pc_gt))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_int.npy"), np.array(int_gt))
                np.save(os.path.join(FEAT_DIR, f"{rec_id}_GT_trans.npy"), np.array(trans_gt))
                
                meta_records.append({
                    'recording_id': rec_id,
                    'dataset': 'SARAGA',
                    'split': 'TEST',
                    'raag': row['mapped_trf_raag'],
                    'n_sequences': len(pc_gt),
                    'tonic_type': 'GT'
                })
                
        except Exception as e:
            print(f"Failed SARAGA {rec_id}: {e}")
            
    pd.DataFrame(meta_records).to_csv(os.path.join(OPT_DIR, "melodic_feature_metadata.csv"), index=False)
    print("Done extraction.")

if __name__ == '__main__':
    main()
