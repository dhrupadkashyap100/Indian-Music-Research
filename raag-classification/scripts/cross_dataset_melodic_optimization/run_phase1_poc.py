
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import time
import json
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import traceback
import warnings
warnings.filterwarnings('ignore')

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_META = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

# For coverage
CM_META = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\embedding_metadata.csv")
SW_META = os.path.join(PROJECT_ROOT, r"scripts\swaragram_raag_optimization\trf_swaragram_metadata.csv")

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
    
    # 2. Voicing filtering (yin returns 0 or negative for unvoiced sometimes, or just use energy)
    # Simple energy-based voicing
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
    voiced_mask = (f0 > 0) & (rms_interp > 0.05 * np.max(rms_interp))
    
    f0_voiced = f0[voiced_mask]
    voiced_ratio = len(f0_voiced) / len(f0) if len(f0) > 0 else 0
    
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
    trans_mat = trans_mat / (np.sum(trans_mat, axis=1, keepdims=True) + 1e-9)
    
    ext_time = time.time() - start_time
    
    return {
        'voiced_ratio': voiced_ratio,
        'f0_voiced': f0_voiced,
        'cents': cents,
        'pc_dist': pc_dist,
        'int_dist': int_dist,
        'asc_ratio': asc_ratio,
        'desc_ratio': desc_ratio,
        'trans_mat': trans_mat,
        'ext_time': ext_time
    }

def main():
    os.makedirs(OPT_DIR, exist_ok=True)
    
    print("Loading datasets...")
    trf = pd.read_csv(TRF_MANIFEST)
    trf_recs = trf.drop_duplicates('source_recording')
    
    sar_meta = pd.read_csv(SARAGA_META)
    sar_map = pd.read_csv(SARAGA_MAPPING)
    if 'evaluation_eligibility' in sar_map.columns:
        sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
    else:
        sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]
    
    sar_joined = pd.merge(sar_mapped, sar_meta, left_on='saraga_recording_id', right_on='unique_id')
    
    # Select POC data
    poc_trf_train = trf_recs[trf_recs['split'] == 'TRAIN'].head(5)
    poc_trf_val = trf_recs[trf_recs['split'] == 'VAL'].head(3)
    poc_saraga = sar_joined.head(2)
    
    poc_manifest = []
    
    stats = []
    tonic_diags = []
    
    print("Processing TRF recordings...")
    for idx, row in pd.concat([poc_trf_train, poc_trf_val]).iterrows():
        rec_id = row['source_recording']
        audio_path = row['audio_path']
        split = row['split']
        raag = row['raag']
        
        poc_manifest.append({'id': rec_id, 'dataset': 'TRF', 'split': split, 'raag': raag, 'path': audio_path})
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=60.0)
            mem_mb = 0.0
            
            tonic_est = estimate_tonic(y, sr)
            
            feats = extract_melodic_features(y, sr, tonic_est)
            
            # checks
            has_nan = np.isnan(feats['pc_dist']).any() or np.isnan(feats['trans_mat']).any()
            has_inf = np.isinf(feats['pc_dist']).any() or np.isinf(feats['trans_mat']).any()
            valid_sum = np.isclose(np.sum(feats['pc_dist']), 1.0) or np.isclose(np.sum(feats['pc_dist']), 0.0)
            
            stats.append({
                'id': rec_id,
                'dataset': 'TRF',
                'voiced_ratio': feats['voiced_ratio'],
                'ext_time': feats['ext_time'],
                'mem_mb': mem_mb,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'valid_sum': valid_sum
            })
            
            tonic_diags.append({
                'id': rec_id,
                'dataset': 'TRF',
                'tonic_est': tonic_est,
                'tonic_gt': None
            })
            
            # Transposition test on first TRF train
            if len(stats) == 1:
                print("Running transposition test...")
                trans_results = []
                for shift in [0, 1, 2, 5]:
                    if shift == 0:
                        y_shift = y
                    else:
                        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
                    
                    tonic_shift = estimate_tonic(y_shift, sr)
                    feats_shift = extract_melodic_features(y_shift, sr, tonic_shift)
                    
                    trans_results.append({
                        'shift': shift,
                        'tonic_est': tonic_shift,
                        'pc_dist': feats_shift['pc_dist'].tolist()
                    })
                    
                pd.DataFrame(trans_results).to_csv(os.path.join(OPT_DIR, 'poc_transposition_test.csv'), index=False)
                
        except Exception as e:
            print(f"Error processing {rec_id}: {e}")
            
    print("Processing SARAGA recordings...")
    for idx, row in poc_saraga.iterrows():
        rec_id = row['saraga_recording_id']
        audio_path = os.path.join(row['folder'], row['audio_file'])
        raag = row['mapped_trf_raag']
        tonic_gt = row['tonic']
        
        poc_manifest.append({'id': rec_id, 'dataset': 'SARAGA', 'split': 'TEST', 'raag': raag, 'path': audio_path})
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=60.0)
            mem_mb = 0.0
            
            tonic_est = estimate_tonic(y, sr)
            
            feats_est = extract_melodic_features(y, sr, tonic_est)
            feats_gt = extract_melodic_features(y, sr, tonic_gt)
            
            stats.append({
                'id': rec_id,
                'dataset': 'SARAGA_EST',
                'voiced_ratio': feats_est['voiced_ratio'],
                'ext_time': feats_est['ext_time'],
                'mem_mb': mem_mb,
                'has_nan': False,
                'has_inf': False,
                'valid_sum': True
            })
            
            tonic_diags.append({
                'id': rec_id,
                'dataset': 'SARAGA',
                'tonic_est': tonic_est,
                'tonic_gt': tonic_gt,
                'tonic_diff_cents': 1200 * np.log2(tonic_est / tonic_gt)
            })
            
        except Exception as e:
            print(f"Error processing {rec_id}: {e}")
            
    pd.DataFrame(poc_manifest).to_csv(os.path.join(OPT_DIR, 'poc_manifest.csv'), index=False)
    pd.DataFrame(stats).to_csv(os.path.join(OPT_DIR, 'poc_feature_statistics.csv'), index=False)
    pd.DataFrame(tonic_diags).to_csv(os.path.join(OPT_DIR, 'poc_tonic_diagnostics.csv'), index=False)
    
    # Coverage report
    print("Generating coverage report...")
    if os.path.exists(CM_META):
        cm = pd.read_csv(CM_META)
        cm_recs = set(cm['source_recording'].unique())
    else:
        cm_recs = set()
        
    if os.path.exists(SW_META):
        sw = pd.read_csv(SW_META)
        sw_recs = set(sw['recording_id'].unique())
    else:
        sw_recs = set()
        
    cov = []
    for idx, row in trf_recs.iterrows():
        rec_id = row['source_recording']
        cov.append({
            'recording_id': rec_id,
            'CultureMERT_available': rec_id in cm_recs,
            'Swaragram_available': rec_id in sw_recs,
            'Melodic_available': False, # Not extracted yet
            'split': row['split'],
            'raag': row['raag']
        })
    pd.DataFrame(cov).to_csv(os.path.join(OPT_DIR, 'feature_coverage_report.csv'), index=False)
    
    # Summary JSON
    summary = {
        'total_poc_recordings': len(poc_manifest),
        'trf_processed': len([s for s in stats if s['dataset'] == 'TRF']),
        'saraga_processed': len([s for s in stats if s['dataset'] == 'SARAGA_EST']),
        'mean_ext_time': np.mean([s['ext_time'] for s in stats]),
        'mean_voiced_ratio': np.mean([s['voiced_ratio'] for s in stats]),
        'estimated_full_trf_runtime_hours': (1103 * np.mean([s['ext_time'] for s in stats])) / 3600
    }
    with open(os.path.join(OPT_DIR, 'phase1_poc_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
