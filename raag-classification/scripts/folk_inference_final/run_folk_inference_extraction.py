
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import librosa
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')

# Directories
WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")
FOLK_AUDIO_DIR = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\raw_audio")
FOLK_FEAT_DIR = os.path.join(WORKSPACE, "features")
os.makedirs(FOLK_FEAT_DIR, exist_ok=True)

# Frozen Model Directories
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")

# Pre-Inference Integrity Audit already done manually.
# Generating config
with open(os.path.join(WORKSPACE, "production_model_config.json"), "r") as f:
    config = json.load(f)

# ==========================================================
# PHASE 4: FOLK FEATURE CONSTRUCTION
# ==========================================================

def estimate_tonic(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0: return 261.63 # Default C4
    midi_rounded = np.round(midi * 10) / 10
    most_common = Counter(midi_rounded).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def extract_melodic_features(y, sr, tonic_hz):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
    voiced_mask = (f0 > 0) & (rms_interp > 0.05 * np.max(rms_interp))
    
    f0_voiced = f0[voiced_mask]
    if len(f0_voiced) == 0: return None
        
    cents = 1200 * np.log2(f0_voiced / tonic_hz)
    semitones = cents / 100.0
    
    pc = np.mod(np.round(semitones), 12)
    pc_hist = np.histogram(pc, bins=np.arange(13)-0.5)[0]
    pc_dist = pc_hist / (np.sum(pc_hist) + 1e-9)
    
    trans_mat = np.zeros((12, 12))
    pc_seq = pc.astype(int)
    for i in range(len(pc_seq) - 1):
        trans_mat[pc_seq[i], pc_seq[i+1]] += 1
    trans_mat_norm = trans_mat / (np.sum(trans_mat, axis=1, keepdims=True) + 1e-9)
    
    cents_full = np.full(len(f0), np.nan, dtype=np.float32)
    cents_full[voiced_mask] = 1200 * np.log2(f0[voiced_mask] / tonic_hz)
    
    return {
        'pc_dist': pc_dist.astype(np.float32),
        'trans_mat': trans_mat_norm.flatten().astype(np.float32),
        'f0_cents': cents_full
    }

print("Locating Folk audio files...")
folk_files = []
for ext in ['*.wav', '*.mp3']:
    folk_files.extend(glob.glob(os.path.join(FOLK_AUDIO_DIR, '**', ext), recursive=True))

print(f"Found {len(folk_files)} Folk recordings.")

for audio_path in folk_files:
    rec_id = os.path.splitext(os.path.basename(audio_path))[0]
    out_trans = os.path.join(FOLK_FEAT_DIR, f"{rec_id}_trans.npy")
    out_pc = os.path.join(FOLK_FEAT_DIR, f"{rec_id}_pc.npy")
    out_f0 = os.path.join(FOLK_FEAT_DIR, f"{rec_id}_f0.npy")
    
    if os.path.exists(out_trans) and os.path.exists(out_pc) and os.path.exists(out_f0):
        continue
        
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        chunk_samples = 60 * sr
        n_chunks = len(y) // chunk_samples
        
        pc_dists, trans_mats, f0_seqs = [], [], []
        for i in range(n_chunks):
            y_chunk = y[i*chunk_samples : (i+1)*chunk_samples]
            tonic_est = estimate_tonic(y_chunk, sr)
            feats = extract_melodic_features(y_chunk, sr, tonic_est)
            if feats is not None:
                pc_dists.append(feats['pc_dist'])
                trans_mats.append(feats['trans_mat'])
                f0_seqs.append(feats['f0_cents'])
                
        if pc_dists:
            np.save(out_pc, np.array(pc_dists))
            np.save(out_trans, np.array(trans_mats))
            np.save(out_f0, np.array(f0_seqs, dtype=object)) # Object to handle variable lengths or NaN arrays
    except Exception as e:
        print(f"Error extracting {rec_id}: {e}")
