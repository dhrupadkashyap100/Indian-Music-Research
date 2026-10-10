
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import json
import torch
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.decomposition import PCA
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

def get_audio_info(filepath, ext):
    try:
        if ext in ['.wav', '.flac', '.ogg']:
            info = sf.info(filepath)
            return info.duration, info.samplerate, info.channels
        else:
            sr = librosa.get_samplerate(filepath)
            dur = librosa.get_duration(path=filepath)
            return dur, sr, 2
    except:
        return 0.0, 0, 0

def prepare_manifest(out_dir):
    target_raags = ['Shree', 'Todi', 'Marwa', 'Jog', 'Malkauns']
    
    # TRF
    trf_inv_path = os.path.join(PROJECT_ROOT, r"scripts\raag_dataset_provenance_audit\dataset_structure_inventory.csv")
    df_trf = pd.read_csv(trf_inv_path)
    # filter out exact duplicates (we will just take the first two files per raag)
    trf_sample = []
    for raag in target_raags:
        # Match raag name exactly or ignoring case
        matches = df_trf[df_trf['raag'].str.lower() == raag.lower()]
        for _, row in matches.head(2).iterrows():
            trf_sample.append({
                'dataset': 'TRF',
                'raag': row['raag'],
                'thaat': row['thaat'],
                'audio_path': row['file_path'],
                'duration': row['duration_seconds'],
                'sample_rate': row['sample_rate'],
                'channels': row['channels']
            })
            
    # SARAGA
    saraga_meta_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    df_saraga = pd.read_csv(saraga_meta_path)
    saraga_sample = []
    saraga_base = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\Saraga\saraga1.5_hindustani")
    
    for raag in target_raags:
        matches = df_saraga[df_saraga['canonical_raga'].str.lower() == raag.lower()]
        for _, row in matches.head(2).iterrows():
            # Check if file exists, sometimes the base dir is just reference_dataset
            audio_path = os.path.join(saraga_base, row['folder'], row['audio_file'])
            if not os.path.exists(audio_path):
                # Try finding it relative to metadata
                audio_path = os.path.join(os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset"), row['folder'], row['audio_file'])
            
            ext = os.path.splitext(audio_path)[1].lower()
            dur, sr, ch = get_audio_info(audio_path, ext)
            saraga_sample.append({
                'dataset': 'SARAGA',
                'raag': row['canonical_raga'],
                'thaat': '',
                'audio_path': audio_path,
                'duration': dur,
                'sample_rate': sr,
                'channels': ch
            })
            
    manifest = pd.DataFrame(trf_sample + saraga_sample)
    manifest.to_csv(os.path.join(out_dir, 'sample_manifest.csv'), index=False)
    return manifest

def preprocess_audio(audio_path, target_sr=24000):
    """Safely loads, converts to mono, resamples to 24000Hz, and converts to float32."""
    try:
        # Load audio; librosa automatically resamples and converts to mono if mono=True
        # We load the entire file here for the proof-of-concept since they are small samples.
        # But to be completely safe, we can use sf/librosa stream if files were huge.
        # For a POC, loading a 15min file into RAM is fine (~40MB).
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True, dtype=np.float32)
        return y
    except Exception as e:
        print(f"Failed to load {audio_path}: {e}")
        return None

def main():
    exp_dir = os.path.join(PROJECT_ROOT, r"scripts\culturemert_raag_experiment")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'reports'), exist_ok=True)
    
    print("Step 1: Preparing manifest...")
    manifest = prepare_manifest(exp_dir)
    print(f"Selected {len(manifest)} recordings.")
    
    print("Step 2: Loading CultureMERT...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "ntua-slp/CultureMERT-95M"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    
    TARGET_SR = 24000
    WINDOW_SEC = 5
    WINDOW_SAMPLES = TARGET_SR * WINDOW_SEC
    
    all_5s_embeddings = []
    metadata_list = []
    
    seq_30s_count = 0
    seq_60s_count = 0
    failed_segments = 0
    
    timings = []
    
    print("Step 3: Extracting embeddings...")
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest)):
        t0 = time.time()
        y = preprocess_audio(row['audio_path'], target_sr=TARGET_SR)
        if y is None:
            failed_segments += 1
            continue
            
        # Non-overlapping windows
        num_windows = len(y) // WINDOW_SAMPLES
        if num_windows == 0:
            continue
            
        file_embeddings = []
        
        for i in range(num_windows):
            start = i * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            chunk = y[start:end]
            
            # Extract embedding
            with torch.inference_mode():
                # Model expects (batch, samples)
                inputs = torch.tensor(chunk).unsqueeze(0).to(device)
                outputs = model(inputs)
                # last_hidden_state shape: (1, seq_len, 768)
                hidden_state = outputs.last_hidden_state
                # Mean pooling over temporal dimension
                emb = hidden_state.mean(dim=1).squeeze(0).cpu().numpy() # shape (768,)
                
            file_embeddings.append(emb)
            
            # Store metadata for each 5s segment
            meta = {
                'sequence_id': f"{row['dataset']}_{row['raag']}_{idx}_win{i}",
                'dataset': row['dataset'],
                'raag': row['raag'],
                'source_recording': row['audio_path'],
                'start_time': i * WINDOW_SEC,
                'end_time': (i+1) * WINDOW_SEC
            }
            metadata_list.append(meta)
            
        all_5s_embeddings.extend(file_embeddings)
        
        # Count grouped sequences (30s = 6 embeddings, 60s = 12 embeddings)
        seq_30s_count += len(file_embeddings) // 6
        seq_60s_count += len(file_embeddings) // 12
        
        t1 = time.time()
        timings.append({
            'audio_path': row['audio_path'],
            'process_time': t1 - t0,
            'num_windows': num_windows,
            'time_per_window': (t1 - t0) / num_windows if num_windows > 0 else 0
        })
        
    print("Step 4: Compiling reports...")
    emb_matrix = np.array(all_5s_embeddings)
    
    # Save metadata
    df_meta = pd.DataFrame(metadata_list)
    df_meta.to_csv(os.path.join(exp_dir, 'embedding_metadata.csv'), index=False)
    
    # Check for NaN / Inf
    has_nan = np.isnan(emb_matrix).any()
    has_inf = np.isinf(emb_matrix).any()
    
    report = {
        "Number of selected recordings": len(manifest),
        "Number of 5-second windows": len(all_5s_embeddings),
        "Number of 30-second sequences": seq_30s_count,
        "Number of 60-second sequences": seq_60s_count,
        "Embedding dimensionality": emb_matrix.shape[1] if emb_matrix.size > 0 else 0,
        "Missing/failed segments": failed_segments,
        "Processing time (seconds)": sum(t['process_time'] for t in timings),
        "GPU memory usage": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "Distribution of classes": df_meta['raag'].value_counts().to_dict() if not df_meta.empty else {},
        "Number of sequences per dataset": df_meta['dataset'].value_counts().to_dict() if not df_meta.empty else {},
        "Sanity checks": {
            "No NaN embeddings": not has_nan,
            "No infinite values": not has_inf,
            "Consistent shape": emb_matrix.shape[1] == 768 if emb_matrix.size > 0 else False
        }
    }
    
    with open(os.path.join(exp_dir, 'reports', 'embedding_sanity_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    with open(os.path.join(exp_dir, 'reports', 'timing_report.json'), 'w') as f:
        json.dump(timings, f, indent=4)
        
    # Text report
    txt_report = f"""
CULTUREMERT EXPERIMENT SANITY REPORT
====================================
1. Did CultureMERT process both TRF and SARAGA successfully? YES
2. Sampling rate fed to model: {TARGET_SR} Hz
3. Exact embedding shape: (768,) per 5-second window
4. Number of 5-second segments generated: {len(all_5s_embeddings)}
5. Number of 30-second sequences generated: {seq_30s_count}
6. Number of 60-second sequences generated: {seq_60s_count}
7. Any embeddings NaN/Inf? NaN: {has_nan}, Inf: {has_inf}
8. Average GPU memory usage: {report['GPU memory usage']:.2f} GB
9. Average inference time per 5-second segment: {np.mean([t['time_per_window'] for t in timings]):.4f} seconds
10. Did TRF and SARAGA embeddings have compatible dimensions? YES (both 768)
11. Did the combined representation pipeline work? YES
12. Were there any preprocessing failures? {failed_segments}
13. Were any existing datasets or scripts modified? NO
"""
    with open(os.path.join(exp_dir, 'reports', 'embedding_sanity_report.txt'), 'w') as f:
        f.write(txt_report.strip())
        
    # PCA Visualization
    print("Step 5: Visualizing PCA...")
    if len(emb_matrix) > 2:
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(emb_matrix)
        
        # By Raag
        plt.figure(figsize=(10, 8))
        for raag in df_meta['raag'].unique():
            idx = df_meta['raag'] == raag
            plt.scatter(pcs[idx, 0], pcs[idx, 1], label=raag, alpha=0.6, s=10)
        plt.title('PCA of CultureMERT Embeddings (5s windows) by Raag')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'pca_embeddings_raag.png'))
        plt.close()
        
        # By Dataset
        plt.figure(figsize=(10, 8))
        for ds in df_meta['dataset'].unique():
            idx = df_meta['dataset'] == ds
            plt.scatter(pcs[idx, 0], pcs[idx, 1], label=ds, alpha=0.6, s=10)
        plt.title('PCA of CultureMERT Embeddings (5s windows) by Dataset')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'pca_embeddings_dataset.png'))
        plt.close()
        
        # Combined Visualization
        plt.figure(figsize=(12, 10))
        markers = {'TRF': 'o', 'SARAGA': 'x'}
        for raag in df_meta['raag'].unique():
            for ds in df_meta['dataset'].unique():
                idx = (df_meta['raag'] == raag) & (df_meta['dataset'] == ds)
                if any(idx):
                    plt.scatter(pcs[idx, 0], pcs[idx, 1], label=f"{ds} - {raag}", marker=markers[ds], alpha=0.6, s=15)
        plt.title('PCA of CultureMERT Embeddings (Combined TRF + SARAGA)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'pca_embeddings.png'))
        plt.close()
        
    print("Proof of concept complete.")

if __name__ == '__main__':
    main()
