
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import sys
import time

sys.path.append(os.path.join(PROJECT_ROOT, r"third_party\swaragram-master"))
from swaragram.core import computeSwaragram

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\multifeature_raag_optimization")
CACHE_DIR = os.path.join(OPT_DIR, "feature_caches")
os.makedirs(CACHE_DIR, exist_ok=True)

SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")
SARAGA_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
SARAGA_AUDIO = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\Saraga\saraga1.5_hindustani")

def robust_load_audio(path, sr=24000, duration=60.0):
    try:
        data, samplerate = sf.read(path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if samplerate != sr:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)
        if duration:
            data = data[:int(sr * duration)]
        return data
    except Exception as e:
        try:
            y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
            return y
        except Exception as e2:
            return None

def extract_features(y, sr=24000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_log = librosa.power_to_db(mel, ref=np.max)
    mel_vec = np.mean(mel_log, axis=1)
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    chroma_vec = np.mean(chroma, axis=1)
    
    stft = librosa.stft(y)
    swara = computeSwaragram(stft, sampleRate=sr, winLength=2048)
    swara_vec = np.mean(swara, axis=1)
    
    # L2 normalize
    mel_vec = mel_vec / np.linalg.norm(mel_vec)
    chroma_vec = chroma_vec / np.linalg.norm(chroma_vec)
    swara_vec = swara_vec / np.linalg.norm(swara_vec)
    
    return mel_vec, chroma_vec, swara_vec

def process_dataset(df, out_prefix, id_col, path_col, search_dir=None):
    os.makedirs(os.path.join(CACHE_DIR, out_prefix), exist_ok=True)
    count = 0
    total = len(df)
    
    for _, row in df.iterrows():
        rec_id = str(row[id_col])
        out_path = os.path.join(CACHE_DIR, out_prefix, f"{rec_id}.npy")
        if os.path.exists(out_path):
            count += 1
            continue
            
        p = row.get(path_col, '')
        audio_path = None
        if isinstance(p, str) and os.path.exists(p):
            audio_path = p
        elif search_dir:
            matches = glob.glob(os.path.join(search_dir, "**", f"{rec_id}*.mp3"), recursive=True)
            if matches:
                audio_path = matches[0]
                
        if not audio_path:
            print(f"[{out_prefix}] Missing audio for {rec_id}")
            continue
            
        y = robust_load_audio(audio_path)
        if y is None or len(y) < 24000:
            print(f"[{out_prefix}] Failed to load audio for {rec_id}")
            continue
            
        mel, chroma, swara = extract_features(y)
        
        # Save as dictionary
        np.save(out_path, {
            "mel": mel,
            "chroma": chroma,
            "swara": swara
        })
        count += 1
        if count % 50 == 0:
            print(f"[{out_prefix}] Processed {count}/{total}")
            
    print(f"[{out_prefix}] Done processing {count}/{total} files.")

def main():
    print("Extracting features for TRF...")
    df_trf = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording')
    process_dataset(df_trf, "TRF", "source_recording", "audio_path")
    
    print("Extracting features for SARAGA...")
    df_saraga = pd.read_csv(SARAGA_MANIFEST).drop_duplicates('saraga_recording_id')
    process_dataset(df_saraga, "SARAGA", "saraga_recording_id", "audio_path", search_dir=SARAGA_AUDIO)

if __name__ == "__main__":
    main()
