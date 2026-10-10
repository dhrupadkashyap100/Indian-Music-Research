
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import sys
import traceback

sys.path.append(os.path.join(PROJECT_ROOT, r"third_party\swaragram-master"))
from swaragram.core import computeSwaragram

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\multifeature_raag_optimization")
os.makedirs(OPT_DIR, exist_ok=True)

# Assets
SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")
SARAGA_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
FOLK_MASTER = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\metadata\master_dataset.csv")
TRF_AUDIO = os.path.join(PROJECT_ROOT, r"Thaat and Raga Forest (TRF) Dataset")
SARAGA_AUDIO = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\Saraga\saraga1.5_hindustani")
FOLK_AUDIO = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets")

def robust_load_audio(path, sr=24000, duration=60.0):
    """Robust audio loader trying soundfile first, then librosa fallback"""
    try:
        # Avoid librosa's audioread deprecation issues by using soundfile directly when possible
        data, samplerate = sf.read(path)
        if len(data.shape) > 1:
            data = data.mean(axis=1) # to mono
        # Resample
        if samplerate != sr:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)
        # Trim duration
        if duration:
            data = data[:int(sr * duration)]
        return data
    except Exception as e:
        try:
            # Fallback to librosa
            y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
            return y
        except Exception as e2:
            raise Exception(f"sf error: {e}, librosa error: {e2}")

def extract_features(y, sr=24000):
    # 1. Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_log = librosa.power_to_db(mel, ref=np.max)
    mel_vec = np.mean(mel_log, axis=1)
    
    # 2. Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    chroma_vec = np.mean(chroma, axis=1)
    
    # 3. Swaragram
    stft = librosa.stft(y)
    swara = computeSwaragram(stft, sampleRate=sr, winLength=2048)
    swara_vec = np.mean(swara, axis=1)
    
    return mel_vec, chroma_vec, swara_vec

def verify_splits():
    print("Verifying TRF Splits...")
    merged = pd.read_csv(SPLIT_MANIFEST).drop_duplicates('source_recording')
    
    train_recs = merged[merged['split'] == 'TRAIN']
    val_recs = merged[merged['split'] == 'VAL']
    test_recs = merged[merged['split'] == 'TEST']
    
    report = {
        "Train_Recordings": len(train_recs),
        "Val_Recordings": len(val_recs),
        "Test_Recordings": len(test_recs),
        "Train_Raags": train_recs['raag'].nunique(),
        "Val_Raags": val_recs['raag'].nunique(),
        "Test_Raags": test_recs['raag'].nunique()
    }
    print(json.dumps(report, indent=2))
    
    missing_val = set(train_recs['raag']) - set(val_recs['raag'])
    missing_test = set(train_recs['raag']) - set(test_recs['raag'])
    print(f"Raags absent from Val: {len(missing_val)}")
    print(f"Raags absent from Test: {len(missing_test)}")
    return report

def main():
    verify_splits()
    
    print("\n--- Running Technical Feature POC ---")
    
    # 1. Select 10 TRF
    df_trf = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording').head(10)
    
    # 2. Select 5 SARAGA
    df_saraga = pd.read_csv(SARAGA_MANIFEST).drop_duplicates('saraga_recording_id').head(5)
    
    # 3. Select 10 Folk
    df_folk = pd.read_csv(FOLK_MASTER).drop_duplicates('Original Song ID').head(10)
    
    poc_files = []
    
    # Resolve TRF paths
    for _, row in df_trf.iterrows():
        if os.path.exists(row['audio_path']):
            poc_files.append({"dataset": "TRF", "id": row['source_recording'], "path": row['audio_path']})
            
    # Resolve SARAGA paths
    for _, row in df_saraga.iterrows():
        p = row.get('audio_path', '')
        if isinstance(p, str) and os.path.exists(p):
            poc_files.append({"dataset": "SARAGA", "id": row['saraga_recording_id'], "path": p})
        else:
            # Try to find it
            matches = glob.glob(os.path.join(SARAGA_AUDIO, "**", f"{row['saraga_recording_id']}*.mp3"), recursive=True)
            if matches:
                poc_files.append({"dataset": "SARAGA", "id": row['saraga_recording_id'], "path": matches[0]})
                
    # Resolve Folk paths
    for _, row in df_folk.iterrows():
        p = row.get('Segment Path', '')
        if isinstance(p, str) and os.path.exists(p):
            poc_files.append({"dataset": "FOLK", "id": row['Original Song ID'], "path": p})
            
    print(f"POC Selected {len(poc_files)} files (Expected ~25).")
    
    results = []
    failures = 0
    
    import time
    
    for item in poc_files:
        start = time.time()
        res = {"dataset": item['dataset'], "id": item['id'], "status": "FAIL", "mel_dim": None, "chroma_dim": None, "swara_dim": None, "error": None}
        try:
            y = robust_load_audio(item['path'])
            mel, chroma, swara = extract_features(y)
            
            if np.isnan(mel).any() or np.isnan(chroma).any() or np.isnan(swara).any():
                res['error'] = "NaN encountered"
            else:
                res['status'] = "SUCCESS"
                res['mel_dim'] = len(mel)
                res['chroma_dim'] = len(chroma)
                res['swara_dim'] = len(swara)
                
        except Exception as e:
            failures += 1
            res['error'] = str(e)
            
        res['time_sec'] = round(time.time() - start, 2)
        print(f"[{item['dataset']}] {item['id']} -> {res['status']} ({res['time_sec']}s)")
        results.append(res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OPT_DIR, "feature_poc_report.csv"), index=False)
    
    poc_summary = {
        "Total_Tested": len(results),
        "Total_Success": len(df_res[df_res['status'] == 'SUCCESS']),
        "Total_Failures": failures
    }
    with open(os.path.join(OPT_DIR, "feature_poc_report.json"), "w") as f:
        json.dump(poc_summary, f, indent=4)
        
    print(f"POC Complete. Success: {poc_summary['Total_Success']}/{poc_summary['Total_Tested']}")

if __name__ == "__main__":
    main()
