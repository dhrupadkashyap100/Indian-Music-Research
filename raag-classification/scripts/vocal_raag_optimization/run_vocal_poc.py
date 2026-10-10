
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import subprocess
import time

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\vocal_raag_optimization")
os.makedirs(OPT_DIR, exist_ok=True)
DEMUCS_OUT = os.path.join(OPT_DIR, "demucs_output")
os.makedirs(DEMUCS_OUT, exist_ok=True)

# Assets
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")
SARAGA_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
FOLK_MASTER = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\metadata\master_dataset.csv")
SARAGA_AUDIO = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\Saraga\saraga1.5_hindustani")

def get_poc_files():
    # TRF
    df_trf = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording').head(5)
    # SARAGA
    df_saraga = pd.read_csv(SARAGA_MANIFEST).drop_duplicates('saraga_recording_id').head(5)
    # Folk
    df_folk = pd.read_csv(FOLK_MASTER).drop_duplicates('Original Song ID').head(5)
    
    poc_files = []
    
    for _, row in df_trf.iterrows():
        p = row['audio_path']
        if os.path.exists(p):
            poc_files.append({"dataset": "TRF", "id": row['source_recording'], "path": p})
            
    for _, row in df_saraga.iterrows():
        p = row.get('audio_path', '')
        if isinstance(p, str) and os.path.exists(p):
            poc_files.append({"dataset": "SARAGA", "id": row['saraga_recording_id'], "path": p})
        else:
            matches = glob.glob(os.path.join(SARAGA_AUDIO, "**", f"{row['saraga_recording_id']}*.mp3"), recursive=True)
            if matches:
                poc_files.append({"dataset": "SARAGA", "id": row['saraga_recording_id'], "path": matches[0]})
                
    for _, row in df_folk.iterrows():
        p = row.get('Segment Path', '')
        if isinstance(p, str) and os.path.exists(p):
            poc_files.append({"dataset": "FOLK", "id": row['Original Song ID'], "path": p})
            
    return poc_files

def run_demucs_poc():
    poc_files = get_poc_files()
    print(f"Starting Demucs POC on {len(poc_files)} files...")
    
    results = []
    
    for item in poc_files:
        path = item['path']
        rec_id = item['id']
        ds = item['dataset']
        
        # 1. Get Source Audio Duration
        try:
            dur = librosa.get_duration(path=path)
        except:
            print(f"[{ds}] {rec_id} Failed to read duration.")
            results.append({"id": rec_id, "status": "FAIL", "reason": "duration_read"})
            continue
            
        print(f"[{ds}] {rec_id} - Source Duration: {dur:.2f} seconds")
        
        # 2. Run Demucs
        start_time = time.time()
        # using the demucs CLI. Output will go to DEMUCS_OUT / htdemucs / filename / vocals.wav
        cmd = ["demucs", "--out", DEMUCS_OUT, "-n", "htdemucs", path]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        runtime = time.time() - start_time
        
        if process.returncode != 0:
            print(f"[{ds}] {rec_id} Demucs crashed.")
            results.append({"id": rec_id, "status": "FAIL", "reason": "demucs_crash", "stderr": process.stderr.decode('utf-8')[:200]})
            continue
            
        # 3. Check output
        basename = os.path.splitext(os.path.basename(path))[0]
        vocal_path = os.path.join(DEMUCS_OUT, "htdemucs", basename, "vocals.wav")
        
        if not os.path.exists(vocal_path):
            print(f"[{ds}] {rec_id} Vocal stem not found.")
            results.append({"id": rec_id, "status": "FAIL", "reason": "missing_stem"})
            continue
            
        # 4. Verify Stem
        try:
            y, sr = sf.read(vocal_path)
            stem_dur = len(y) / sr
            is_silent = np.allclose(y, 0)
            has_nan = np.isnan(y).any() or np.isinf(y).any()
            
            if is_silent or has_nan or stem_dur == 0:
                results.append({"id": rec_id, "status": "FAIL", "reason": "invalid_stem_data"})
                continue
                
            status = "SUCCESS"
            reason = "OK"
        except Exception as e:
            status = "FAIL"
            reason = f"Read error: {str(e)}"
            stem_dur = 0
            
        res = {
            "dataset": ds,
            "id": rec_id,
            "status": status,
            "source_dur_min": dur / 60.0,
            "stem_dur_min": stem_dur / 60.0,
            "demucs_runtime_min": runtime / 60.0,
            "runtime_per_audio_min": (runtime / 60.0) / (dur / 60.0) if dur > 0 else 0
        }
        results.append(res)
        print(f"[{ds}] {rec_id} -> {status}. Runtime/Audio-Min: {res['runtime_per_audio_min']:.2f}x")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OPT_DIR, "vocal_poc_report.csv"), index=False)
    
    # Generate Estimates
    success_df = df_res[df_res['status'] == 'SUCCESS']
    if len(success_df) > 0:
        avg_runtime_per_min = success_df['runtime_per_audio_min'].mean()
        # TRF Val split is 223 recordings (assume ~4 mins each on average)
        # TRF Test is 245
        # SARAGA is 93 (assume ~10 mins each on average for hindustani)
        # FOLK is 328 (assume ~5 mins each)
        # TRF Total is 1103
        
        est_val = 223 * 4 * avg_runtime_per_min
        est_saraga = 93 * 10 * avg_runtime_per_min
        est_folk = 328 * 5 * avg_runtime_per_min
        est_trf_all = 1103 * 4 * avg_runtime_per_min
        
        report = f"""POC files processed: {len(results)}
Successful: {len(success_df)}/{len(results)}
Failed: {len(results) - len(success_df)}/{len(results)}

Average runtime/minute: {avg_runtime_per_min:.2f} minutes processing per 1 minute of audio

Estimated validation runtime: {est_val / 60.0:.2f} hours
Estimated SARAGA runtime: {est_saraga / 60.0:.2f} hours
Estimated 328-Folk runtime: {est_folk / 60.0:.2f} hours
Estimated worst-case 1,103-TRF runtime: {est_trf_all / 60.0:.2f} hours

Vocal stem quality: {'PASS' if len(success_df) == len(results) else 'FAIL'}
Deadline feasibility: {'PASS' if (est_val + est_saraga + est_folk)/60.0 < 24.0 else 'FAIL'}

RECOMMENDATION:
{'PROCEED' if (est_val + est_saraga + est_folk)/60.0 < 24.0 else 'ABANDON'}
"""
    else:
        report = "ALL POC FILES FAILED.\nRECOMMENDATION: ABANDON"
        
    print("\n--- FINAL REPORT ---")
    print(report)
    with open(os.path.join(OPT_DIR, "feasibility_report.txt"), "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    run_demucs_poc()
