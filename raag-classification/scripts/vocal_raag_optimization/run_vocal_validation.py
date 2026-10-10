
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import time
import subprocess
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import rankdata

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\vocal_raag_optimization")
os.makedirs(OPT_DIR, exist_ok=True)
DEMUCS_OUT = os.path.join(OPT_DIR, "demucs_cache")
os.makedirs(DEMUCS_OUT, exist_ok=True)
EMB_DIR = os.path.join(OPT_DIR, "vocal_embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")

def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def extract_demucs(audio_path, out_dir):
    cmd = ["demucs", "--out", out_dir, "-n", "htdemucs", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(out_dir, "htdemucs", basename, "vocals.wav")

def load_mert_embeddings(df, window_size=12):
    embs = {}
    labels = {}
    splits = {}
    for rec, group in df.groupby("source_recording"):
        paths = group['embedding_path'].tolist()
        if not paths: continue
        try:
            arr = np.load(paths[0])
            num_seqs = len(arr) // window_size
            if num_seqs > 0:
                seqs = [np.mean(arr[i*window_size:(i+1)*window_size], axis=0) for i in range(num_seqs)]
                embs[rec] = np.array(seqs)
                labels[rec] = group['raag'].iloc[0]
                splits[rec] = group['split'].iloc[0]
        except: pass
    return embs, labels, splits

def build_similarity_matrix(query_embs, ref_embs):
    return np.dot(query_embs, ref_embs.T)

def aggregate_and_rank(sim_matrix, ref_labels, unique_raags):
    raag_scores = np.zeros((sim_matrix.shape[0], len(unique_raags)))
    for i, r in enumerate(unique_raags):
        mask = ref_labels == r
        if np.any(mask):
            r_sims = sim_matrix[:, mask]
            raag_scores[:, i] = np.max(r_sims, axis=1) # MAX ref aggregation
    return raag_scores

def evaluate_scores(raw_scores, true_labels, unique_raags):
    preds, preds_2, preds_3 = [], [], []
    m12, m13 = [], []
    correct_m12, incorrect_m12 = [], []
    
    top3_acc = 0
    for i in range(len(raw_scores)):
        s = raw_scores[i]
        idx = np.argsort(s)[::-1]
        
        p1 = unique_raags[idx[0]]
        preds.append(p1)
        preds_2.append(unique_raags[idx[1]] if len(idx)>1 else p1)
        preds_3.append(unique_raags[idx[2]] if len(idx)>2 else p1)
        
        if true_labels[i] in [unique_raags[idx[k]] for k in range(min(3, len(idx)))]:
            top3_acc += 1
            
        margin_12 = s[idx[0]] - s[idx[1]] if len(idx)>1 else 0
        margin_13 = s[idx[0]] - s[idx[2]] if len(idx)>2 else 0
        m12.append(margin_12)
        m13.append(margin_13)
        
        if p1 == true_labels[i]:
            correct_m12.append(margin_12)
        else:
            incorrect_m12.append(margin_12)
            
    return {
        "Top1": accuracy_score(true_labels, preds),
        "Top3": top3_acc / len(raw_scores),
        "MacroF1": f1_score(true_labels, preds, average='macro', zero_division=0),
        "Mean_M12": np.mean(m12),
        "Median_M12": np.median(m12),
        "Mean_M13": np.mean(m13),
        "Median_M13": np.median(m13),
        "Correct_M12": np.mean(correct_m12) if correct_m12 else 0,
        "Incorrect_M12": np.mean(incorrect_m12) if incorrect_m12 else 0,
        "preds": preds,
        "preds_2": preds_2,
        "preds_3": preds_3,
        "m12": m12,
        "scores": raw_scores
    }

def main():
    print("Loading Manifests...")
    df_split = pd.read_csv(SPLIT_MANIFEST)
    orig_embs, orig_labels, splits = load_mert_embeddings(df_split)
    
    val_recs = [r for r in orig_labels.keys() if splits[r] == 'VAL']
    train_recs = [r for r in orig_labels.keys() if splits[r] == 'TRAIN']
    
    df_manifest = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording')
    val_paths = {row['source_recording']: row['audio_path'] for _, row in df_manifest.iterrows() if row['source_recording'] in val_recs}
    
    # 1. Extract Vocal Embeddings for VAL subset
    print(f"Extracting Vocals for {len(val_paths)} validation recordings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True).to(device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    
    for i, (rec, path) in enumerate(val_paths.items()):
        out_emb = os.path.join(EMB_DIR, f"{rec}.npy")
        if os.path.exists(out_emb): continue
        
        print(f"  [{i+1}/{len(val_paths)}] Demucs on {rec}...")
        vocal_path = extract_demucs(path, DEMUCS_OUT)
        if not os.path.exists(vocal_path):
            print(f"  Failed separation for {rec}")
            continue
            
        y, _ = librosa.load(vocal_path, sr=24000, mono=True, dtype=np.float32)
        window_samples = 24000 * 5
        num_windows = len(y) // window_samples
        embs = []
        for w in range(num_windows):
            chunk = y[w*window_samples:(w+1)*window_samples]
            with torch.inference_mode():
                inputs = torch.tensor(chunk).unsqueeze(0).to(device)
                emb = model(inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                embs.append(emb)
        np.save(out_emb, np.array(embs))
        
    print("Loading Vocal Embeddings...")
    vocal_embs = {}
    for rec in val_recs:
        p = os.path.join(EMB_DIR, f"{rec}.npy")
        if os.path.exists(p):
            arr = np.load(p)
            num_seqs = len(arr) // 12
            if num_seqs > 0:
                vocal_embs[rec] = np.array([np.mean(arr[j*12:(j+1)*12], axis=0) for j in range(num_seqs)])
    
    valid_val_recs = [r for r in val_recs if r in vocal_embs and r in orig_embs]
    val_labels = np.array([orig_labels[r] for r in valid_val_recs])
    
    print("Fitting PCA128 + Whiten on TRF Train (Original)...")
    train_labels = np.array([orig_labels[r] for r in train_recs])
    train_flat = np.vstack([orig_embs[r] for r in train_recs])
    pca = PCA(n_components=128, whiten=True).fit(train_flat)
    
    def process_embs(recs, embs_dict):
        # We process sequences independently for PCA -> L2, then return the list of sequences per recording
        return {r: l2_norm(pca.transform(embs_dict[r])) for r in recs}
        
    train_X_dict = process_embs(train_recs, orig_embs)
    orig_val_X_dict = process_embs(valid_val_recs, orig_embs)
    vocal_val_X_dict = process_embs(valid_val_recs, vocal_embs)
    
    # Flatten train for reference lookup
    # To compute similarity properly: MAX similarity against all eligible TRF reference *sequences* belonging to that Raag
    # Wait, the instruction says: "MAX similarity against all eligible TRF reference sequences belonging to that Raag".
    # And "MEAN across all valid 60-second sequences".
    # So for a given query sequence Q, RaagScore = MAX(cos(Q, Refs_Raag)).
    # Then RecordingRaagScore = MEAN(RaagScore for all Qs in Recording).
    
    train_flat_X = np.vstack([train_X_dict[r] for r in train_recs])
    train_flat_labels = np.hstack([[orig_labels[r]] * len(train_X_dict[r]) for r in train_recs])
    unique_raags = np.unique(train_flat_labels)
    
    def evaluate_representation(val_X_dict):
        recording_scores = []
        for r in valid_val_recs:
            Q_seqs = val_X_dict[r]
            sims = build_similarity_matrix(Q_seqs, train_flat_X) # shape: (NumSeqs, NumRefs)
            seq_scores = aggregate_and_rank(sims, train_flat_labels, unique_raags) # shape: (NumSeqs, NumRaags)
            rec_score = np.mean(seq_scores, axis=0) # shape: (NumRaags,)
            recording_scores.append(rec_score)
        return np.array(recording_scores)
        
    print("Evaluating ORIGINAL Baseline...")
    orig_scores = evaluate_representation(orig_val_X_dict)
    res_orig = evaluate_scores(orig_scores, val_labels, unique_raags)
    
    print("Evaluating VOCAL...")
    vocal_scores = evaluate_representation(vocal_val_X_dict)
    res_vocal = evaluate_scores(vocal_scores, val_labels, unique_raags)
    
    # Make decisions
    imp_top1 = res_vocal['Top1'] - res_orig['Top1']
    imp_f1 = res_vocal['MacroF1'] - res_orig['MacroF1']
    
    # Normalize score distributions for potential fusion (rank norm as before)
    def rank_norm(s):
        n = np.zeros_like(s)
        for i in range(s.shape[0]): n[i] = rankdata(s[i]) / s.shape[1]
        return n
        
    norm_orig = rank_norm(orig_scores)
    norm_vocal = rank_norm(vocal_scores)
    
    results_list = [
        {"Method": "ORIGINAL", **{k:v for k,v in res_orig.items() if isinstance(v, (int, float))}},
        {"Method": "VOCAL", **{k:v for k,v in res_vocal.items() if isinstance(v, (int, float))}}
    ]
    
    decision = "C. Results are inconclusive"
    if imp_top1 >= 0.01 and res_vocal['Mean_M12'] >= res_orig['Mean_M12'] - 0.01:
        decision = "A. VOCAL clearly wins"
    elif imp_top1 < 0.01 or res_vocal['Mean_M12'] < res_orig['Mean_M12'] - 0.01:
        decision = "B. ORIGINAL remains better"
        
    if decision == "A. VOCAL clearly wins":
        print("Running Fusion Experiments...")
        for name, w_orig in [("Fusion 75_25", 0.75), ("Fusion 50_50", 0.50), ("Fusion 25_75", 0.25)]:
            fus_s = (w_orig * norm_orig) + ((1.0 - w_orig) * norm_vocal)
            res_fus = evaluate_scores(fus_s, val_labels, unique_raags)
            results_list.append({"Method": name, **{k:v for k,v in res_fus.items() if isinstance(v, (int, float))}})
            
    df_res = pd.DataFrame(results_list)
    df_res.to_csv(os.path.join(OPT_DIR, "vocal_validation_comparison.csv"), index=False)
    df_res.to_json(os.path.join(OPT_DIR, "vocal_validation_comparison.json"), orient="records", indent=4)
    
    # Save Predictions CSV
    pred_data = {
        "Recording_ID": valid_val_recs,
        "True_Raag": val_labels,
        "Original_Top1": res_orig['preds'],
        "Original_Top1_Score": [orig_scores[i][np.where(unique_raags == res_orig['preds'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Original_Top2": res_orig['preds_2'],
        "Original_Top2_Score": [orig_scores[i][np.where(unique_raags == res_orig['preds_2'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Original_Top3": res_orig['preds_3'],
        "Original_Top3_Score": [orig_scores[i][np.where(unique_raags == res_orig['preds_3'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Original_Margin_1_2": res_orig['m12'],
        "Vocal_Top1": res_vocal['preds'],
        "Vocal_Top1_Score": [vocal_scores[i][np.where(unique_raags == res_vocal['preds'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Vocal_Top2": res_vocal['preds_2'],
        "Vocal_Top2_Score": [vocal_scores[i][np.where(unique_raags == res_vocal['preds_2'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Vocal_Top3": res_vocal['preds_3'],
        "Vocal_Top3_Score": [vocal_scores[i][np.where(unique_raags == res_vocal['preds_3'][i])[0][0]] for i in range(len(valid_val_recs))],
        "Vocal_Margin_1_2": res_vocal['m12']
    }
    pd.DataFrame(pred_data).to_csv(os.path.join(OPT_DIR, "vocal_validation_predictions.csv"), index=False)
    
    # Markdown report
    md = f"""# Vocal Raag Optimization Validation Report

## Configuration Results

**BASELINE (Original Audio):**
- Top-1: {res_orig['Top1']:.4f}
- Top-3: {res_orig['Top3']:.4f}
- Macro F1: {res_orig['MacroF1']:.4f}
- Mean Margin: {res_orig['Mean_M12']:.4f}

**VOCAL (Demucs Stem):**
- Top-1: {res_vocal['Top1']:.4f}
- Top-3: {res_vocal['Top3']:.4f}
- Macro F1: {res_vocal['MacroF1']:.4f}
- Mean Margin: {res_vocal['Mean_M12']:.4f}

## IMPROVEMENT
- Top-1: {imp_top1:+.4f}
- Macro F1: {imp_f1:+.4f}
- Top-3: {res_vocal['Top3'] - res_orig['Top3']:+.4f}
- Margin: {res_vocal['Mean_M12'] - res_orig['Mean_M12']:+.4f}

## DECISION
{decision}

**Conclusion:**
{"The PCA128 CultureMERT baseline remains the final method. No further vocal extraction will be performed." if decision == "B. ORIGINAL remains better" else "Vocal extraction produced a meaningful improvement and justifies further external testing."}
"""
    with open(os.path.join(OPT_DIR, "vocal_validation_report.md"), "w") as f:
        f.write(md)
        
    print(md)

if __name__ == "__main__":
    main()
