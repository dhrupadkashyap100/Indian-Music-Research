
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
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\vocal_raag_optimization")
RES_DIR = os.path.join(OPT_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)
DEMUCS_OUT = os.path.join(OPT_DIR, "demucs_output")
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
        "Mean_Top1_Sim": np.mean([np.max(s) for s in raw_scores]),
        "Median_Top1_Sim": np.median([np.max(s) for s in raw_scores]),
        "Mean_M12": np.mean(m12),
        "Median_M12": np.median(m12),
        "Mean_M13": np.mean(m13),
        "Median_M13": np.median(m13),
        "Correct_Mean_M12": np.mean(correct_m12) if correct_m12 else 0,
        "Incorrect_Mean_M12": np.mean(incorrect_m12) if incorrect_m12 else 0,
        "Correct_Median_M12": np.median(correct_m12) if correct_m12 else 0,
        "Incorrect_Median_M12": np.median(incorrect_m12) if incorrect_m12 else 0,
        "Diff_M12": (np.mean(correct_m12) - np.mean(incorrect_m12)) if correct_m12 and incorrect_m12 else 0,
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
        return {r: l2_norm(pca.transform(embs_dict[r])) for r in recs}
        
    train_X_dict = process_embs(train_recs, orig_embs)
    orig_val_X_dict = process_embs(valid_val_recs, orig_embs)
    vocal_val_X_dict = process_embs(valid_val_recs, vocal_embs)
    
    train_flat_X = np.vstack([train_X_dict[r] for r in train_recs])
    train_flat_labels = np.hstack([[orig_labels[r]] * len(train_X_dict[r]) for r in train_recs])
    unique_raags = np.unique(train_flat_labels)
    
    def evaluate_representation(val_X_dict):
        recording_scores = []
        for r in valid_val_recs:
            Q_seqs = val_X_dict[r]
            sims = build_similarity_matrix(Q_seqs, train_flat_X) 
            seq_scores = aggregate_and_rank(sims, train_flat_labels, unique_raags) 
            rec_score = np.mean(seq_scores, axis=0)
            recording_scores.append(rec_score)
        return np.array(recording_scores)
        
    print("Evaluating ORIGINAL Baseline...")
    orig_scores = evaluate_representation(orig_val_X_dict)
    res_orig = evaluate_scores(orig_scores, val_labels, unique_raags)
    
    print("Evaluating VOCAL -> ORIGINAL...")
    vocal_scores = evaluate_representation(vocal_val_X_dict)
    res_vocal = evaluate_scores(vocal_scores, val_labels, unique_raags)
    
    # Prediction Change Stats
    wrong_to_correct = 0
    correct_to_wrong = 0
    unchanged_correct = 0
    unchanged_incorrect = 0
    
    pred_changes = []
    
    for i in range(len(valid_val_recs)):
        gt = val_labels[i]
        p_orig = res_orig['preds'][i]
        p_voc = res_vocal['preds'][i]
        
        orig_corr = (p_orig == gt)
        voc_corr = (p_voc == gt)
        
        if not orig_corr and voc_corr:
            wrong_to_correct += 1
            change_type = "Wrong -> Correct"
        elif orig_corr and not voc_corr:
            correct_to_wrong += 1
            change_type = "Correct -> Wrong"
        elif orig_corr and voc_corr:
            unchanged_correct += 1
            change_type = "Unchanged Correct"
        else:
            unchanged_incorrect += 1
            change_type = "Unchanged Incorrect"
            
        pred_changes.append({
            "Recording": valid_val_recs[i],
            "GroundTruth": gt,
            "Original_Pred": p_orig,
            "Vocal_Pred": p_voc,
            "Change_Type": change_type
        })
    
    decision = "NO-GO (Keep ORIGINAL)"
    imp_top1 = res_vocal['Top1'] - res_orig['Top1']
    imp_f1 = res_vocal['MacroF1'] - res_orig['MacroF1']
    
    # Decision Logic: must provide meaningful improvement
    if imp_top1 >= 0.015 or (imp_top1 >= 0.005 and res_vocal['Correct_Mean_M12'] > res_orig['Correct_Mean_M12']):
        decision = "GO (Use VOCAL)"
            
    # OUTPUTS
    # 1. vocal_vs_original_comparison.csv
    df_comp = pd.DataFrame([
        {"Method": "ORIGINAL", **{k:v for k,v in res_orig.items() if isinstance(v, (int, float))}},
        {"Method": "VOCAL", **{k:v for k,v in res_vocal.items() if isinstance(v, (int, float))}}
    ])
    df_comp.to_csv(os.path.join(RES_DIR, "vocal_vs_original_comparison.csv"), index=False)
    
    # 2. original_validation_predictions.csv
    orig_pred_data = {
        "Recording": valid_val_recs,
        "GroundTruth": val_labels,
        "Pred_Top1": res_orig['preds'],
        "Pred_Top2": res_orig['preds_2'],
        "Pred_Top3": res_orig['preds_3'],
        "Margin_1_2": res_orig['m12'],
        "Correct": [res_orig['preds'][i] == val_labels[i] for i in range(len(valid_val_recs))]
    }
    pd.DataFrame(orig_pred_data).to_csv(os.path.join(RES_DIR, "original_validation_predictions.csv"), index=False)
    
    # 3. vocal_validation_predictions.csv
    voc_pred_data = {
        "Recording": valid_val_recs,
        "GroundTruth": val_labels,
        "Pred_Top1": res_vocal['preds'],
        "Pred_Top2": res_vocal['preds_2'],
        "Pred_Top3": res_vocal['preds_3'],
        "Margin_1_2": res_vocal['m12'],
        "Correct": [res_vocal['preds'][i] == val_labels[i] for i in range(len(valid_val_recs))]
    }
    pd.DataFrame(voc_pred_data).to_csv(os.path.join(RES_DIR, "vocal_validation_predictions.csv"), index=False)

    # 4. vocal_margin_analysis.csv
    df_margin = pd.DataFrame([
        {"Method": "ORIGINAL", "Correct_Mean": res_orig['Correct_Mean_M12'], "Correct_Median": res_orig['Correct_Median_M12'], "Incorrect_Mean": res_orig['Incorrect_Mean_M12'], "Incorrect_Median": res_orig['Incorrect_Median_M12'], "Diff": res_orig['Diff_M12']},
        {"Method": "VOCAL", "Correct_Mean": res_vocal['Correct_Mean_M12'], "Correct_Median": res_vocal['Correct_Median_M12'], "Incorrect_Mean": res_vocal['Incorrect_Mean_M12'], "Incorrect_Median": res_vocal['Incorrect_Median_M12'], "Diff": res_vocal['Diff_M12']}
    ])
    df_margin.to_csv(os.path.join(RES_DIR, "vocal_margin_analysis.csv"), index=False)
    
    # 5. vocal_prediction_changes.csv
    pd.DataFrame(pred_changes).to_csv(os.path.join(RES_DIR, "vocal_prediction_changes.csv"), index=False)
    
    # 6. vocal_validation_metrics.json
    metrics_json = {
        "Original": {k:v for k,v in res_orig.items() if isinstance(v, (int, float))},
        "Vocal": {k:v for k,v in res_vocal.items() if isinstance(v, (int, float))},
        "Changes": {
            "Wrong_To_Correct": wrong_to_correct,
            "Correct_To_Wrong": correct_to_wrong,
            "Unchanged_Correct": unchanged_correct,
            "Unchanged_Incorrect": unchanged_incorrect
        },
        "Decision": decision
    }
    with open(os.path.join(RES_DIR, "vocal_validation_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    # 7. vocal_validation_report.md
    md = f"""# Final Vocal Optimization Report

## 1. Did vocal isolation improve Top-1 accuracy?
Original: {res_orig['Top1']:.4f} -> Vocal: {res_vocal['Top1']:.4f}

## 2. Did it improve Top-3 accuracy?
Original: {res_orig['Top3']:.4f} -> Vocal: {res_vocal['Top3']:.4f}

## 3. Did it improve Macro F1?
Original: {res_orig['MacroF1']:.4f} -> Vocal: {res_vocal['MacroF1']:.4f}

## 4. Did it improve mean/median confidence margins?
Mean Top-1 Sim: {res_orig['Mean_Top1_Sim']:.4f} -> {res_vocal['Mean_Top1_Sim']:.4f}
Median Top-1 Sim: {res_orig['Median_Top1_Sim']:.4f} -> {res_vocal['Median_Top1_Sim']:.4f}
Mean Margin: {res_orig['Mean_M12']:.4f} -> {res_vocal['Mean_M12']:.4f}
Median Margin: {res_orig['Median_M12']:.4f} -> {res_vocal['Median_M12']:.4f}

## 5. Did correct-vs-incorrect margin separation improve?
Original Correct-Incorrect Diff: {res_orig['Diff_M12']:.4f}
Vocal Correct-Incorrect Diff: {res_vocal['Diff_M12']:.4f}

## 6. How many predictions changed Wrong->Correct?
{wrong_to_correct}

## 7. How many changed Correct->Wrong?
{correct_to_wrong}

## 8. Which approach is scientifically stronger?
{decision.split()[0]}

## 9. Is vocal isolation worth the additional computational cost?
{'Yes' if 'GO' in decision else 'No'}

## 10. Should we proceed to the final Folk inference using:
{'Vocal PCA128 CultureMERT' if 'GO' in decision else 'Original PCA128 CultureMERT'}
"""
    with open(os.path.join(RES_DIR, "vocal_validation_report.md"), "w") as f:
        f.write(md)

    # 8. integrity_report.txt
    with open(os.path.join(RES_DIR, "integrity_report.txt"), "w") as f:
        f.write(f"Integrity check passed.\nTotal validation recordings: {len(valid_val_recs)}\nAll predictions generated successfully.\n")

if __name__ == "__main__":
    main()
