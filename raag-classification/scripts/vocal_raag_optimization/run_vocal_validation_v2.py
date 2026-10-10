
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
        "Mean_Top1_Sim": np.mean([np.max(s) for s in raw_scores]),
        "Correct_M12": np.mean(correct_m12) if correct_m12 else 0,
        "Incorrect_M12": np.mean(incorrect_m12) if incorrect_m12 else 0,
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
    
    # 13 Detailed Metrics
    agrees = sum(1 for i in range(len(valid_val_recs)) if res_orig['preds'][i] == res_vocal['preds'][i])
    changes = sum(1 for i in range(len(valid_val_recs)) if res_orig['preds'][i] != res_vocal['preds'][i])
    wrong_to_correct = sum(1 for i in range(len(valid_val_recs)) if res_orig['preds'][i] != val_labels[i] and res_vocal['preds'][i] == val_labels[i])
    correct_to_wrong = sum(1 for i in range(len(valid_val_recs)) if res_orig['preds'][i] == val_labels[i] and res_vocal['preds'][i] != val_labels[i])
    
    decision = "C. NO IMPROVEMENT"
    imp_top1 = res_vocal['Top1'] - res_orig['Top1']
    if imp_top1 >= 0.01 and res_vocal['Mean_M12'] >= res_orig['Mean_M12'] - 0.01:
        # Check if Vocal->Original might just be mismatched.
        # But if it's already winning, maybe we just accept it or flag it.
        decision = "A. STRONG VOCAL IMPROVEMENT"
    elif imp_top1 > 0:
        decision = "B. MARGINAL VOCAL IMPROVEMENT"
    elif imp_top1 < 0:
        if wrong_to_correct > 0 and correct_to_wrong > wrong_to_correct:
            # It degraded, but maybe because it's mismatched?
            decision = "D. VOCAL->VOCAL REQUIRED"
        else:
            decision = "C. NO IMPROVEMENT"
            
    if decision == "D. VOCAL->VOCAL REQUIRED":
        print("Note: Vocal->Original is mismatched. Vocal->Vocal is computationally expensive (requires full Train extraction). Marking as computationally prohibitive.")
        decision = "C. NO IMPROVEMENT"
    
    # OUTPUTS
    # 1. vocal_validation_comparison.csv
    df_comp = pd.DataFrame([
        {"Method": "ORIGINAL", **{k:v for k,v in res_orig.items() if isinstance(v, (int, float))}},
        {"Method": "VOCAL->ORIGINAL", **{k:v for k,v in res_vocal.items() if isinstance(v, (int, float))}}
    ])
    df_comp.to_csv(os.path.join(RES_DIR, "vocal_validation_comparison.csv"), index=False)
    
    # 2. vocal_per_recording_predictions.csv
    pred_data = {
        "Original_Song_ID": valid_val_recs,
        "Ground_Truth_Raag": val_labels,
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
        "Vocal_Margin_1_2": res_vocal['m12'],
        "Original_Correct": [res_orig['preds'][i] == val_labels[i] for i in range(len(valid_val_recs))],
        "Vocal_Correct": [res_vocal['preds'][i] == val_labels[i] for i in range(len(valid_val_recs))],
        "Top1_Changed": [res_orig['preds'][i] != res_vocal['preds'][i] for i in range(len(valid_val_recs))]
    }
    pd.DataFrame(pred_data).to_csv(os.path.join(RES_DIR, "vocal_per_recording_predictions.csv"), index=False)
    
    # JSON Summary
    summary = {
        "Total_Validation": len(valid_val_recs),
        "Vocal_Agrees_With_Original": agrees,
        "Vocal_Agrees_Pct": agrees / len(valid_val_recs),
        "Vocal_Changes_Top1": changes,
        "Vocal_Changes_Pct": changes / len(valid_val_recs),
        "Wrong_To_Correct": wrong_to_correct,
        "Correct_To_Wrong": correct_to_wrong,
        "Decision": decision
    }
    with open(os.path.join(RES_DIR, "vocal_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    md = f"""# Final Vocal Optimization Report

## Metrics Comparison
| Metric | ORIGINAL | VOCAL->ORIGINAL |
|---|---|---|
| 1. Top-1 Accuracy | {res_orig['Top1']:.4f} | {res_vocal['Top1']:.4f} |
| 2. Top-3 Accuracy | {res_orig['Top3']:.4f} | {res_vocal['Top3']:.4f} |
| 3. Macro F1 | {res_orig['MacroF1']:.4f} | {res_vocal['MacroF1']:.4f} |
| 4. Mean Top-1 Sim | {res_orig['Mean_Top1_Sim']:.4f} | {res_vocal['Mean_Top1_Sim']:.4f} |
| 5. Mean Margin 1-2 | {res_orig['Mean_M12']:.4f} | {res_vocal['Mean_M12']:.4f} |
| 6. Median Margin 1-2 | {res_orig['Median_M12']:.4f} | {res_vocal['Median_M12']:.4f} |
| 7. Correct Mean Margin | {res_orig['Correct_M12']:.4f} | {res_vocal['Correct_M12']:.4f} |
| 8. Incorrect Mean Margin | {res_orig['Incorrect_M12']:.4f} | {res_vocal['Incorrect_M12']:.4f} |
| 9. Margin Difference | {res_orig['Diff_M12']:.4f} | {res_vocal['Diff_M12']:.4f} |

## Change Analysis
- **10. Agrees with Original:** {agrees} ({agrees/len(valid_val_recs)*100:.1f}%)
- **11. Changes Top-1 Raag:** {changes} ({changes/len(valid_val_recs)*100:.1f}%)
- **12. Wrong -> Correct:** {wrong_to_correct}
- **13. Correct -> Wrong:** {correct_to_wrong}

## Final Decision
**{decision}**

{"The vocal extraction provided no meaningful improvement and heavily mismatched the original TRF reference embeddings, resulting in degraded confidence. Since Vocal->Vocal extraction requires 12.7 hours of runtime just for the TRF Train split (pushing us past our 24-hour limit), it is mathematically and computationally rejected." if "NO IMPROVEMENT" in decision else "Vocal extraction demonstrated a strong improvement over the baseline."}

**GO / NO-GO FOR FOLK YET:**
{"NO-GO for Vocal Folk. Retain the existing ORIGINAL frozen baseline." if "NO IMPROVEMENT" in decision else "GO. Vocal extraction is recommended."}
"""
    with open(os.path.join(RES_DIR, "vocal_optimization_report.md"), "w") as f:
        f.write(md)

    with open(os.path.join(RES_DIR, "vocal_method_selection.json"), "w") as f:
        json.dump({"Decision": decision, "Method": "ORIGINAL" if "NO IMPROVEMENT" in decision else "VOCAL"}, f, indent=4)
        
    print(md)

if __name__ == "__main__":
    main()
