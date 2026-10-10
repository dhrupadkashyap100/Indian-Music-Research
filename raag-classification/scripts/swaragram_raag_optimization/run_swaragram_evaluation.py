
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

# Paths
OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\swaragram_raag_optimization")
META_CSV = os.path.join(OPT_DIR, "trf_swaragram_metadata.csv")
TRF_METADATA_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\embedding_metadata.csv")
SPLIT_MANIFEST_CSV = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")

def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def z_score_norm(x):
    std = np.std(x)
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std

# PHASE 1: VERIFICATION
def verify_inputs():
    print("=== PHASE 1: INPUT VERIFICATION ===")
    df_swara = pd.read_csv(META_CSV)
    
    total_recs = df_swara['recording_id'].nunique()
    total_seqs = len(df_swara)
    print(f"Total recordings: {total_recs} (Expected 1103)")
    print(f"Total sequences: {total_seqs} (Expected 11355)")
    
    rec_to_split = df_swara.groupby('recording_id')['split'].nunique()
    overlaps = rec_to_split[rec_to_split > 1]
    if len(overlaps) > 0:
        print(f"WARNING: {len(overlaps)} recordings in multiple splits!")
    else:
        print("Verification passed: No recordings appear in multiple splits.")
    
    has_nans = df_swara['nan_count'].sum() > 0
    has_infs = df_swara['inf_count'].sum() > 0
    print(f"Contains NaNs: {has_nans}")
    print(f"Contains Infs: {has_infs}")
    
    return df_swara

# PHASE 2: SWARAGRAM REPRESENTATION
def load_swaragrams(df_swara):
    print("\n=== PHASE 2: SWARAGRAM NORMALIZATION ===")
    mean_pcp = {}
    top20_pcp = {}
    labels = {}
    splits = {}
    
    for rec, group in df_swara.groupby('recording_id'):
        labels[rec] = group['raag'].iloc[0]
        splits[rec] = group['split'].iloc[0]
        
        rec_mean = []
        rec_top20 = []
        
        for path in group['feature_path']:
            if not os.path.exists(path): continue
            arr = np.load(path) # (12, T)
            
            # A. Mean PCP
            mean_vec = np.mean(arr, axis=1)
            rec_mean.append(mean_vec)
            
            # B. Top-20%-Frame PCP
            frame_energies = np.sum(arr, axis=0)
            threshold = np.percentile(frame_energies, 80)
            mask = frame_energies >= threshold
            if np.any(mask):
                top20_vec = np.mean(arr[:, mask], axis=1)
            else:
                top20_vec = mean_vec
            rec_top20.append(top20_vec)
            
        if len(rec_mean) > 0:
            mean_pcp[rec] = np.array(rec_mean)
            top20_pcp[rec] = np.array(rec_top20)
            
    print(f"Loaded {len(mean_pcp)} recordings.")
    return mean_pcp, top20_pcp, labels, splits

# PHASE 3: CULTUREMERT BASELINE
def load_culturemert():
    print("\n=== PHASE 3: CULTUREMERT BASELINE REPRODUCTION ===")
    df_cm = pd.read_csv(TRF_METADATA_CSV)
    df_split = pd.read_csv(SPLIT_MANIFEST_CSV)
    splits_dict = df_split.set_index('source_recording')['split'].to_dict()
    
    embs = {}
    labels = {}
    splits = {}
    window_size = 12 # 60s
    
    for rec, group in df_cm.groupby("source_recording"):
        if rec not in splits_dict: continue
        paths = group['embedding_path'].tolist()
        if len(paths) == 0: continue
        try:
            arr = np.load(paths[0])
            if len(arr) == 0: continue
            
            num_seqs = len(arr) // window_size
            if num_seqs == 0: continue
            
            seqs = []
            for i in range(num_seqs):
                chunk = arr[i*window_size:(i+1)*window_size]
                if len(chunk) == window_size:
                    seqs.append(np.mean(chunk, axis=0))
            
            if len(seqs) > 0:
                embs[rec] = np.array(seqs)
                labels[rec] = group['raag'].iloc[0]
                splits[rec] = splits_dict[rec]
        except:
            pass
            
    return embs, labels, splits

# RETRIEVAL LOGIC
def evaluate_retrieval(train_embs, train_labels, query_embs, query_labels, transform_fn=None):
    # Flatten train
    X_train = []
    y_train = []
    for rec, seqs in train_embs.items():
        X_train.extend(seqs)
        y_train.extend([train_labels[rec]] * len(seqs))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if transform_fn: X_train = transform_fn(X_train)
    X_train = l2_norm(X_train)
    
    unique_raags = np.unique(y_train)
    
    preds = []
    top3_acc = 0
    correct_count = 0
    margins_1_2 = []
    margins_1_3 = []
    correct_margins = []
    incorrect_margins = []
    top1_sims = []
    
    raw_scores = {}
    
    for q_rec, q_seqs in query_embs.items():
        if transform_fn: q_seqs = transform_fn(q_seqs)
        q_seqs = l2_norm(q_seqs)
        true_raag = query_labels[q_rec]
        
        sim_matrix = np.dot(q_seqs, X_train.T)
        raag_scores = np.zeros(len(unique_raags))
        
        for i, r in enumerate(unique_raags):
            mask = y_train == r
            if not np.any(mask): continue
            r_sims = sim_matrix[:, mask]
            q_raag_sims = np.max(r_sims, axis=1) # MAX reference per Raag
            raag_scores[i] = np.mean(q_raag_sims) # MEAN across sequences
            
        raw_scores[q_rec] = {unique_raags[i]: raag_scores[i] for i in range(len(unique_raags))}
            
        top_idx = np.argsort(raag_scores)[::-1]
        top_raags = unique_raags[top_idx]
        top_scores = raag_scores[top_idx]
        
        pred_raag = top_raags[0]
        correct = 1 if pred_raag == true_raag else 0
        correct_count += correct
        if true_raag in top_raags[:3]: top3_acc += 1
        
        m12 = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 0
        m13 = top_scores[0] - top_scores[2] if len(top_scores) > 2 else 0
        margins_1_2.append(m12)
        margins_1_3.append(m13)
        top1_sims.append(top_scores[0])
        
        if correct: correct_margins.append(m12)
        else: incorrect_margins.append(m12)
        
        preds.append({
            "recording_id": q_rec,
            "true_raag": true_raag,
            "predicted_raag": pred_raag,
            "top2_raag": top_raags[1] if len(top_raags) > 1 else None,
            "top3_raag": top_raags[2] if len(top_raags) > 2 else None,
            "score_top1": top_scores[0],
            "score_top2": top_scores[1] if len(top_scores) > 1 else 0,
            "score_top3": top_scores[2] if len(top_scores) > 2 else 0,
            "margin_1_2": m12,
            "margin_1_3": m13,
            "correct": correct
        })
        
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro'),
        "Mean_Top1_Sim": np.mean(top1_sims),
        "Median_Top1_Sim": np.median(top1_sims),
        "Mean_Margin_1_2": np.mean(margins_1_2),
        "Median_Margin_1_2": np.median(margins_1_2),
        "Mean_Margin_1_3": np.mean(margins_1_3),
        "Mean_Correct_Margin": np.mean(correct_margins) if correct_margins else 0,
        "Mean_Incorrect_Margin": np.mean(incorrect_margins) if incorrect_margins else 0,
        "Margin_Separation": (np.mean(correct_margins) - np.mean(incorrect_margins)) if correct_margins and incorrect_margins else 0
    }
    
    return metrics, preds, raw_scores

def eval_fusion(raw_a, raw_b, w_a, w_b, query_labels):
    preds = []
    top3_acc = 0
    correct_margins = []
    incorrect_margins = []
    margins_1_2 = []
    
    for q_rec in raw_a.keys():
        if q_rec not in raw_b: continue
        true_raag = query_labels[q_rec]
        
        raags = list(raw_a[q_rec].keys())
        scores_a = np.array([raw_a[q_rec][r] for r in raags])
        scores_b = np.array([raw_b[q_rec].get(r, 0) for r in raags])
        
        z_a = z_score_norm(scores_a)
        z_b = z_score_norm(scores_b)
        
        fused = w_a * z_a + w_b * z_b
        
        top_idx = np.argsort(fused)[::-1]
        top_raags = [raags[i] for i in top_idx]
        top_scores = fused[top_idx]
        
        pred = top_raags[0]
        correct = 1 if pred == true_raag else 0
        if true_raag in top_raags[:3]: top3_acc += 1
        
        m12 = top_scores[0] - top_scores[1]
        margins_1_2.append(m12)
        if correct: correct_margins.append(m12)
        else: incorrect_margins.append(m12)
        
        preds.append({"recording_id": q_rec, "true_raag": true_raag, "predicted_raag": pred})
        
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro'),
        "Mean_Margin_1_2": np.mean(margins_1_2),
        "Median_Margin_1_2": np.median(margins_1_2),
        "Margin_Separation": (np.mean(correct_margins) - np.mean(incorrect_margins)) if correct_margins and incorrect_margins else 0
    }
    return metrics, preds

def calculate_distribution_stats(raw_scores):
    all_scores = []
    margins = []
    for q_rec, s_dict in raw_scores.items():
        vals = list(s_dict.values())
        all_scores.extend(vals)
        vals_sorted = sorted(vals, reverse=True)
        margins.append(vals_sorted[0] - vals_sorted[1] if len(vals_sorted) > 1 else 0)
    
    return {
        "Min": np.min(all_scores),
        "Max": np.max(all_scores),
        "Mean": np.mean(all_scores),
        "Median": np.median(all_scores),
        "Std_Dev": np.std(all_scores),
        "Mean_Margin": np.mean(margins)
    }

def main():
    df_swara = verify_inputs()
    mean_pcp, top20_pcp, labels_sw, splits_sw = load_swaragrams(df_swara)
    cm_embs, labels_cm, splits_cm = load_culturemert()
    
    # Train / Val splits for CM
    train_cm = {r: e for r, e in cm_embs.items() if splits_cm[r] == "TRAIN"}
    train_labels_cm = {r: labels_cm[r] for r in train_cm.keys()}
    val_cm = {r: e for r, e in cm_embs.items() if splits_cm[r] == "VAL"}
    val_labels_cm = {r: labels_cm[r] for r in val_cm.keys()}
    test_cm = {r: e for r, e in cm_embs.items() if splits_cm[r] == "TEST"}
    test_labels_cm = {r: labels_cm[r] for r in test_cm.keys()}
    
    # Train / Val splits for Swara
    train_sw_mean = {r: e for r, e in mean_pcp.items() if splits_sw[r] == "TRAIN"}
    train_sw_top20 = {r: e for r, e in top20_pcp.items() if splits_sw[r] == "TRAIN"}
    train_labels_sw = {r: labels_sw[r] for r in train_sw_mean.keys()}
    
    val_sw_mean = {r: e for r, e in mean_pcp.items() if splits_sw[r] == "VAL"}
    val_sw_top20 = {r: e for r, e in top20_pcp.items() if splits_sw[r] == "VAL"}
    val_labels_sw = {r: labels_sw[r] for r in val_sw_mean.keys()}
    
    test_sw_mean = {r: e for r, e in mean_pcp.items() if splits_sw[r] == "TEST"}
    test_sw_top20 = {r: e for r, e in top20_pcp.items() if splits_sw[r] == "TEST"}
    test_labels_sw = {r: labels_sw[r] for r in test_sw_mean.keys()}
    
    # CM PCA
    flat_train_cm = np.vstack(list(train_cm.values()))
    pca128_w = PCA(n_components=128, whiten=True).fit(flat_train_cm)
    def pca128w_transform(x): return pca128_w.transform(x)
    
    print("\n--- Evaluating CultureMERT Baseline (VAL) ---")
    cm_val_metrics, cm_val_preds, cm_val_raw = evaluate_retrieval(train_cm, train_labels_cm, val_cm, val_labels_cm, transform_fn=pca128w_transform)
    print(cm_val_metrics)
    
    print("\n--- Evaluating Swaragram Mean PCP (VAL) ---")
    sw_mean_val_metrics, sw_mean_val_preds, sw_mean_val_raw = evaluate_retrieval(train_sw_mean, train_labels_sw, val_sw_mean, val_labels_sw)
    print(sw_mean_val_metrics)
    
    print("\n--- Evaluating Swaragram Top-20% PCP (VAL) ---")
    sw_top20_val_metrics, sw_top20_val_preds, sw_top20_val_raw = evaluate_retrieval(train_sw_top20, train_labels_sw, val_sw_top20, val_labels_sw)
    print(sw_top20_val_metrics)
    
    # Phase 5: Score Distribution Analysis
    dist_stats = {
        "CultureMERT": calculate_distribution_stats(cm_val_raw),
        "Swaragram_Mean": calculate_distribution_stats(sw_mean_val_raw),
        "Swaragram_Top20": calculate_distribution_stats(sw_top20_val_raw)
    }
    pd.DataFrame(dist_stats).T.to_csv(os.path.join(OPT_DIR, "representation_score_distribution.csv"))
    
    # Pick the best Swaragram method for fusion & final evaluation
    sw_best_raw = sw_mean_val_raw if sw_mean_val_metrics["Accuracy"] >= sw_top20_val_metrics["Accuracy"] else sw_top20_val_raw
    sw_best_name = "Swaragram_Mean" if sw_mean_val_metrics["Accuracy"] >= sw_top20_val_metrics["Accuracy"] else "Swaragram_Top20"
    
    # Phase 8: Fusion
    print("\n--- Evaluating Fusion (VAL) ---")
    fusion_results = []
    configs = [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]
    for wa, wb in configs:
        fm, _ = eval_fusion(cm_val_raw, sw_best_raw, wa, wb, val_labels_cm)
        fusion_results.append({"Method": f"Fusion_CM{int(wa*100)}_SW{int(wb*100)}", **fm})
        print(f"Fusion {wa}/{wb}: Acc={fm['Accuracy']:.3f}, F1={fm['Macro_F1']:.3f}")
        
    pd.DataFrame(fusion_results).to_csv(os.path.join(OPT_DIR, "fusion_comparison.csv"), index=False)
    
    # Phase 9: Model Selection Hierarchy
    all_methods = [
        {"Method": "CultureMERT_Baseline", **cm_val_metrics},
        {"Method": "Swaragram_Mean", **sw_mean_val_metrics},
        {"Method": "Swaragram_Top20", **sw_top20_val_metrics}
    ] + fusion_results
    
    df_methods = pd.DataFrame(all_methods)
    df_methods.to_csv(os.path.join(OPT_DIR, "swaragram_raag_metrics.csv"), index=False)
    
    # Determine best
    best_method = df_methods.sort_values(by=["Accuracy", "Macro_F1", "Margin_Separation"], ascending=[False, False, False]).iloc[0].to_dict()
    
    with open(os.path.join(OPT_DIR, "method_selection.json"), "w") as f:
        json.dump(best_method, f, indent=4)
        
    # Phase 10: TEST EVALUATION
    if "CultureMERT" in best_method["Method"] and "Fusion" not in best_method["Method"]:
        print("\n--- Evaluating CultureMERT on TEST ---")
        test_metrics, test_preds, _ = evaluate_retrieval(train_cm, train_labels_cm, test_cm, test_labels_cm, transform_fn=pca128w_transform)
    elif "Swaragram_Mean" in best_method["Method"]:
        print("\n--- Evaluating Swaragram_Mean on TEST ---")
        test_metrics, test_preds, _ = evaluate_retrieval(train_sw_mean, train_labels_sw, test_sw_mean, test_labels_sw)
    elif "Swaragram_Top20" in best_method["Method"]:
        print("\n--- Evaluating Swaragram_Top20 on TEST ---")
        test_metrics, test_preds, _ = evaluate_retrieval(train_sw_top20, train_labels_sw, test_sw_top20, test_labels_sw)
    elif "Fusion" in best_method["Method"]:
        print("\n--- Evaluating Fusion on TEST ---")
        wa = int(best_method["Method"].split("_")[1].replace("CM", "")) / 100.0
        wb = int(best_method["Method"].split("_")[2].replace("SW", "")) / 100.0
        
        # We need raw scores for test
        _, _, cm_test_raw = evaluate_retrieval(train_cm, train_labels_cm, test_cm, test_labels_cm, transform_fn=pca128w_transform)
        
        if sw_best_name == "Swaragram_Mean":
            _, _, sw_test_raw = evaluate_retrieval(train_sw_mean, train_labels_sw, test_sw_mean, test_labels_sw)
        else:
            _, _, sw_test_raw = evaluate_retrieval(train_sw_top20, train_labels_sw, test_sw_top20, test_labels_sw)
            
        test_metrics, test_preds = eval_fusion(cm_test_raw, sw_test_raw, wa, wb, test_labels_cm)
    
    print(f"TEST METRICS: {test_metrics}")
    
    # Save outputs
    pd.DataFrame(test_preds).to_csv(os.path.join(OPT_DIR, "swaragram_validation_predictions.csv"), index=False)
    
    # Write summary
    md = f"""# Swaragram Raag Retrieval Evaluation Report

## 1. Does Swaragram outperform CultureMERT?
- **CultureMERT Val Accuracy:** {cm_val_metrics['Accuracy']:.4f}
- **Swaragram Mean Val Accuracy:** {sw_mean_val_metrics['Accuracy']:.4f}
- **Swaragram Top20 Val Accuracy:** {sw_top20_val_metrics['Accuracy']:.4f}

## 2. Does Top-20%-frame pooling outperform simple mean pooling?
- Difference in Accuracy: {sw_top20_val_metrics['Accuracy'] - sw_mean_val_metrics['Accuracy']:.4f}

## 3. Does Swaragram improve Top-1 accuracy?
See section 1.

## 4. Does it improve Top-3?
- **CultureMERT Top-3:** {cm_val_metrics['Top3_Accuracy']:.4f}
- **Swaragram Top-3:** {sw_mean_val_metrics['Top3_Accuracy']:.4f}

## 5. Does it improve Macro F1?
- **CultureMERT Macro F1:** {cm_val_metrics['Macro_F1']:.4f}
- **Swaragram Macro F1:** {sw_mean_val_metrics['Macro_F1']:.4f}

## 6. Does it improve Margin_1_2?
- **CultureMERT Margin_1_2:** {cm_val_metrics['Mean_Margin_1_2']:.4f}
- **Swaragram Margin_1_2:** {sw_mean_val_metrics['Mean_Margin_1_2']:.4f}

## 7. Does it improve correct-vs-incorrect margin separation?
- **CultureMERT Separation:** {cm_val_metrics['Margin_Separation']:.4f}
- **Swaragram Separation:** {sw_mean_val_metrics['Margin_Separation']:.4f}

## 8. Does fusion outperform CultureMERT?
- **Best Fusion Accuracy:** {max(fm['Accuracy'] for fm in fusion_results):.4f}

## 9. Does the improvement survive the TRF Test?
- **Test Accuracy (using {best_method['Method']}):** {test_metrics['Accuracy']:.4f}

## 10. Is Swaragram actually useful enough to justify using it for Folk inference?
Based on the metrics above, the final selected model is **{best_method['Method']}**.
"""
    with open(os.path.join(OPT_DIR, "swaragram_evaluation_report.md"), "w") as f:
        f.write(md)
        
    print("Done.")

if __name__ == '__main__':
    main()
