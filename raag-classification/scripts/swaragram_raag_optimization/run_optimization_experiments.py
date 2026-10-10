
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
"""
Optimization Experiments for Raag Retrieval.
Experiments 1-3 + Combined configurations.
All evaluation is on TRF Validation only.
TRF Test is sealed until final method is frozen.
"""
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

# ============================================================
# DATA LOADING
# ============================================================

def load_swaragrams():
    print("Loading Swaragrams...")
    df = pd.read_csv(META_CSV)
    mean_pcp = {}
    labels = {}
    splits = {}
    
    for rec, group in df.groupby('recording_id'):
        labels[rec] = group['raag'].iloc[0]
        splits[rec] = group['split'].iloc[0]
        
        vecs = []
        for path in group['feature_path']:
            if not os.path.exists(path): continue
            arr = np.load(path)  # (12, T)
            vecs.append(np.mean(arr, axis=1))  # Mean PCP -> 12-D
        
        if len(vecs) > 0:
            mean_pcp[rec] = np.array(vecs)
    
    print(f"  Loaded {len(mean_pcp)} recordings.")
    return mean_pcp, labels, splits

def load_culturemert():
    print("Loading CultureMERT...")
    df_cm = pd.read_csv(TRF_METADATA_CSV)
    df_split = pd.read_csv(SPLIT_MANIFEST_CSV)
    splits_dict = df_split.set_index('source_recording')['split'].to_dict()
    
    embs = {}
    labels = {}
    splits = {}
    window_size = 12  # 60s
    
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
    
    print(f"  Loaded {len(embs)} recordings.")
    return embs, labels, splits

# ============================================================
# CORE RETRIEVAL ENGINE (parameterized)
# ============================================================

def evaluate_config(train_embs, train_labels, query_embs, query_labels,
                    transform_fn=None, ref_agg="max", temp_agg="mean",
                    top_k_temporal=None, top_k_ref=None,
                    baseline_preds=None):
    """
    Parameterized retrieval evaluation.
    
    ref_agg: "max", "top3", "top5", "top10"
    temp_agg: "mean" (applied after top_k_temporal selection)
    top_k_temporal: None=all, int=top-K, float=top-fraction (e.g. 0.2=top-20%)
    top_k_ref: None=use ref_agg, int=top-K mean references
    """
    # Flatten train
    X_train = []
    y_train = []
    for rec, seqs in train_embs.items():
        X_train.extend(seqs)
        y_train.extend([train_labels[rec]] * len(seqs))
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if transform_fn:
        X_train = transform_fn(X_train)
    X_train = l2_norm(X_train)
    
    unique_raags = np.unique(y_train)
    
    preds = []
    top3_acc = 0
    margins_1_2 = []
    margins_1_3 = []
    correct_margins = []
    incorrect_margins = []
    top1_sims = []
    raw_scores = {}
    
    for q_rec, q_seqs in query_embs.items():
        if transform_fn:
            q_seqs = transform_fn(q_seqs)
        q_seqs = l2_norm(q_seqs)
        true_raag = query_labels[q_rec]
        n_seqs = len(q_seqs)
        
        # Full similarity matrix: (n_query_seqs, n_train_seqs)
        sim_matrix = np.dot(q_seqs, X_train.T)
        
        # --- TOP-K TEMPORAL SELECTION (label-independent) ---
        if top_k_temporal is not None and n_seqs > 1:
            # Label-independent informativeness: max similarity across ALL references
            # for each query sequence (regardless of Raag)
            seq_max_sims = np.max(sim_matrix, axis=1)  # (n_seqs,)
            
            if isinstance(top_k_temporal, float):
                k = max(1, int(np.ceil(n_seqs * top_k_temporal)))
            else:
                k = min(top_k_temporal, n_seqs)
            
            top_seq_idx = np.argsort(seq_max_sims)[::-1][:k]
            sim_matrix = sim_matrix[top_seq_idx]
        
        # --- RAAG SCORING ---
        raag_scores = np.zeros(len(unique_raags))
        
        for i, r in enumerate(unique_raags):
            mask = y_train == r
            if not np.any(mask):
                continue
            r_sims = sim_matrix[:, mask]  # (selected_seqs, n_ref_for_raag)
            
            # Reference aggregation per query sequence
            if ref_agg == "max" or top_k_ref is None:
                if ref_agg == "max":
                    q_raag_sims = np.max(r_sims, axis=1)
                elif ref_agg == "top3":
                    sorted_refs = np.sort(r_sims, axis=1)[:, ::-1]
                    q_raag_sims = np.mean(sorted_refs[:, :min(3, sorted_refs.shape[1])], axis=1)
                elif ref_agg == "top5":
                    sorted_refs = np.sort(r_sims, axis=1)[:, ::-1]
                    q_raag_sims = np.mean(sorted_refs[:, :min(5, sorted_refs.shape[1])], axis=1)
                elif ref_agg == "top10":
                    sorted_refs = np.sort(r_sims, axis=1)[:, ::-1]
                    q_raag_sims = np.mean(sorted_refs[:, :min(10, sorted_refs.shape[1])], axis=1)
            else:
                # top_k_ref overrides ref_agg
                sorted_refs = np.sort(r_sims, axis=1)[:, ::-1]
                q_raag_sims = np.mean(sorted_refs[:, :min(top_k_ref, sorted_refs.shape[1])], axis=1)
            
            # Temporal aggregation
            raag_scores[i] = np.mean(q_raag_sims)
        
        raw_scores[q_rec] = {unique_raags[i]: raag_scores[i] for i in range(len(unique_raags))}
        
        top_idx = np.argsort(raag_scores)[::-1]
        top_raags = unique_raags[top_idx]
        top_scores = raag_scores[top_idx]
        
        pred_raag = top_raags[0]
        correct = 1 if pred_raag == true_raag else 0
        if true_raag in top_raags[:3]:
            top3_acc += 1
        
        m12 = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 0
        m13 = top_scores[0] - top_scores[2] if len(top_scores) > 2 else 0
        margins_1_2.append(m12)
        margins_1_3.append(m13)
        top1_sims.append(top_scores[0])
        
        if correct:
            correct_margins.append(m12)
        else:
            incorrect_margins.append(m12)
        
        preds.append({
            "recording_id": q_rec,
            "true_raag": true_raag,
            "predicted_raag": pred_raag,
            "top2_raag": top_raags[1] if len(top_raags) > 1 else None,
            "top3_raag": top_raags[2] if len(top_raags) > 2 else None,
            "score_top1": float(top_scores[0]),
            "score_top2": float(top_scores[1]) if len(top_scores) > 1 else 0,
            "score_top3": float(top_scores[2]) if len(top_scores) > 2 else 0,
            "margin_1_2": float(m12),
            "margin_1_3": float(m13),
            "correct": correct
        })
    
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    # Prediction change analysis vs baseline
    w2c = 0
    c2w = 0
    if baseline_preds is not None:
        base_map = {p["recording_id"]: p for p in baseline_preds}
        for p in preds:
            rid = p["recording_id"]
            if rid in base_map:
                bp = base_map[rid]
                if bp["correct"] == 0 and p["correct"] == 1:
                    w2c += 1
                elif bp["correct"] == 1 and p["correct"] == 0:
                    c2w += 1
    
    metrics = {
        "Top1_Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Mean_Top1_Sim": float(np.mean(top1_sims)),
        "Median_Top1_Sim": float(np.median(top1_sims)),
        "Mean_Margin_1_2": float(np.mean(margins_1_2)),
        "Median_Margin_1_2": float(np.median(margins_1_2)),
        "Mean_Margin_1_3": float(np.mean(margins_1_3)),
        "Mean_Correct_Margin": float(np.mean(correct_margins)) if correct_margins else 0,
        "Mean_Incorrect_Margin": float(np.mean(incorrect_margins)) if incorrect_margins else 0,
        "Margin_Separation": float(np.mean(correct_margins) - np.mean(incorrect_margins)) if correct_margins and incorrect_margins else 0,
        "Wrong_to_Correct": w2c,
        "Correct_to_Wrong": c2w,
        "N_Queries": len(preds)
    }
    
    return metrics, preds, raw_scores


# ============================================================
# FUSION METHODS
# ============================================================

def fusion_zscore(raw_a, raw_b, w_a, w_b, query_labels, baseline_preds=None):
    preds = []
    top3_acc = 0
    margins_1_2 = []
    correct_margins = []
    incorrect_margins = []
    w2c = 0
    c2w = 0
    base_map = {}
    if baseline_preds:
        base_map = {p["recording_id"]: p for p in baseline_preds}
    
    for q_rec in raw_a.keys():
        if q_rec not in raw_b or q_rec not in query_labels:
            continue
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
        if true_raag in top_raags[:3]:
            top3_acc += 1
        
        m12 = float(top_scores[0] - top_scores[1])
        margins_1_2.append(m12)
        if correct:
            correct_margins.append(m12)
        else:
            incorrect_margins.append(m12)
        
        if q_rec in base_map:
            bp = base_map[q_rec]
            if bp["correct"] == 0 and correct == 1: w2c += 1
            elif bp["correct"] == 1 and correct == 0: c2w += 1
        
        preds.append({
            "recording_id": q_rec,
            "true_raag": true_raag,
            "predicted_raag": pred,
            "top2_raag": top_raags[1] if len(top_raags) > 1 else None,
            "top3_raag": top_raags[2] if len(top_raags) > 2 else None,
            "score_top1": float(top_scores[0]),
            "score_top2": float(top_scores[1]) if len(top_scores) > 1 else 0,
            "margin_1_2": m12,
            "correct": correct
        })
    
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    metrics = {
        "Top1_Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Mean_Margin_1_2": float(np.mean(margins_1_2)),
        "Median_Margin_1_2": float(np.median(margins_1_2)),
        "Margin_Separation": float(np.mean(correct_margins) - np.mean(incorrect_margins)) if correct_margins and incorrect_margins else 0,
        "Wrong_to_Correct": w2c,
        "Correct_to_Wrong": c2w,
        "N_Queries": len(preds)
    }
    return metrics, preds


def fusion_rrf(raw_a, raw_b, k_param, query_labels, baseline_preds=None):
    """Reciprocal Rank Fusion: RRF_score(raag) = 1/(k+rank_a) + 1/(k+rank_b)"""
    preds = []
    top3_acc = 0
    margins_1_2 = []
    correct_margins = []
    incorrect_margins = []
    w2c = 0
    c2w = 0
    base_map = {}
    if baseline_preds:
        base_map = {p["recording_id"]: p for p in baseline_preds}
    
    for q_rec in raw_a.keys():
        if q_rec not in raw_b or q_rec not in query_labels:
            continue
        true_raag = query_labels[q_rec]
        raags = list(raw_a[q_rec].keys())
        
        scores_a = np.array([raw_a[q_rec][r] for r in raags])
        scores_b = np.array([raw_b[q_rec].get(r, 0) for r in raags])
        
        # Convert to ranks (1-indexed, rank 1 = highest score)
        ranks_a = np.argsort(np.argsort(-scores_a)) + 1
        ranks_b = np.argsort(np.argsort(-scores_b)) + 1
        
        rrf_scores = 1.0 / (k_param + ranks_a) + 1.0 / (k_param + ranks_b)
        
        top_idx = np.argsort(rrf_scores)[::-1]
        top_raags = [raags[i] for i in top_idx]
        top_scores = rrf_scores[top_idx]
        
        pred = top_raags[0]
        correct = 1 if pred == true_raag else 0
        if true_raag in top_raags[:3]:
            top3_acc += 1
        
        m12 = float(top_scores[0] - top_scores[1])
        margins_1_2.append(m12)
        if correct:
            correct_margins.append(m12)
        else:
            incorrect_margins.append(m12)
        
        if q_rec in base_map:
            bp = base_map[q_rec]
            if bp["correct"] == 0 and correct == 1: w2c += 1
            elif bp["correct"] == 1 and correct == 0: c2w += 1
        
        preds.append({
            "recording_id": q_rec,
            "true_raag": true_raag,
            "predicted_raag": pred,
            "top2_raag": top_raags[1] if len(top_raags) > 1 else None,
            "top3_raag": top_raags[2] if len(top_raags) > 2 else None,
            "score_top1": float(top_scores[0]),
            "score_top2": float(top_scores[1]) if len(top_scores) > 1 else 0,
            "margin_1_2": m12,
            "correct": correct
        })
    
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    metrics = {
        "Top1_Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Mean_Margin_1_2": float(np.mean(margins_1_2)),
        "Median_Margin_1_2": float(np.median(margins_1_2)),
        "Margin_Separation": float(np.mean(correct_margins) - np.mean(incorrect_margins)) if correct_margins and incorrect_margins else 0,
        "Wrong_to_Correct": w2c,
        "Correct_to_Wrong": c2w,
        "N_Queries": len(preds)
    }
    return metrics, preds


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    
    # Load data
    sw_pcp, labels_sw, splits_sw = load_swaragrams()
    cm_embs, labels_cm, splits_cm = load_culturemert()
    
    # Split CultureMERT
    train_cm = {r: e for r, e in cm_embs.items() if splits_cm.get(r) == "TRAIN"}
    train_labels_cm = {r: labels_cm[r] for r in train_cm}
    val_cm = {r: e for r, e in cm_embs.items() if splits_cm.get(r) == "VAL"}
    val_labels_cm = {r: labels_cm[r] for r in val_cm}
    test_cm = {r: e for r, e in cm_embs.items() if splits_cm.get(r) == "TEST"}
    test_labels_cm = {r: labels_cm[r] for r in test_cm}
    
    # Split Swaragram
    train_sw = {r: e for r, e in sw_pcp.items() if splits_sw.get(r) == "TRAIN"}
    train_labels_sw = {r: labels_sw[r] for r in train_sw}
    val_sw = {r: e for r, e in sw_pcp.items() if splits_sw.get(r) == "VAL"}
    val_labels_sw = {r: labels_sw[r] for r in val_sw}
    test_sw = {r: e for r, e in sw_pcp.items() if splits_sw.get(r) == "TEST"}
    test_labels_sw = {r: labels_sw[r] for r in test_sw}
    
    # PCA transform (fitted on TRAIN only)
    flat_train_cm = np.vstack(list(train_cm.values()))
    pca128_w = PCA(n_components=128, whiten=True).fit(flat_train_cm)
    def pca_transform(x): return pca128_w.transform(x)
    
    print(f"\nCultureMERT: TRAIN={len(train_cm)}, VAL={len(val_cm)}, TEST={len(test_cm)}")
    print(f"Swaragram:   TRAIN={len(train_sw)}, VAL={len(val_sw)}, TEST={len(test_sw)}")
    
    all_results = []
    
    # =========================================================
    # BASELINE: CultureMERT MAX+MEAN (current frozen method)
    # =========================================================
    print("\n" + "="*60)
    print("BASELINE: CultureMERT PCA128 MAX+MEAN")
    print("="*60)
    
    base_m, base_p, base_raw = evaluate_config(
        train_cm, train_labels_cm, val_cm, val_labels_cm,
        transform_fn=pca_transform, ref_agg="max", top_k_temporal=None
    )
    print(f"  Top-1: {base_m['Top1_Accuracy']:.4f}  Top-3: {base_m['Top3_Accuracy']:.4f}  F1: {base_m['Macro_F1']:.4f}  Margin: {base_m['Mean_Margin_1_2']:.4f}  Sep: {base_m['Margin_Separation']:.4f}")
    all_results.append({"Method": "CM_Baseline_MAX_MEAN", **base_m})
    
    # =========================================================
    # EXPERIMENT 1: TOP-K TEMPORAL EVIDENCE
    # =========================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: TOP-K TEMPORAL EVIDENCE")
    print("="*60)
    
    temporal_configs = [
        ("Mean_All", None),
        ("Top_3_seqs", 3),
        ("Top_5_seqs", 5),
        ("Top_10_seqs", 10),
        ("Top_20pct", 0.2),
        ("Top_30pct", 0.3),
    ]
    
    for name, tk in temporal_configs:
        m, p, _ = evaluate_config(
            train_cm, train_labels_cm, val_cm, val_labels_cm,
            transform_fn=pca_transform, ref_agg="max", top_k_temporal=tk,
            baseline_preds=base_p
        )
        print(f"  {name:15s}  Top-1: {m['Top1_Accuracy']:.4f}  Top-3: {m['Top3_Accuracy']:.4f}  F1: {m['Macro_F1']:.4f}  Margin: {m['Mean_Margin_1_2']:.4f}  Sep: {m['Margin_Separation']:.4f}  W2C: {m['Wrong_to_Correct']}  C2W: {m['Correct_to_Wrong']}")
        all_results.append({"Method": f"EXP1_Temporal_{name}", **m})
    
    # =========================================================
    # EXPERIMENT 2: REFERENCE AGGREGATION
    # =========================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: REFERENCE AGGREGATION")
    print("="*60)
    
    ref_configs = [
        ("MAX", "max"),
        ("Top3_Mean", "top3"),
        ("Top5_Mean", "top5"),
        ("Top10_Mean", "top10"),
    ]
    
    for name, ra in ref_configs:
        m, p, _ = evaluate_config(
            train_cm, train_labels_cm, val_cm, val_labels_cm,
            transform_fn=pca_transform, ref_agg=ra, top_k_temporal=None,
            baseline_preds=base_p
        )
        print(f"  {name:15s}  Top-1: {m['Top1_Accuracy']:.4f}  Top-3: {m['Top3_Accuracy']:.4f}  F1: {m['Macro_F1']:.4f}  Margin: {m['Mean_Margin_1_2']:.4f}  Sep: {m['Margin_Separation']:.4f}  W2C: {m['Wrong_to_Correct']}  C2W: {m['Correct_to_Wrong']}")
        all_results.append({"Method": f"EXP2_RefAgg_{name}", **m})
    
    # =========================================================
    # EXPERIMENT 3: FUSION METHODS
    # =========================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: FUSION METHODS")
    print("="*60)
    
    # Swaragram baseline raw scores
    sw_m, sw_p, sw_raw = evaluate_config(
        train_sw, train_labels_sw, val_sw, val_labels_sw,
        ref_agg="max", top_k_temporal=None, baseline_preds=base_p
    )
    print(f"  Swaragram Only  Top-1: {sw_m['Top1_Accuracy']:.4f}  Top-3: {sw_m['Top3_Accuracy']:.4f}  F1: {sw_m['Macro_F1']:.4f}")
    all_results.append({"Method": "Swaragram_Only", **sw_m})
    
    # Z-score fusion at various weights
    zscore_configs = [(0.75, 0.25), (0.60, 0.40), (0.50, 0.50), (0.40, 0.60), (0.25, 0.75)]
    for wa, wb in zscore_configs:
        fm, fp = fusion_zscore(base_raw, sw_raw, wa, wb, val_labels_cm, baseline_preds=base_p)
        print(f"  ZScore {wa:.0%}/{wb:.0%}  Top-1: {fm['Top1_Accuracy']:.4f}  Top-3: {fm['Top3_Accuracy']:.4f}  F1: {fm['Macro_F1']:.4f}  Margin: {fm['Mean_Margin_1_2']:.4f}  Sep: {fm['Margin_Separation']:.4f}  W2C: {fm['Wrong_to_Correct']}  C2W: {fm['Correct_to_Wrong']}")
        all_results.append({"Method": f"EXP3_ZScore_CM{int(wa*100)}_SW{int(wb*100)}", **fm})
    
    # RRF fusion at various k values
    rrf_k_values = [10, 30, 60]
    for k_val in rrf_k_values:
        fm, fp = fusion_rrf(base_raw, sw_raw, k_val, val_labels_cm, baseline_preds=base_p)
        print(f"  RRF k={k_val:3d}      Top-1: {fm['Top1_Accuracy']:.4f}  Top-3: {fm['Top3_Accuracy']:.4f}  F1: {fm['Macro_F1']:.4f}  Margin: {fm['Mean_Margin_1_2']:.4f}  Sep: {fm['Margin_Separation']:.4f}  W2C: {fm['Wrong_to_Correct']}  C2W: {fm['Correct_to_Wrong']}")
        all_results.append({"Method": f"EXP3_RRF_k{k_val}", **fm})
    
    # =========================================================
    # COMBINED EXPERIMENTS
    # =========================================================
    print("\n" + "="*60)
    print("COMBINED EXPERIMENTS")
    print("="*60)
    
    # Identify best temporal config from Experiment 1
    exp1_results = [r for r in all_results if r["Method"].startswith("EXP1_")]
    best_exp1 = max(exp1_results, key=lambda x: (x["Top1_Accuracy"], x["Macro_F1"]))
    best_temp_name = best_exp1["Method"].replace("EXP1_Temporal_", "")
    best_temp_val = dict(temporal_configs).get(best_temp_name)
    print(f"Best Temporal: {best_temp_name} (Top-1: {best_exp1['Top1_Accuracy']:.4f})")
    
    # Identify best reference config from Experiment 2
    exp2_results = [r for r in all_results if r["Method"].startswith("EXP2_")]
    best_exp2 = max(exp2_results, key=lambda x: (x["Top1_Accuracy"], x["Macro_F1"]))
    best_ref_name = best_exp2["Method"].replace("EXP2_RefAgg_", "")
    best_ref_val = dict(ref_configs).get(best_ref_name, "max")
    print(f"Best Reference: {best_ref_name} (Top-1: {best_exp2['Top1_Accuracy']:.4f})")
    
    # Combo 1: Best temporal + MAX reference
    m, p, raw1 = evaluate_config(
        train_cm, train_labels_cm, val_cm, val_labels_cm,
        transform_fn=pca_transform, ref_agg="max", top_k_temporal=best_temp_val,
        baseline_preds=base_p
    )
    print(f"  Combo1 (BestTemp+MAX)       Top-1: {m['Top1_Accuracy']:.4f}  Top-3: {m['Top3_Accuracy']:.4f}  F1: {m['Macro_F1']:.4f}  Sep: {m['Margin_Separation']:.4f}")
    all_results.append({"Method": "COMBO_BestTemp_MAX", **m})
    
    # Combo 2: Mean temporal + Best reference
    m, p, raw2 = evaluate_config(
        train_cm, train_labels_cm, val_cm, val_labels_cm,
        transform_fn=pca_transform, ref_agg=best_ref_val, top_k_temporal=None,
        baseline_preds=base_p
    )
    print(f"  Combo2 (Mean+BestRef)       Top-1: {m['Top1_Accuracy']:.4f}  Top-3: {m['Top3_Accuracy']:.4f}  F1: {m['Macro_F1']:.4f}  Sep: {m['Margin_Separation']:.4f}")
    all_results.append({"Method": "COMBO_Mean_BestRef", **m})
    
    # Combo 3: Best temporal + Best reference
    m, p, raw3 = evaluate_config(
        train_cm, train_labels_cm, val_cm, val_labels_cm,
        transform_fn=pca_transform, ref_agg=best_ref_val, top_k_temporal=best_temp_val,
        baseline_preds=base_p
    )
    print(f"  Combo3 (BestTemp+BestRef)   Top-1: {m['Top1_Accuracy']:.4f}  Top-3: {m['Top3_Accuracy']:.4f}  F1: {m['Macro_F1']:.4f}  Sep: {m['Margin_Separation']:.4f}")
    all_results.append({"Method": "COMBO_BestTemp_BestRef", **m})
    
    # Combo 4: Best temporal + Best reference + 50/50 Z-score fusion
    # Need to recompute Swaragram with same temporal config
    sw_combo_m, sw_combo_p, sw_combo_raw = evaluate_config(
        train_sw, train_labels_sw, val_sw, val_labels_sw,
        ref_agg="max", top_k_temporal=best_temp_val, baseline_preds=base_p
    )
    fm, fp = fusion_zscore(raw3, sw_combo_raw, 0.50, 0.50, val_labels_cm, baseline_preds=base_p)
    print(f"  Combo4 (BT+BR+ZScore50/50)  Top-1: {fm['Top1_Accuracy']:.4f}  Top-3: {fm['Top3_Accuracy']:.4f}  F1: {fm['Macro_F1']:.4f}  Sep: {fm['Margin_Separation']:.4f}  W2C: {fm['Wrong_to_Correct']}  C2W: {fm['Correct_to_Wrong']}")
    all_results.append({"Method": "COMBO_BT_BR_ZScore50", **fm})
    
    # Combo 5: Best temporal + Best reference + RRF
    fm, fp = fusion_rrf(raw3, sw_combo_raw, 60, val_labels_cm, baseline_preds=base_p)
    print(f"  Combo5 (BT+BR+RRF_k60)     Top-1: {fm['Top1_Accuracy']:.4f}  Top-3: {fm['Top3_Accuracy']:.4f}  F1: {fm['Macro_F1']:.4f}  Sep: {fm['Margin_Separation']:.4f}  W2C: {fm['Wrong_to_Correct']}  C2W: {fm['Correct_to_Wrong']}")
    all_results.append({"Method": "COMBO_BT_BR_RRF60", **fm})
    
    # =========================================================
    # MODEL SELECTION
    # =========================================================
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OPT_DIR, "optimization_comparison.csv"), index=False)
    
    # Sort by hierarchy: Top1 > Macro_F1 > Margin_Separation
    df_sorted = df_results.sort_values(
        by=["Top1_Accuracy", "Macro_F1", "Margin_Separation"],
        ascending=[False, False, False]
    )
    
    best = df_sorted.iloc[0]
    baseline = df_results[df_results["Method"] == "CM_Baseline_MAX_MEAN"].iloc[0]
    
    print(f"\nBest Method: {best['Method']}")
    print(f"  Val Top-1: {best['Top1_Accuracy']:.4f} (Baseline: {baseline['Top1_Accuracy']:.4f})")
    print(f"  Val Top-3: {best['Top3_Accuracy']:.4f} (Baseline: {baseline['Top3_Accuracy']:.4f})")
    print(f"  Val F1:    {best['Macro_F1']:.4f} (Baseline: {baseline['Macro_F1']:.4f})")
    
    # Apply decision rules
    improvement = best['Top1_Accuracy'] - baseline['Top1_Accuracy']
    f1_improvement = best['Macro_F1'] - baseline['Macro_F1']
    
    if improvement < 0.01 and f1_improvement < 0.01:
        selected = "CM_Baseline_MAX_MEAN"
        reason = "No method produced meaningful improvement over baseline (improvement < 1pp in both accuracy and F1)"
    else:
        selected = best['Method']
        reason = f"Improvement of {improvement:.4f} in Top-1 and {f1_improvement:.4f} in Macro F1"
    
    print(f"\nSELECTED: {selected}")
    print(f"REASON: {reason}")
    
    # =========================================================
    # SEALED TRF TEST EVALUATION
    # =========================================================
    print("\n" + "="*60)
    print("SEALED TRF TEST EVALUATION")
    print("="*60)
    
    # Parse the selected method to determine its config
    # We need to reconstruct the exact configuration
    
    # First run baseline on test
    test_base_m, test_base_p, test_base_raw = evaluate_config(
        train_cm, train_labels_cm, test_cm, test_labels_cm,
        transform_fn=pca_transform, ref_agg="max", top_k_temporal=None
    )
    print(f"  Test Baseline:  Top-1: {test_base_m['Top1_Accuracy']:.4f}  Top-3: {test_base_m['Top3_Accuracy']:.4f}  F1: {test_base_m['Macro_F1']:.4f}  Sep: {test_base_m['Margin_Separation']:.4f}")
    
    # Run selected method on test
    if selected == "CM_Baseline_MAX_MEAN":
        test_sel_m = test_base_m
        test_sel_p = test_base_p
    elif "ZScore" in selected or "RRF" in selected:
        # Need fusion on test
        # Determine the CM config
        if "COMBO_BT_BR" in selected:
            cm_test_m, cm_test_p, cm_test_raw = evaluate_config(
                train_cm, train_labels_cm, test_cm, test_labels_cm,
                transform_fn=pca_transform, ref_agg=best_ref_val, top_k_temporal=best_temp_val
            )
            sw_test_m, sw_test_p, sw_test_raw = evaluate_config(
                train_sw, train_labels_sw, test_sw, test_labels_sw,
                ref_agg="max", top_k_temporal=best_temp_val
            )
        elif "EXP3_ZScore" in selected:
            cm_test_raw = test_base_raw
            sw_test_m, sw_test_p, sw_test_raw = evaluate_config(
                train_sw, train_labels_sw, test_sw, test_labels_sw,
                ref_agg="max"
            )
        elif "EXP3_RRF" in selected:
            cm_test_raw = test_base_raw
            sw_test_m, sw_test_p, sw_test_raw = evaluate_config(
                train_sw, train_labels_sw, test_sw, test_labels_sw,
                ref_agg="max"
            )
        
        if "ZScore" in selected:
            # Extract weights
            if "CM50" in selected or "50" in selected:
                wa, wb = 0.50, 0.50
            elif "CM75" in selected:
                wa, wb = 0.75, 0.25
            elif "CM60" in selected:
                wa, wb = 0.60, 0.40
            elif "CM40" in selected:
                wa, wb = 0.40, 0.60
            elif "CM25" in selected:
                wa, wb = 0.25, 0.75
            else:
                wa, wb = 0.50, 0.50
            test_sel_m, test_sel_p = fusion_zscore(cm_test_raw, sw_test_raw, wa, wb, test_labels_cm)
        elif "RRF" in selected:
            k_val = 60
            for kv in [10, 30, 60]:
                if f"k{kv}" in selected:
                    k_val = kv
            test_sel_m, test_sel_p = fusion_rrf(cm_test_raw, sw_test_raw, k_val, test_labels_cm)
    else:
        # Pure CM config with different temporal/ref settings
        if "COMBO_BestTemp_BestRef" in selected:
            test_sel_m, test_sel_p, _ = evaluate_config(
                train_cm, train_labels_cm, test_cm, test_labels_cm,
                transform_fn=pca_transform, ref_agg=best_ref_val, top_k_temporal=best_temp_val
            )
        elif "COMBO_BestTemp_MAX" in selected:
            test_sel_m, test_sel_p, _ = evaluate_config(
                train_cm, train_labels_cm, test_cm, test_labels_cm,
                transform_fn=pca_transform, ref_agg="max", top_k_temporal=best_temp_val
            )
        elif "COMBO_Mean_BestRef" in selected:
            test_sel_m, test_sel_p, _ = evaluate_config(
                train_cm, train_labels_cm, test_cm, test_labels_cm,
                transform_fn=pca_transform, ref_agg=best_ref_val, top_k_temporal=None
            )
        elif "EXP1_" in selected:
            # Find the temporal config
            for tname, tval in temporal_configs:
                if tname in selected:
                    test_sel_m, test_sel_p, _ = evaluate_config(
                        train_cm, train_labels_cm, test_cm, test_labels_cm,
                        transform_fn=pca_transform, ref_agg="max", top_k_temporal=tval
                    )
                    break
        elif "EXP2_" in selected:
            for rname, rval in ref_configs:
                if rname in selected:
                    test_sel_m, test_sel_p, _ = evaluate_config(
                        train_cm, train_labels_cm, test_cm, test_labels_cm,
                        transform_fn=pca_transform, ref_agg=rval, top_k_temporal=None
                    )
                    break
        else:
            test_sel_m = test_base_m
            test_sel_p = test_base_p
    
    print(f"  Test Selected:  Top-1: {test_sel_m['Top1_Accuracy']:.4f}  Top-3: {test_sel_m['Top3_Accuracy']:.4f}  F1: {test_sel_m['Macro_F1']:.4f}  Sep: {test_sel_m['Margin_Separation']:.4f}")
    
    # Also run 50/50 Z-score on test for comparison
    sw_test_base_m, sw_test_base_p, sw_test_base_raw = evaluate_config(
        train_sw, train_labels_sw, test_sw, test_labels_sw,
        ref_agg="max"
    )
    z50_test_m, z50_test_p = fusion_zscore(test_base_raw, sw_test_base_raw, 0.50, 0.50, test_labels_cm)
    print(f"  Test Z50/50:    Top-1: {z50_test_m['Top1_Accuracy']:.4f}  Top-3: {z50_test_m['Top3_Accuracy']:.4f}  F1: {z50_test_m['Macro_F1']:.4f}  Sep: {z50_test_m['Margin_Separation']:.4f}")
    
    # =========================================================
    # SAVE ALL OUTPUTS
    # =========================================================
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    pd.DataFrame(all_results).to_csv(os.path.join(OPT_DIR, "optimization_comparison.csv"), index=False)
    
    with open(os.path.join(OPT_DIR, "optimization_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    pd.DataFrame(test_sel_p).to_csv(os.path.join(OPT_DIR, "per_recording_validation_predictions.csv"), index=False)
    
    # Margin analysis
    margin_data = []
    for r in all_results:
        margin_data.append({
            "Method": r["Method"],
            "Mean_Margin": r.get("Mean_Margin_1_2", 0),
            "Median_Margin": r.get("Median_Margin_1_2", 0),
            "Correct_Margin": r.get("Mean_Correct_Margin", 0),
            "Incorrect_Margin": r.get("Mean_Incorrect_Margin", 0),
            "Separation": r.get("Margin_Separation", 0)
        })
    pd.DataFrame(margin_data).to_csv(os.path.join(OPT_DIR, "margin_analysis.csv"), index=False)
    
    # Prediction change analysis
    change_data = []
    for r in all_results:
        change_data.append({
            "Method": r["Method"],
            "Wrong_to_Correct": r.get("Wrong_to_Correct", 0),
            "Correct_to_Wrong": r.get("Correct_to_Wrong", 0),
            "Net_Change": r.get("Wrong_to_Correct", 0) - r.get("Correct_to_Wrong", 0)
        })
    pd.DataFrame(change_data).to_csv(os.path.join(OPT_DIR, "prediction_change_analysis.csv"), index=False)
    
    # Final test results
    test_results = [
        {"Method": "CM_Baseline", **test_base_m},
        {"Method": selected, **test_sel_m},
        {"Method": "ZScore_50_50", **z50_test_m}
    ]
    pd.DataFrame(test_results).to_csv(os.path.join(OPT_DIR, "final_test_results.csv"), index=False)
    
    # Method selection JSON
    method_sel = {
        "selected_method": selected,
        "selected_temporal_aggregation": best_temp_name if "BestTemp" in selected or "EXP1" in selected else "Mean_All",
        "selected_reference_aggregation": best_ref_name if "BestRef" in selected or "EXP2" in selected else "MAX",
        "selected_fusion_method": "ZScore_50_50" if "ZScore50" in selected else ("RRF" if "RRF" in selected else "None"),
        "validation_top1": float(best['Top1_Accuracy']) if selected != "CM_Baseline_MAX_MEAN" else float(baseline['Top1_Accuracy']),
        "validation_top3": float(best['Top3_Accuracy']) if selected != "CM_Baseline_MAX_MEAN" else float(baseline['Top3_Accuracy']),
        "validation_macro_f1": float(best['Macro_F1']) if selected != "CM_Baseline_MAX_MEAN" else float(baseline['Macro_F1']),
        "validation_margin": float(best.get('Mean_Margin_1_2', 0)),
        "test_top1": float(test_sel_m['Top1_Accuracy']),
        "test_top3": float(test_sel_m['Top3_Accuracy']),
        "test_macro_f1": float(test_sel_m['Macro_F1']),
        "test_margin_separation": float(test_sel_m['Margin_Separation']),
        "reason_for_selection": reason,
        "baseline_val_top1": float(baseline['Top1_Accuracy']),
        "baseline_test_top1": float(test_base_m['Top1_Accuracy']),
        "z50_test_top1": float(z50_test_m['Top1_Accuracy']),
        "rejected_alternatives": [r["Method"] for r in all_results if r["Method"] != selected]
    }
    
    with open(os.path.join(OPT_DIR, "final_method_selection.json"), "w") as f:
        json.dump(method_sel, f, indent=4, default=str)
    
    # Integrity report
    with open(os.path.join(OPT_DIR, "integrity_report.txt"), "w") as f:
        f.write("OPTIMIZATION EXPERIMENT INTEGRITY REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        f.write(f"Total runtime: {time.time() - t_start:.1f}s\n")
        f.write(f"CultureMERT TRAIN recordings: {len(train_cm)}\n")
        f.write(f"CultureMERT VAL recordings: {len(val_cm)}\n")
        f.write(f"CultureMERT TEST recordings: {len(test_cm)}\n")
        f.write(f"Swaragram TRAIN recordings: {len(train_sw)}\n")
        f.write(f"Swaragram VAL recordings: {len(val_sw)}\n")
        f.write(f"Swaragram TEST recordings: {len(test_sw)}\n")
        f.write(f"PCA fitted on TRAIN only: Yes\n")
        f.write(f"Test used for model selection: No\n")
        f.write(f"Folk data touched: No\n")
        f.write(f"Total configurations tested: {len(all_results)}\n")
        f.write(f"Selected method: {selected}\n")
        f.write(f"Selection reason: {reason}\n")
    
    # Summary
    if selected == "CM_Baseline_MAX_MEAN":
        verdict = "CURRENT METHOD REMAINS BEST"
    else:
        verdict = "NEW METHOD IS BETTER"
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Selected: {selected}")
    print(f"Val Top-1: {method_sel['validation_top1']:.4f}")
    print(f"Test Top-1: {method_sel['test_top1']:.4f}")
    print(f"{'='*60}")
    
    print("\nDone. All outputs saved to:", OPT_DIR)

if __name__ == '__main__':
    main()
