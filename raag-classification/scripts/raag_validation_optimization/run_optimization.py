
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

# Directories
OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization")
os.makedirs(OPT_DIR, exist_ok=True)

# Asset Paths
TRF_EMB_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\embeddings")
TRF_METADATA_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\embedding_metadata.csv")
TRF_MANIFEST_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")

SARAGA_EMB_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_embeddings")
SARAGA_MANIFEST_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
SARAGA_MAPPING_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_trf_raag_mapping.csv")

def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def run_asset_inventory():
    print("Running Asset Inventory...")
    inventory = {
        "TRF_Embeddings": os.path.exists(TRF_EMB_DIR),
        "TRF_Metadata": os.path.exists(TRF_METADATA_CSV),
        "TRF_Manifest": os.path.exists(TRF_MANIFEST_CSV),
        "SARAGA_Embeddings": os.path.exists(SARAGA_EMB_DIR),
        "SARAGA_Manifest": os.path.exists(SARAGA_MANIFEST_CSV),
        "SARAGA_Mapping": os.path.exists(SARAGA_MAPPING_CSV),
    }
    with open(os.path.join(OPT_DIR, "asset_inventory.json"), "w") as f:
        json.dump(inventory, f, indent=4)
    if not all(inventory.values()):
        raise ValueError("Missing critical assets. Inventory failed.")
    return True

def build_data_split(df_trf):
    # Group by source_recording
    recs = df_trf[['source_recording', 'raag']].drop_duplicates()
    
    # We want 60/20/20. We will try to stratify.
    # Count recordings per raag
    counts = recs['raag'].value_counts()
    
    train_recs, val_recs, test_recs = [], [], []
    rare_classes = []
    
    for raag, count in counts.items():
        subset = recs[recs['raag'] == raag]['source_recording'].tolist()
        if count < 3:
            # Cannot stratify perfectly to 3 sets
            rare_classes.append(raag)
            if count == 1:
                train_recs.extend(subset)
            elif count == 2:
                train_recs.append(subset[0])
                val_recs.append(subset[1])
        else:
            # 60/20/20 split
            t, temp = train_test_split(subset, test_size=0.4, random_state=42)
            v, te = train_test_split(temp, test_size=0.5, random_state=42)
            train_recs.extend(t)
            val_recs.extend(v)
            test_recs.extend(te)
            
    print(f"TRF Split: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    print(f"Classes with insufficient representation: {len(rare_classes)}")
    
    split_df = pd.DataFrame([
        {"source_recording": r, "split": "TRAIN"} for r in train_recs
    ] + [
        {"source_recording": r, "split": "VAL"} for r in val_recs
    ] + [
        {"source_recording": r, "split": "TEST"} for r in test_recs
    ])
    
    # Merge with full metadata to assign splits to all sequences
    df_trf = df_trf.merge(split_df, on="source_recording", how="left")
    df_trf.to_csv(os.path.join(OPT_DIR, "split_manifest.csv"), index=False)
    
    return df_trf, rare_classes

def load_embeddings(df_trf, window_size=12):
    """Loads and pools embeddings. window_size=12 is 60s, window_size=6 is 30s"""
    embeddings = {} # recording -> array of sequences
    labels = {}     # recording -> raag
    splits = {}     # recording -> split
    
    for rec, group in df_trf.groupby("source_recording"):
        paths = group['embedding_path'].tolist()
        if len(paths) == 0: continue
        try:
            arr = np.load(paths[0])
            if len(arr) == 0: continue
            
            # Pool
            num_seqs = len(arr) // window_size
            if num_seqs == 0: continue
            
            seqs = []
            for i in range(num_seqs):
                chunk = arr[i*window_size:(i+1)*window_size]
                if len(chunk) == window_size:
                    seqs.append(np.mean(chunk, axis=0))
            
            if len(seqs) > 0:
                embeddings[rec] = np.array(seqs)
                labels[rec] = group['raag'].iloc[0]
                splits[rec] = group['split'].iloc[0]
        except:
            pass
    return embeddings, labels, splits

def evaluate_retrieval(train_embs, train_labels, query_embs, query_labels, transform_fn=None, raag_agg="mean", temporal_agg="mean"):
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
    top3_acc_count = 0
    correct_count = 0
    
    margins_1_2 = []
    margins_1_3 = []
    
    for q_rec, q_seqs in query_embs.items():
        if transform_fn:
            q_seqs = transform_fn(q_seqs)
        q_seqs = l2_norm(q_seqs)
        
        true_raag = query_labels[q_rec]
        
        # Sim matrix for this query (Q_seqs x Train_seqs)
        sim_matrix = np.dot(q_seqs, X_train.T)
        
        # Temporal aggregation FIRST or Raag aggregation FIRST?
        # Standard: 
        # 1. Compute similarity against ALL TRF reference sequences.
        # 2. Group those similarities by Raag.
        # 3. Derive a Raag-level score.
        
        # We need a score for each Raag.
        raag_scores = np.zeros(len(unique_raags))
        
        for i, r in enumerate(unique_raags):
            mask = y_train == r
            if not np.any(mask): continue
            
            r_sims = sim_matrix[:, mask] # Shape (n_query_seqs, n_ref_seqs_for_raag)
            
            # Raag Reference Aggregation
            if raag_agg == "mean":
                q_raag_sims = np.mean(r_sims, axis=1) # (n_query_seqs,)
            elif raag_agg == "max":
                q_raag_sims = np.max(r_sims, axis=1)
            elif raag_agg == "median":
                q_raag_sims = np.median(r_sims, axis=1)
            elif raag_agg.startswith("top"):
                k = int(raag_agg.split("-")[1])
                # sort references descending
                sorted_r_sims = np.sort(r_sims, axis=1)[:, ::-1]
                q_raag_sims = np.mean(sorted_r_sims[:, :min(k, sorted_r_sims.shape[1])], axis=1)
                
            # Temporal Aggregation
            if temporal_agg == "mean":
                final_score = np.mean(q_raag_sims)
            elif temporal_agg == "median":
                final_score = np.median(q_raag_sims)
            elif temporal_agg.startswith("top"):
                k = int(temporal_agg.split("-")[1])
                sorted_q = np.sort(q_raag_sims)[::-1]
                final_score = np.mean(sorted_q[:min(k, len(sorted_q))])
                
            raag_scores[i] = final_score
            
        # Rank Raags
        top_idx = np.argsort(raag_scores)[::-1]
        top_raags = unique_raags[top_idx]
        top_scores = raag_scores[top_idx]
        
        pred_raag = top_raags[0]
        
        correct = 1 if pred_raag == true_raag else 0
        correct_count += correct
        if true_raag in top_raags[:3]: top3_acc_count += 1
        
        margins_1_2.append(top_scores[0] - top_scores[1] if len(top_scores) > 1 else 0)
        margins_1_3.append(top_scores[0] - top_scores[2] if len(top_scores) > 2 else 0)
        
        preds.append({
            "recording_id": q_rec,
            "true_raag": true_raag,
            "predicted_raag": pred_raag,
            "top2_raag": top_raags[1] if len(top_raags) > 1 else None,
            "top3_raag": top_raags[2] if len(top_raags) > 2 else None,
            "score_top1": top_scores[0],
            "score_top2": top_scores[1] if len(top_scores) > 1 else 0,
            "score_top3": top_scores[2] if len(top_scores) > 2 else 0,
            "margin_1_2": margins_1_2[-1],
            "margin_1_3": margins_1_3[-1],
            "correct": correct
        })
        
    y_true = [p["true_raag"] for p in preds]
    y_pred = [p["predicted_raag"] for p in preds]
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Top3_Accuracy": top3_acc_count / max(1, len(preds)),
        "Macro_F1": f1_score(y_true, y_pred, average='macro'),
        "Mean_Margin_1_2": np.mean(margins_1_2),
        "Mean_Margin_1_3": np.mean(margins_1_3)
    }
    
    return metrics, preds

def main():
    run_asset_inventory()
    
    df_trf = pd.read_csv(TRF_METADATA_CSV)
    df_trf, rare_classes = build_data_split(df_trf)
    
    # Load 60s
    print("Loading 60s Embeddings...")
    embs_60, labels_60, splits_60 = load_embeddings(df_trf, window_size=12)
    
    train_embs_60 = {r: e for r, e in embs_60.items() if splits_60[r] == "TRAIN"}
    train_labels_60 = {r: labels_60[r] for r in train_embs_60.keys()}
    val_embs_60 = {r: e for r, e in embs_60.items() if splits_60[r] == "VAL"}
    val_labels_60 = {r: labels_60[r] for r in val_embs_60.keys()}
    test_embs_60 = {r: e for r, e in embs_60.items() if splits_60[r] == "TEST"}
    test_labels_60 = {r: labels_60[r] for r in test_embs_60.keys()}
    
    # Grid Search on Validation
    experiments = []
    
    # 1. Baseline
    print("Evaluating Baseline L2 (60s)...")
    base_metrics, _ = evaluate_retrieval(train_embs_60, train_labels_60, val_embs_60, val_labels_60)
    experiments.append({"Transform": "None", "RaagAgg": "mean", "TempAgg": "mean", **base_metrics})
    
    # Fit Transformations on Train
    flat_train = np.vstack(list(train_embs_60.values()))
    train_mean = np.mean(flat_train, axis=0)
    
    def center_transform(x): return x - train_mean
    
    pca256 = PCA(n_components=256).fit(flat_train)
    def pca256_transform(x): return pca256.transform(x)
    
    pca128_w = PCA(n_components=128, whiten=True).fit(flat_train)
    def pca128w_transform(x): return pca128_w.transform(x)
    
    print("Evaluating Representation Transformations...")
    for name, fn in [("Center", center_transform), ("PCA256", pca256_transform), ("PCA128_Whiten", pca128w_transform)]:
        m, _ = evaluate_retrieval(train_embs_60, train_labels_60, val_embs_60, val_labels_60, transform_fn=fn)
        experiments.append({"Transform": name, "RaagAgg": "mean", "TempAgg": "mean", **m})
        
    # Pick best transform based on Macro F1 + Margin
    best_transform_name = max(experiments, key=lambda x: x["Accuracy"] + x["Mean_Margin_1_2"])["Transform"]
    best_transform_fn = {"None": None, "Center": center_transform, "PCA256": pca256_transform, "PCA128_Whiten": pca128w_transform}[best_transform_name]
    
    print(f"Selected Transform: {best_transform_name}")
    
    print("Evaluating Aggregations...")
    for raag_agg in ["mean", "max", "top-3", "top-10"]:
        for temp_agg in ["mean", "top-3"]:
            m, _ = evaluate_retrieval(train_embs_60, train_labels_60, val_embs_60, val_labels_60, transform_fn=best_transform_fn, raag_agg=raag_agg, temporal_agg=temp_agg)
            experiments.append({"Transform": best_transform_name, "RaagAgg": raag_agg, "TempAgg": temp_agg, **m})
            
    df_exp = pd.DataFrame(experiments)
    df_exp.to_csv(os.path.join(OPT_DIR, "validation_comparison.csv"), index=False)
    
    best_exp = df_exp.loc[df_exp['Accuracy'].idxmax()].to_dict()
    print(f"Best Validation Config: {best_exp}")
    
    with open(os.path.join(OPT_DIR, "final_method_selection.json"), "w") as f:
        json.dump(best_exp, f, indent=4)
        
    print("Running on TRF TEST SET...")
    best_raag_agg = best_exp['RaagAgg']
    best_temp_agg = best_exp['TempAgg']
    
    test_metrics, test_preds = evaluate_retrieval(train_embs_60, train_labels_60, test_embs_60, test_labels_60, transform_fn=best_transform_fn, raag_agg=best_raag_agg, temporal_agg=best_temp_agg)
    
    df_test = pd.DataFrame(test_preds)
    df_test.to_csv(os.path.join(OPT_DIR, "per_recording_predictions.csv"), index=False)
    
    print("TRF Test Accuracy:", test_metrics["Accuracy"])
    
    # SARAGA Evaluation
    print("Running SARAGA Evaluation...")
    saraga_manifest = pd.read_csv(SARAGA_MANIFEST_CSV)
    
    saraga_embs = {}
    saraga_labels = {}
    
    # Extract Saraga Embs
    for rec in saraga_manifest['saraga_recording_id'].unique():
        s_paths = glob.glob(os.path.join(SARAGA_EMB_DIR, f"{rec}.npy"))
        if len(s_paths) > 0:
            try:
                arr = np.load(s_paths[0])
                num_seqs = len(arr) // 12
                if num_seqs > 0:
                    seqs = [np.mean(arr[i*12:(i+1)*12], axis=0) for i in range(num_seqs)]
                    saraga_embs[rec] = np.array(seqs)
                    # Label
                    mapped = saraga_manifest[saraga_manifest['saraga_recording_id'] == rec]['mapped_trf_raag'].iloc[0]
                    if pd.notnull(mapped) and mapped != 'NONE':
                        saraga_labels[rec] = mapped
            except:
                pass
                
    saraga_eval_embs = {k: v for k, v in saraga_embs.items() if k in saraga_labels}
    
    if len(saraga_eval_embs) > 0:
        saraga_metrics, saraga_preds = evaluate_retrieval(train_embs_60, train_labels_60, saraga_eval_embs, saraga_labels, transform_fn=best_transform_fn, raag_agg=best_raag_agg, temporal_agg=best_temp_agg)
        
        pd.DataFrame(saraga_preds).to_csv(os.path.join(OPT_DIR, "saraga_predictions.csv"), index=False)
        pd.DataFrame([saraga_metrics]).to_csv(os.path.join(OPT_DIR, "saraga_metrics.csv"), index=False)
        print("SARAGA Accuracy:", saraga_metrics["Accuracy"])
    else:
        print("No valid SARAGA recordings found.")
        
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
