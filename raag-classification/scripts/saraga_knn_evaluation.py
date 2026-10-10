
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

# ----------------------------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------------------------
TRF_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier")
TRF_EMB_DIR = os.path.join(TRF_DIR, "embeddings")
SARAGA_EVAL_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval")
SARAGA_EMB_DIR = os.path.join(SARAGA_EVAL_DIR, "saraga_embeddings")
OUT_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_knn_saraga_eval")

os.makedirs(OUT_DIR, exist_ok=True)

def build_dataset(context_windows, manifest_df, emb_dir, id_col, raag_col):
    """
    Builds sequence-level representations by mean-pooling the 5s embeddings.
    """
    seqs, rids, raags = [], [], []
    for _, row in manifest_df.iterrows():
        rec_id = row[id_col]
        npy_path = os.path.join(emb_dir, f"{rec_id}.npy")
        if not os.path.exists(npy_path): continue
        emb = np.load(npy_path)
        if len(emb) == 0: continue
        
        num_seqs = len(emb) // context_windows
        if num_seqs == 0: continue
        
        for i in range(num_seqs):
            seq_emb = emb[i*context_windows:(i+1)*context_windows]
            # Mean pool across time
            mean_emb = seq_emb.mean(axis=0)
            seqs.append(mean_emb)
            rids.append(rec_id)
            raags.append(row[raag_col])
            
    return np.array(seqs), np.array(rids), np.array(raags)

def main():
    print("PHASE 1: Loading Manifests & Embeddings...")
    trf_manifest = pd.read_csv(os.path.join(TRF_DIR, "data", "trf_training_manifest.csv"))
    saraga_manifest = pd.read_csv(os.path.join(SARAGA_EVAL_DIR, "saraga_evaluation_manifest.csv"))
    
    # Filter Saraga eligible
    saraga_manifest = saraga_manifest[saraga_manifest['evaluation_eligibility'] == True]
    
    print("PHASE 2: Building 30s & 60s Representations...")
    trf_30_seqs, trf_30_rids, trf_30_raags = build_dataset(6, trf_manifest, TRF_EMB_DIR, 'source_recording', 'raag')
    trf_60_seqs, trf_60_rids, trf_60_raags = build_dataset(12, trf_manifest, TRF_EMB_DIR, 'source_recording', 'raag')
    
    saraga_30_seqs, saraga_30_rids, saraga_30_raags = build_dataset(6, saraga_manifest, SARAGA_EMB_DIR, 'saraga_recording_id', 'mapped_trf_raag')
    saraga_60_seqs, saraga_60_rids, saraga_60_raags = build_dataset(12, saraga_manifest, SARAGA_EMB_DIR, 'saraga_recording_id', 'mapped_trf_raag')
    
    unique_trf_raags = np.unique(trf_30_raags)
    raag_to_idx = {r: i for i, r in enumerate(unique_trf_raags)}
    
    def evaluate_knn(trf_seqs, trf_raags, saraga_seqs, saraga_rids, saraga_raags, context):
        print(f"Evaluating {context} KNN...")
        # Compute full cosine similarity matrix: [Num_Saraga_Seqs, Num_TRF_Seqs]
        sim_matrix = cosine_similarity(saraga_seqs, trf_seqs)
        
        # We need sequence-level predictions and recording-level aggregation.
        # For a given sequence, the score of Raag C is the max similarity to any TRF sequence of Raag C.
        
        # Precompute boolean masks for each TRF Raag for fast masked max operations
        masks = {r: (trf_raags == r) for r in unique_trf_raags}
        
        # seq_scores: [Num_Saraga_Seqs, Num_Unique_Raags]
        seq_scores = np.zeros((len(saraga_seqs), len(unique_trf_raags)))
        
        for i, r in enumerate(unique_trf_raags):
            mask = masks[r]
            if not np.any(mask):
                seq_scores[:, i] = -1.0 # Impossible
                continue
            # Max similarity to class r for all queries
            seq_scores[:, i] = np.max(sim_matrix[:, mask], axis=1)
            
        # Group sequences by recording
        rec_to_scores = defaultdict(list)
        rec_to_truth = {}
        for idx, rid in enumerate(saraga_rids):
            rec_to_scores[rid].append(seq_scores[idx])
            rec_to_truth[rid] = saraga_raags[idx]
            
        r_preds, r_truths, r_scores = [], [], []
        rec_details = []
        
        for rid, scores_list in rec_to_scores.items():
            # Average similarity score per candidate Raag across sequences
            agg_scores = np.mean(np.array(scores_list), axis=0)
            
            top_indices = np.argsort(agg_scores)[::-1]
            top_classes = [unique_trf_raags[i] for i in top_indices]
            
            true_raag = rec_to_truth[rid]
            pred_raag = top_classes[0]
            
            r_preds.append(pred_raag)
            r_truths.append(true_raag)
            r_scores.append(agg_scores)
            
            rec_details.append({
                'context': context,
                'recording_id': rid,
                'ground_truth': true_raag,
                'predicted_raag': pred_raag,
                'top1_similarity': agg_scores[top_indices[0]],
                'top2_similarity': agg_scores[top_indices[1]],
                'top3_similarity': agg_scores[top_indices[2]],
                'margin_top1_top2': agg_scores[top_indices[0]] - agg_scores[top_indices[1]],
                'top1_correct': pred_raag == true_raag,
                'top3_correct': true_raag in top_classes[:3],
                'top5_correct': true_raag in top_classes[:5],
                'top_5_classes': top_classes[:5]
            })
            
        # Calculate metrics
        acc = accuracy_score(r_truths, r_preds)
        p, r, f1_macro, _ = precision_recall_fscore_support(r_truths, r_preds, average='macro', zero_division=0)
        p_w, r_w, f1_weighted, _ = precision_recall_fscore_support(r_truths, r_preds, average='weighted', zero_division=0)
        
        top3_acc = np.mean([d['top3_correct'] for d in rec_details])
        top5_acc = np.mean([d['top5_correct'] for d in rec_details])
        
        metrics = {
            'context': context,
            'accuracy': acc,
            'macro_f1': f1_macro,
            'weighted_f1': f1_weighted,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc
        }
        
        return metrics, rec_details

    res_30, det_30 = evaluate_knn(trf_30_seqs, trf_30_raags, saraga_30_seqs, saraga_30_rids, saraga_30_raags, '30s')
    res_60, det_60 = evaluate_knn(trf_60_seqs, trf_60_raags, saraga_60_seqs, saraga_60_rids, saraga_60_raags, '60s')
    
    # Save predictions
    df_preds = pd.DataFrame(det_30 + det_60)
    df_preds.to_csv(os.path.join(OUT_DIR, "saraga_knn_predictions.csv"), index=False)
    
    # Per-Raag Analysis (on 60s)
    df_det60 = pd.DataFrame(det_60)
    per_raag = []
    for raag in df_det60['ground_truth'].unique():
        subset = df_det60[df_det60['ground_truth'] == raag]
        top1_acc = subset['top1_correct'].mean()
        top5_acc = subset['top5_correct'].mean()
        
        incorrects = subset[subset['top1_correct'] == False]['predicted_raag']
        most_common_incorrect = incorrects.value_counts().idxmax() if not incorrects.empty else "None"
        
        per_raag.append({
            'raag': raag,
            'saraga_recordings': len(subset),
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'most_common_error': most_common_incorrect
        })
    df_per_raag = pd.DataFrame(per_raag)
    df_per_raag.to_csv(os.path.join(OUT_DIR, "saraga_knn_per_raag_metrics.csv"), index=False)
    
    # Compare against Supervised
    # User provided values:
    supervised = [
        {'Model': '30s MeanPool (Supervised)', 'Accuracy': 0.0926, 'Macro_F1': 0.060, 'Top5': 0.1667},
        {'Model': '30s BiLSTM (Supervised)', 'Accuracy': 0.0741, 'Macro_F1': 0.051, 'Top5': 0.1852},
        {'Model': '60s MeanPool (Supervised)', 'Accuracy': 0.1111, 'Macro_F1': 0.082, 'Top5': 0.1667},
        {'Model': '60s BiLSTM (Supervised)', 'Accuracy': 0.1111, 'Macro_F1': 0.087, 'Top5': 0.2222},
        {'Model': '30s MeanPool (KNN)', 'Accuracy': res_30['accuracy'], 'Macro_F1': res_30['macro_f1'], 'Top5': res_30['top5_accuracy']},
        {'Model': '60s MeanPool (KNN)', 'Accuracy': res_60['accuracy'], 'Macro_F1': res_60['macro_f1'], 'Top5': res_60['top5_accuracy']}
    ]
    df_comp = pd.DataFrame(supervised)
    
    report_md = f"""
# SARAGA CULTUREMERT KNN EVALUATION REPORT

## 1. Objective
Determine whether the CultureMERT representation space retains sufficient cross-dataset similarity using parameter-free KNN retrieval, identifying if the bottleneck is the trained MLPs or the embedding space itself.

## 2. Methodology
- **Reference Bank:** {len(trf_30_rids)} (30s) and {len(trf_60_rids)} (60s) TRF sequences.
- **Queries:** {len(saraga_30_rids)} (30s) and {len(saraga_60_rids)} (60s) eligible SARAGA sequences.
- **Similarity:** Maximum Cosine Similarity per TRF Raag class.
- **Aggregation:** Average class similarity across all sequences in a SARAGA recording.

## 3. Results Comparison
{df_comp.to_string(index=False)}

## 4. Interpretation
- **Supervised vs Unsupervised:** If KNN outperforms the supervised MLP, it suggests the downstream MLP failed to generalize or overfitted to TRF recording acoustics. If KNN is lower, it suggests the embedding space requires complex non-linear separation.
- **Representation Stability:** Mean-pooling the 768-D representation over 30s/60s provides a stable acoustic fingerprint for cosine distance.
- **Limitations:** Unsupervised retrieval forces the embeddings to act as identity verifiers without domain adaptation. It assumes Raags form tight spherical clusters, which may not hold for highly variable Hindustani performances.

## 5. Next Steps for Folk Analysis
Depending on the KNN margins, utilizing the embedding bank as a direct similarity anchor for the unlabelled Folk dataset might prove more statistically resilient than forcing hard classifier predictions, given the extreme out-of-distribution nature of Folk music.
"""
    with open(os.path.join(OUT_DIR, "saraga_knn_report.md"), "w") as f:
        f.write(report_md.strip())
        
    summary = {
        'trf_reference_sequences_30s': len(trf_30_rids),
        'saraga_queries_30s': len(saraga_30_rids),
        'knn_30s_accuracy': res_30['accuracy'],
        'knn_60s_accuracy': res_60['accuracy'],
    }
    with open(os.path.join(OUT_DIR, "saraga_knn_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    with open(os.path.join(OUT_DIR, "evaluation_integrity_report.txt"), "w") as f:
        f.write("All TRF and SARAGA metadata locked. No training executed. Completely deterministic cosine similarity.")
        
    print("KNN Evaluation Complete.")

if __name__ == '__main__':
    main()
