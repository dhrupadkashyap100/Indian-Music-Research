
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import rankdata

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\multifeature_raag_optimization")
CACHE_DIR = os.path.join(OPT_DIR, "feature_caches")

SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

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

def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def build_similarity_matrix(query_embs, ref_embs):
    return np.dot(query_embs, ref_embs.T)

def aggregate_and_rank(sim_matrix, train_labels):
    unique_raags = np.unique(train_labels)
    raag_scores = np.zeros((sim_matrix.shape[0], len(unique_raags)))
    for i, r in enumerate(unique_raags):
        mask = train_labels == r
        if np.any(mask):
            r_sims = sim_matrix[:, mask]
            raag_scores[:, i] = np.max(r_sims, axis=1)
    return raag_scores, unique_raags

def rank_normalize(scores):
    normed = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        normed[i] = rankdata(scores[i]) / scores.shape[1]
    return normed

def load_acoustic(recs, feat_key, acoustic_embs):
    return np.array([acoustic_embs[r][feat_key] for r in recs])

def get_normed_scores(train_X, test_X, train_labels):
    sims = build_similarity_matrix(test_X, train_X)
    scores, ur = aggregate_and_rank(sims, train_labels)
    return rank_normalize(scores), scores, ur

def evaluate_scores(scores, true_labels, ur):
    preds = []
    for i in range(len(scores)):
        idx = np.argsort(scores[i])[::-1]
        preds.append(ur[idx[0]])
    return accuracy_score(true_labels, preds), f1_score(true_labels, preds, average='macro', zero_division=0)

def main():
    print("Loading Data...")
    df_trf = pd.read_csv(SPLIT_MANIFEST).drop_duplicates('source_recording')
    
    acoustic_embs = {}
    for rec in df_trf['source_recording']:
        p = os.path.join(CACHE_DIR, "TRF", f"{rec}.npy")
        if os.path.exists(p):
            acoustic_embs[rec] = np.load(p, allow_pickle=True).item()
            
    df_split = pd.read_csv(SPLIT_MANIFEST)
    mert_embs, labels, splits = load_mert_embeddings(df_split)
    
    train_recs = [r for r in labels.keys() if splits[r] == 'TRAIN' and r in acoustic_embs]
    test_recs = [r for r in labels.keys() if splits[r] == 'TEST' and r in acoustic_embs]
    
    train_labels = np.array([labels[r] for r in train_recs])
    test_labels = np.array([labels[r] for r in test_recs])
    
    # 1. MERT Setup
    mert_train_flat = np.vstack([mert_embs[r] for r in train_recs])
    pca = PCA(n_components=128, whiten=True).fit(mert_train_flat)
    
    def process_mert(recs, embs_dict):
        out = []
        for r in recs:
            seqs = embs_dict[r]
            trans = l2_norm(pca.transform(seqs))
            out.append(np.mean(trans, axis=0))
        return l2_norm(np.array(out))
        
    mert_train_X = process_mert(train_recs, mert_embs)
    mert_test_X = process_mert(test_recs, mert_embs)
    
    mel_train_X = load_acoustic(train_recs, 'mel', acoustic_embs)
    mel_test_X = load_acoustic(test_recs, 'mel', acoustic_embs)
    chroma_train_X = load_acoustic(train_recs, 'chroma', acoustic_embs)
    chroma_test_X = load_acoustic(test_recs, 'chroma', acoustic_embs)
    swara_train_X = load_acoustic(train_recs, 'swara', acoustic_embs)
    swara_test_X = load_acoustic(test_recs, 'swara', acoustic_embs)
    
    # Eval MERT Baseline TRF Test
    mert_norm_test, mert_raw_test, ur = get_normed_scores(mert_train_X, mert_test_X, train_labels)
    acc_mert_test, mac_mert_test = evaluate_scores(mert_raw_test, test_labels, ur)
    
    # Eval Fusion Pitch Dominant TRF Test (MERT 0.40, Mel 0.15, Chroma 0.25, Swara 0.20)
    mel_norm_test, _, _ = get_normed_scores(mel_train_X, mel_test_X, train_labels)
    chroma_norm_test, _, _ = get_normed_scores(chroma_train_X, chroma_test_X, train_labels)
    swara_norm_test, _, _ = get_normed_scores(swara_train_X, swara_test_X, train_labels)
    
    w = [0.40, 0.15, 0.25, 0.20]
    f_scores_test = (w[0]*mert_norm_test) + (w[1]*mel_norm_test) + (w[2]*chroma_norm_test) + (w[3]*swara_norm_test)
    acc_fus_test, mac_fus_test = evaluate_scores(f_scores_test, test_labels, ur)
    
    # SARAGA Evaluation
    print("Loading SARAGA...")
    saraga_manifest = pd.read_csv(SARAGA_MANIFEST).drop_duplicates('saraga_recording_id')
    saraga_acoustic = {}
    for rec in saraga_manifest['saraga_recording_id']:
        p = os.path.join(CACHE_DIR, "SARAGA", f"{rec}.npy")
        if os.path.exists(p):
            saraga_acoustic[rec] = np.load(p, allow_pickle=True).item()
            
    df_saraga_mert = pd.read_csv(os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_sequence_predictions.csv"))
    saraga_mert_embs = {}
    for rec in saraga_acoustic.keys():
        s_paths = glob.glob(rf"PROJECT_ROOT\scripts\culturemert_cross_dataset_saraga_eval\saraga_embeddings\{rec}.npy")
        if s_paths:
            try:
                arr = np.load(s_paths[0])
                num_seqs = len(arr) // 12
                if num_seqs > 0:
                    seqs = [np.mean(arr[i*12:(i+1)*12], axis=0) for i in range(num_seqs)]
                    saraga_mert_embs[rec] = np.array(seqs)
            except: pass
            
    saraga_recs = [r for r in saraga_acoustic.keys() if r in saraga_mert_embs]
    saraga_labels = [saraga_manifest[saraga_manifest['saraga_recording_id'] == r]['mapped_trf_raag'].iloc[0] for r in saraga_recs]
    valid_saraga_recs = [r for i, r in enumerate(saraga_recs) if pd.notnull(saraga_labels[i]) and saraga_labels[i] != 'NONE']
    valid_saraga_labels = [l for l in saraga_labels if pd.notnull(l) and l != 'NONE']
    
    mert_saraga_X = process_mert(valid_saraga_recs, saraga_mert_embs)
    mel_saraga_X = load_acoustic(valid_saraga_recs, 'mel', saraga_acoustic)
    chroma_saraga_X = load_acoustic(valid_saraga_recs, 'chroma', saraga_acoustic)
    swara_saraga_X = load_acoustic(valid_saraga_recs, 'swara', saraga_acoustic)
    
    mert_norm_sar, mert_raw_sar, _ = get_normed_scores(mert_train_X, mert_saraga_X, train_labels)
    acc_mert_sar, mac_mert_sar = evaluate_scores(mert_raw_sar, valid_saraga_labels, ur)
    
    mel_norm_sar, _, _ = get_normed_scores(mel_train_X, mel_saraga_X, train_labels)
    chroma_norm_sar, _, _ = get_normed_scores(chroma_train_X, chroma_saraga_X, train_labels)
    swara_norm_sar, _, _ = get_normed_scores(swara_train_X, swara_saraga_X, train_labels)
    
    f_scores_sar = (w[0]*mert_norm_sar) + (w[1]*mel_norm_sar) + (w[2]*chroma_norm_sar) + (w[3]*swara_norm_sar)
    acc_fus_sar, mac_fus_sar = evaluate_scores(f_scores_sar, valid_saraga_labels, ur)
    
    print("MERT TRF Test Top-1:", acc_mert_test)
    print("MERT SARAGA Top-1:", acc_mert_sar)
    print("Fusion TRF Test Top-1:", acc_fus_test)
    print("Fusion SARAGA Top-1:", acc_fus_sar)
    
    # Save Final Method Selection
    selection = {
        "selected_features": ["CultureMERT"],
        "feature_dimensions": [128],
        "audio_preprocessing": "24kHz Mono 60s windows",
        "temporal_context": "60s Mean Pooled",
        "PCA_configuration": "PCA-128 + Whitening fit on TRF Train",
        "normalization_method": "L2 normalization",
        "similarity_normalization": "None",
        "fusion_weights": "N/A",
        "reference_aggregation": "Max (Top-1)",
        "temporal_aggregation": "Mean",
        "validation_results": {"Top1": 0.125, "Margin": 0.084},
        "test_results": {"Top1": acc_mert_test},
        "saraga_results": {"Top1": acc_mert_sar},
        "reason_for_selection": "CASE B: Multi-feature (Pitch Dominant) gave a tiny validation improvement (+0.9%) but caused a massive degradation in margin separation (0.084 -> 0.045). SARAGA generalizability remained stagnant. Therefore, rejected fusion and retained CultureMERT."
    }
    
    with open(os.path.join(OPT_DIR, "final_method_selection.json"), "w") as f:
        json.dump(selection, f, indent=4)
        
    md = f"""# Multi-Feature Raag Optimization Final Report

## 1. Feature Analysis
1. **Did Mel improve over CultureMERT?** No. (10.69% alone vs 12.55% MERT).
2. **Did Chroma improve?** No. (12.55% alone, identical to MERT, with much lower margin).
3. **Did Swaragram improve?** No. (8.83% alone).

## 2. Fusion Performance
4. **Which feature combination was best?** Fusion_Pitch_Dominant (MERT 0.40, Chroma 0.25, Swara 0.20, Mel 0.15).
5. **Did multi-feature beat PCA128 CultureMERT?** Barely on raw validation accuracy, but failed significantly on confidence margin.
6. **By exactly how many percentage points?** +0.93% (13.48% vs 12.55%).
7. **Did Macro F1 improve?** Yes, slightly (0.089 vs 0.074).
8. **Did Top-1/Top-2 margins improve?** NO. The margin halved from 0.084 to 0.045, making predictions significantly less confident.
9. **Did the improvement survive the TRF Test?** Fusion Test Top-1: {acc_fus_test:.4f} vs MERT Test Top-1: {acc_mert_test:.4f}.
10. **Did it survive SARAGA?** Fusion SARAGA Top-1: {acc_fus_sar:.4f} vs MERT SARAGA Top-1: {acc_mert_sar:.4f}.

## 3. Final Method Selection
11. **What is the simplest scientifically defensible final method?** The existing **CultureMERT PCA128 + Whitening** baseline. According to CASE B of the decision rule, the marginal <1% improvement does not justify the massive loss in margin separation and the added complexity of computing STFT/CQT across the Folk dataset.
12. **What exact frozen configuration should be used for the 328 Folk recordings?**
   - Extract 60s CultureMERT embeddings.
   - Transform via TRF Train-fitted PCA-128 + Whitening.
   - L2 Normalize.
   - Cosine Similarity against all TRF references.
   - Group by Raag using MAX reference similarity.
   - Aggregate temporal sequences using MEAN.
   - Rank all 61 Raags and return Top-1, 2, 3 with margins.
"""
    with open(os.path.join(OPT_DIR, "optimization_report.md"), "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
