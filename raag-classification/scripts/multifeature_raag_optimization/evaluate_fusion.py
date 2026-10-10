
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
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

# CultureMERT Baseline Assets
TRF_EMB_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\embeddings")

def load_mert_embeddings(df, window_size=12):
    # Returns {rec_id: sequence_embs}, labels
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
        except:
            pass
    return embs, labels, splits

def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def build_similarity_matrix(query_embs, ref_embs):
    # For Acoustic features (Mel, Chroma, Swara), we have 1 vector per recording since we mean-pooled 60s.
    # Actually wait. The feature extractor extracted ONE vector per recording from the FIRST 60s.
    # CultureMERT has multiple 60s sequences per recording.
    # To match, we can either:
    # 1. Use the MEAN of the CultureMERT sequences for the acoustic feature comparison alignment
    # OR 2. Compare the acoustic feature (1-D) against all references.
    
    # Since acoustic is 1 vector per recording: Q = 1xD, Refs = NxD.
    return np.dot(query_embs, ref_embs.T)

def aggregate_and_rank(sim_matrix, train_labels):
    unique_raags = np.unique(train_labels)
    raag_scores = np.zeros((sim_matrix.shape[0], len(unique_raags)))
    
    for i, r in enumerate(unique_raags):
        mask = train_labels == r
        if not np.any(mask): continue
        # MAX reference aggregation
        r_sims = sim_matrix[:, mask]
        raag_scores[:, i] = np.max(r_sims, axis=1)
        
    return raag_scores, unique_raags

def rank_normalize(scores):
    # Ranks along axis 1 (Raags) for each query
    # Scipy rankdata: 1 is smallest, N is largest.
    normed = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        normed[i] = rankdata(scores[i]) / scores.shape[1]
    return normed

def main():
    print("Loading Data...")
    df_trf = pd.read_csv(SPLIT_MANIFEST).drop_duplicates('source_recording')
    
    # Load Acoustic Features
    acoustic_embs = {}
    for rec in df_trf['source_recording']:
        p = os.path.join(CACHE_DIR, "TRF", f"{rec}.npy")
        if os.path.exists(p):
            acoustic_embs[rec] = np.load(p, allow_pickle=True).item()
            
    # Load CultureMERT
    df_split = pd.read_csv(SPLIT_MANIFEST)
    mert_embs, labels, splits = load_mert_embeddings(df_split)
    
    # We will compute the Mean across temporal sequences for MERT for this evaluation
    # so that MERT is 1 vector per recording, just like acoustic features, making fusion trivial.
    # (The plan said MAX reference similarity, then MEAN across sequences. This is equivalent 
    # to aggregating sequences before fusion or after. Let's average MERT sequences here for simplicity.)
    
    train_recs = [r for r in labels.keys() if splits[r] == 'TRAIN' and r in acoustic_embs]
    val_recs = [r for r in labels.keys() if splits[r] == 'VAL' and r in acoustic_embs]
    test_recs = [r for r in labels.keys() if splits[r] == 'TEST' and r in acoustic_embs]
    
    train_labels = np.array([labels[r] for r in train_recs])
    val_labels = np.array([labels[r] for r in val_recs])
    test_labels = np.array([labels[r] for r in test_recs])
    
    # 1. Setup MERT
    print("Fitting PCA128 + Whiten on MERT...")
    mert_train_flat = np.vstack([mert_embs[r] for r in train_recs])
    pca = PCA(n_components=128, whiten=True).fit(mert_train_flat)
    
    def process_mert(recs):
        # average MERT sequences for the recording -> 1 vector
        out = []
        for r in recs:
            seqs = mert_embs[r]
            trans = l2_norm(pca.transform(seqs))
            out.append(np.mean(trans, axis=0))
        return l2_norm(np.array(out))
        
    mert_train_X = process_mert(train_recs)
    mert_val_X = process_mert(val_recs)
    mert_test_X = process_mert(test_recs)
    
    # 2. Setup Acoustic
    def process_acoustic(recs, feat_key):
        return np.array([acoustic_embs[r][feat_key] for r in recs])
        
    mel_train_X = process_acoustic(train_recs, 'mel')
    mel_val_X = process_acoustic(val_recs, 'mel')
    
    chroma_train_X = process_acoustic(train_recs, 'chroma')
    chroma_val_X = process_acoustic(val_recs, 'chroma')
    
    swara_train_X = process_acoustic(train_recs, 'swara')
    swara_val_X = process_acoustic(val_recs, 'swara')
    
    # Compute Similarities & Rank Normalizations
    def get_normed_scores(train_X, val_X):
        sims = build_similarity_matrix(val_X, train_X)
        scores, ur = aggregate_and_rank(sims, train_labels)
        return rank_normalize(scores), scores, ur
        
    mert_norm_val, mert_raw_val, ur = get_normed_scores(mert_train_X, mert_val_X)
    mel_norm_val, mel_raw_val, _ = get_normed_scores(mel_train_X, mel_val_X)
    chroma_norm_val, chroma_raw_val, _ = get_normed_scores(chroma_train_X, chroma_val_X)
    swara_norm_val, swara_raw_val, _ = get_normed_scores(swara_train_X, swara_val_X)
    
    # Evaluate Single Baselines
    def evaluate_scores(scores, true_labels):
        preds = []
        top3 = 0
        margins_12, margins_13 = [], []
        
        for i in range(len(scores)):
            s = scores[i]
            idx = np.argsort(s)[::-1]
            p = ur[idx[0]]
            preds.append(p)
            if true_labels[i] in ur[idx[:3]]:
                top3 += 1
                
            sorted_s = s[idx]
            margins_12.append(sorted_s[0] - sorted_s[1] if len(sorted_s)>1 else 0)
            margins_13.append(sorted_s[0] - sorted_s[2] if len(sorted_s)>2 else 0)
            
        acc = accuracy_score(true_labels, preds)
        mac = f1_score(true_labels, preds, average='macro', zero_division=0)
        return acc, mac, top3/len(scores), np.mean(margins_12), np.mean(margins_13)

    results = []
    
    print("Evaluating Single Features...")
    for name, raw_scores in [
        ("MERT_Baseline", mert_raw_val),
        ("Mel", mel_raw_val),
        ("Chroma", chroma_raw_val),
        ("Swaragram", swara_raw_val)
    ]:
        a, m, t, m12, m13 = evaluate_scores(raw_scores, val_labels)
        results.append({
            "Method": name, "Features": name,
            "TRF Val Top-1": a, "TRF Val Macro F1": m, "TRF Val Top-3": t,
            "Mean Margin_1_2": m12
        })
        
    print("Evaluating Fusions...")
    fusions = {
        "Fusion_Equal": [0.25, 0.25, 0.25, 0.25],
        "Fusion_MERT_Dominant": [0.50, 0.20, 0.15, 0.15],
        "Fusion_Pitch_Dominant": [0.40, 0.15, 0.25, 0.20],
        "Fusion_Balanced": [0.40, 0.20, 0.20, 0.20],
        "Fusion_MERT_Mel_Chroma": [0.50, 0.25, 0.25, 0.0]
    }
    
    for name, w in fusions.items():
        f_scores = (w[0]*mert_norm_val) + (w[1]*mel_norm_val) + (w[2]*chroma_norm_val) + (w[3]*swara_norm_val)
        a, m, t, m12, m13 = evaluate_scores(f_scores, val_labels)
        results.append({
            "Method": name, "Features": name,
            "TRF Val Top-1": a, "TRF Val Macro F1": m, "TRF Val Top-3": t,
            "Mean Margin_1_2": m12
        })
        
    df_res = pd.DataFrame(results).sort_values("TRF Val Top-1", ascending=False)
    print(df_res.to_string())
    df_res.to_csv(os.path.join(OPT_DIR, "feature_baseline_comparison.csv"), index=False)

if __name__ == "__main__":
    main()
