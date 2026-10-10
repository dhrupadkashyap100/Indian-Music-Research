
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")
FOLK_FEAT_DIR = os.path.join(WORKSPACE, "features")

OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")

# Load config
with open(os.path.join(WORKSPACE, "production_model_config.json"), "r") as f:
    config = json.load(f)

print("Phase 0: Loading TRF Split Manifest...")
trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]

ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}
IDX_TO_RAAG = {i: r for r, i in RAAG_TO_IDX.items()}

# ==========================================================
# 1. TRF TRAIN FEATURE EXTRACTION (Reference & SVM Fit)
# ==========================================================

def compute_n3(cents_seq):
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    return n3 / (np.sum(n3) + 1e-9)

def load_trf_data(k):
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}_trans.npy"), os.path.join(OLD_FEAT_DIR, f"{k}_pc.npy"), os.path.join(FEAT_DIR, f"{k}_f0.npy")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path, allow_pickle=True)
    n3_l, valid_idx = [], []
    for i, seq in enumerate(f0):
        n3 = compute_n3(np.array(seq, dtype=float))
        if n3 is not None:
            n3_l.append(n3); valid_idx.append(i)
    min_len = min(len(t), len(p), len(f0))
    t, p = t[:min_len], p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    if not valid_idx: return None
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    t, p = t[valid_idx], p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return {'t': t, 'p': p, 'n3': np.array(n3_l)}

print("Loading TRF Train Reference Bank...")
train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
rt_l, rp_l, rn3_l, rr_l = [], [], [], []
train_data = {}
for k in train_keys:
    d = load_trf_data(k)
    if d is not None:
        train_data[k] = d
        rt_l.append(d['t'].mean(axis=0))
        rp_l.append(d['p'].mean(axis=0))
        rn3_l.append(d['n3'].mean(axis=0))
        rr_l.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])

rt = np.array(rt_l) / (np.linalg.norm(rt_l, axis=1, keepdims=True) + 1e-9)
rp = np.array(rp_l) / (np.linalg.norm(rp_l, axis=1, keepdims=True) + 1e-9)
rn3 = np.array(rn3_l) / (np.linalg.norm(rn3_l, axis=1, keepdims=True) + 1e-9)
rr = np.array(rr_l)

def get_scaled_features(d):
    return np.hstack([0.75 * d['t'], 0.20 * d['p'], 0.05 * d['n3']])

print("Training Frozen RBF-SVM...")
X_train_seqs, y_train_seqs = [], []
for k in train_keys:
    if k in train_data:
        feat = get_scaled_features(train_data[k])
        X_train_seqs.extend(feat)
        y_train_seqs.extend([RAAG_TO_IDX[trf_meta.loc[k, 'raag']]] * len(feat))
X_train_seqs = np.array(X_train_seqs)
y_train_seqs = np.array(y_train_seqs)

svm = SVC(C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_seqs, y_train_seqs)

# ==========================================================
# 2. FOLK INFERENCE PIPELINE
# ==========================================================
def load_folk_data(k):
    t_path, p_path, f_path = os.path.join(FOLK_FEAT_DIR, f"{k}_trans.npy"), os.path.join(FOLK_FEAT_DIR, f"{k}_pc.npy"), os.path.join(FOLK_FEAT_DIR, f"{k}_f0.npy")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path, allow_pickle=True)
    n3_l, valid_idx = [], []
    for i, seq in enumerate(f0):
        n3 = compute_n3(np.array(seq, dtype=float))
        if n3 is not None:
            n3_l.append(n3); valid_idx.append(i)
    min_len = min(len(t), len(p), len(f0))
    t, p = t[:min_len], p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    if not valid_idx: return None
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    t, p = t[valid_idx], p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return {'t': t, 'p': p, 'n3': np.array(n3_l)}

folk_files = glob.glob(os.path.join(FOLK_FEAT_DIR, "*_trans.npy"))
folk_keys = [os.path.basename(f).replace("_trans.npy", "") for f in folk_files]
print(f"Extracted {len(folk_keys)} Folk feature sets.")

predictions = []
confidence_analysis = []
agreement_analysis = []
ambiguous_pairs = []
raw_distributions = []

for rec_id in folk_keys:
    d = load_folk_data(rec_id)
    if d is None: continue
    
    # Assert dimensions
    assert d['t'].shape[1] == 144
    assert d['p'].shape[1] == 12
    assert d['n3'].shape[1] == 1728
    
    # 1. Geometric Score
    sim_t = cosine_similarity(d['t'], rt)
    sim_p = cosine_similarity(d['p'], rp)
    sim_n3 = cosine_similarity(d['n3'], rn3)
    raw = 0.75 * sim_t + 0.20 * sim_p + 0.05 * sim_n3
    rs = np.zeros((len(d['t']), 61))
    for r in range(61):
        idx = np.where(rr == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(raw[:, idx], axis=1)
    rs = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)
    num_seq = max(1, int(len(rs) * 0.5))
    final = np.mean(rs[np.argsort(np.max(rs, axis=1))[-num_seq:]], axis=0) if len(rs) > num_seq else np.mean(rs, axis=0)
    e_x = np.exp(final - np.max(final))
    pg = e_x / e_x.sum()
    
    # 2. SVM Score
    X = get_scaled_features(d)
    ps = np.mean(svm.predict_proba(X), axis=0)
    
    # Assert probability shape and sum
    assert ps.shape == (61,)
    assert not np.isnan(ps).any()
    assert np.isclose(np.sum(ps), 1.0, atol=1e-3)
    
    # 3. Hybrid
    ph = 0.75 * pg + 0.25 * ps
    
    # Ranks
    g_top1 = IDX_TO_RAAG[np.argmax(pg)]
    s_top1 = IDX_TO_RAAG[np.argmax(ps)]
    h_args = np.argsort(ph)[::-1]
    
    h_top1 = IDX_TO_RAAG[h_args[0]]
    h_top2 = IDX_TO_RAAG[h_args[1]]
    h_top3 = IDX_TO_RAAG[h_args[2]]
    
    s1 = float(ph[h_args[0]])
    s2 = float(ph[h_args[1]])
    s3 = float(ph[h_args[2]])
    margin = s1 - s2
    
    # 4. Confidence Tier
    if margin > 0.12:
        tier = "HIGH"
    elif margin > 0.05:
        tier = "MODERATE"
    else:
        tier = "AMBIGUOUS"
        
    # 5. Agreement
    if g_top1 == s_top1 == h_top1:
        agree = "AGREEMENT"
    elif g_top1 != s_top1 and h_top1 in [g_top1, s_top1]:
        agree = "HYBRID RESOLUTION"
    elif h_top1 == s_top1:
        agree = "SVM DOMINANT"
    elif h_top1 == g_top1:
        agree = "GEOMETRIC DOMINANT"
    else:
        agree = "DISAGREEMENT"
        
    if margin < 0.02:
        agree = "HIGH UNCERTAINTY"
        
    predictions.append({
        'recording_id': rec_id,
        'predicted_raag_top1': h_top1,
        'predicted_raag_top2': h_top2,
        'predicted_raag_top3': h_top3,
        'hybrid_top1_score': s1,
        'hybrid_top2_score': s2,
        'hybrid_top3_score': s3,
        'hybrid_score_margin': margin,
        'confidence_tier': tier,
        'geometric_top1': g_top1,
        'svm_top1': s_top1,
        'models_agree': agree,
        'tonic_estimate': "Computed",
        'processing_status': "SUCCESS"
    })
    
    confidence_analysis.append({
        'recording_id': rec_id,
        'Top-1 Raag': h_top1,
        'Top-2 Raag': h_top2,
        'Top-3 Raag': h_top3,
        'hybrid_margin': margin,
        'confidence_tier': tier,
        'geometric_score': float(np.max(pg)),
        'svm_probability': float(np.max(ps)),
        'hybrid_score': s1,
        'model_agreement': agree
    })
    
    agreement_analysis.append({
        'recording_id': rec_id,
        'geometric_top1': g_top1,
        'svm_top1': s_top1,
        'hybrid_top1': h_top1,
        'relationship': agree
    })
    
    # Save the distribution for analysis
    raw_distributions.append(h_top1)
    
    if tier == "AMBIGUOUS":
        pair = tuple(sorted([h_top1, h_top2]))
        ambiguous_pairs.append({
            'Raag A': pair[0],
            'Raag B': pair[1],
            'margin': margin,
            'top1_score': s1,
            'confidence': tier
        })

print("Generating DataFrames...")
df_pred = pd.DataFrame(predictions)
df_conf = pd.DataFrame(confidence_analysis)
df_agree = pd.DataFrame(agreement_analysis)

df_pred.to_csv(os.path.join(WORKSPACE, "folk_raag_predictions.csv"), index=False)
df_conf.to_csv(os.path.join(WORKSPACE, "folk_confidence_analysis.csv"), index=False)
df_agree.to_csv(os.path.join(WORKSPACE, "folk_model_agreement_analysis.csv"), index=False)

# Raag Distribution
from collections import Counter
c = Counter(raw_distributions)
dist_data = []
for r, count in c.items():
    sub = df_pred[df_pred['predicted_raag_top1'] == r]
    dist_data.append({
        'Raag': r,
        'Predicted_Count': count,
        'Percentage': count / len(df_pred) * 100,
        'HIGH_Count': sum(sub['confidence_tier'] == 'HIGH'),
        'MODERATE_Count': sum(sub['confidence_tier'] == 'MODERATE'),
        'AMBIGUOUS_Count': sum(sub['confidence_tier'] == 'AMBIGUOUS')
    })
df_dist = pd.DataFrame(dist_data).sort_values('Predicted_Count', ascending=False)
df_dist.to_csv(os.path.join(WORKSPACE, "folk_raag_distribution.csv"), index=False)

# Ambiguous Pairs
if len(ambiguous_pairs) > 0:
    df_amb = pd.DataFrame(ambiguous_pairs)
    amb_sum = df_amb.groupby(['Raag A', 'Raag B']).agg({
        'margin': ['count', 'mean'],
        'top1_score': 'mean'
    }).reset_index()
    amb_sum.columns = ['Raag A', 'Raag B', 'Occurrences', 'Average_Margin', 'Average_Top1_Score']
    amb_sum = amb_sum.sort_values('Occurrences', ascending=False)
    amb_sum.to_csv(os.path.join(WORKSPACE, "folk_ambiguous_raag_pairs.csv"), index=False)
else:
    pd.DataFrame(columns=['Raag A', 'Raag B', 'Occurrences', 'Average_Margin', 'Average_Top1_Score']).to_csv(os.path.join(WORKSPACE, "folk_ambiguous_raag_pairs.csv"), index=False)

# Manual Review Priority
# Score: AMBIGUOUS (3), VERY SMALL MARGIN < 0.02 (3), DISAGREEMENT (2)
df_review = df_pred.copy()
df_review['priority_score'] = 0
df_review.loc[df_review['confidence_tier'] == 'AMBIGUOUS', 'priority_score'] += 3
df_review.loc[df_review['hybrid_score_margin'] < 0.02, 'priority_score'] += 3
df_review.loc[df_review['models_agree'].isin(['HYBRID RESOLUTION', 'DISAGREEMENT']), 'priority_score'] += 2
df_review = df_review.sort_values('priority_score', ascending=False)
df_review.to_csv(os.path.join(WORKSPACE, "folk_manual_review_priority.csv"), index=False)

# JSON Summary
summary = {
    "total_recordings": len(df_pred),
    "model_version": "Hybrid SVM B",
    "feature_dimension": 1884,
    "geometric_weight": 0.75,
    "svm_weight": 0.25,
    "svm_C": 10,
    "svm_gamma": "scale",
    "high_confidence_count": int(sum(df_pred['confidence_tier'] == 'HIGH')),
    "moderate_confidence_count": int(sum(df_pred['confidence_tier'] == 'MODERATE')),
    "ambiguous_count": int(sum(df_pred['confidence_tier'] == 'AMBIGUOUS')),
    "model_agreement_count": int(sum(df_pred['models_agree'] == 'AGREEMENT')),
    "model_disagreement_count": int(sum(df_pred['models_agree'] != 'AGREEMENT')),
    "top_predicted_raags": df_dist.head(5).to_dict('records')
}
with open(os.path.join(WORKSPACE, "folk_inference_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("INFERENCE COMPLETE.")
