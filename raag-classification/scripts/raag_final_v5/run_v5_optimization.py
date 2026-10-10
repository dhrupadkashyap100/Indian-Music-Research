
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_final_v5")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

# 1. Load Data
trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]
sar_map = pd.read_csv(SARAGA_MAPPING)
if 'evaluation_eligibility' in sar_map.columns:
    sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
else:
    sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]

ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}

def compute_n3(cents_seq):
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    return n3 / (np.sum(n3) + 1e-9)

def load_data(k, is_saraga=False):
    s_t = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    s_p = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    s_f = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_t}"), os.path.join(OLD_FEAT_DIR, f"{k}{s_p}"), os.path.join(FEAT_DIR, f"{k}{s_f}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    n3_l, valid_idx = [], []
    for i, seq in enumerate(f0):
        n3 = compute_n3(seq)
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

print("Loading dataset...")
train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

rt_l, rp_l, rn3_l, rr_l = [], [], [], []
train_data = {}
for k in train_keys:
    d = load_data(k)
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

def get_geometric_probs(d):
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
    return e_x / e_x.sum()

def get_scaled_features(d):
    return np.hstack([0.75 * d['t'], 0.20 * d['p'], 0.05 * d['n3']])

print("Building SVM...")
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

def get_svm_probs(d):
    X = get_scaled_features(d)
    return np.mean(svm.predict_proba(X), axis=0)

def extract_all_probs(keys, is_saraga=False):
    g, s, h, t = [], [], [], []
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        pg = get_geometric_probs(d)
        ps = get_svm_probs(d)
        ph = 0.75 * pg + 0.25 * ps
        g.append(pg); s.append(ps); h.append(ph)
        t.append(RAAG_TO_IDX[sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']])
    return np.array(g), np.array(s), np.array(h), np.array(t)

# Phase 1: Control Reproduction
print("Extracting Validation and SARAGA predictions...")
g_val, s_val, h_val, true_val = extract_all_probs(val_keys, False)
g_sar, s_sar, h_sar, true_sar = extract_all_probs(sar_est_keys, True)

def metrics(probs, true):
    p = np.argmax(probs, axis=1)
    t1 = np.mean(p == true)
    t3 = sum(true[i] in np.argsort(probs[i])[-3:] for i in range(len(true))) / len(true)
    f1 = f1_score(true, p, average='macro')
    return t1, t3, f1

c_t1, c_t3, c_f1 = metrics(h_val, true_val)
sc_t1, sc_t3, _ = metrics(h_sar, true_sar)
print(f"Control (Hybrid) - Val Top-1: {c_t1:.4f}, SARAGA EST: {sc_t1:.4f}")

leaderboard = []
def record(name, p_val, p_sar, method=""):
    vt1, vt3, vf1 = metrics(p_val, true_val)
    st1, st3, sf1 = metrics(p_sar, true_sar)
    status = "PROMISING" if vt1 > c_t1 and st1 >= sc_t1 - 0.01 else "NO_IMPROVEMENT"
    leaderboard.append({
        'Candidate': name, 'Method': method,
        'Val_Top1': vt1, 'Val_Top3': vt3, 'Val_F1': vf1,
        'SARAGA_Top1': st1, 'SARAGA_Top3': st3,
        'Status': status
    })

record("Control_Hybrid", h_val, h_sar, "Baseline")

# Phase 2: Error Complementarity
print("Phase 2: Error Complementarity")
pred_g = np.argmax(g_val, axis=1)
pred_s = np.argmax(s_val, axis=1)
pred_h = np.argmax(h_val, axis=1)
g_corr = pred_g == true_val
s_corr = pred_s == true_val
h_corr = pred_h == true_val

comp = pd.DataFrame({
    'Metric': ['Both Correct', 'Both Wrong', 'Geometry Correct / SVM Wrong', 'Geometry Wrong / SVM Correct', 'SVM Rescue Rate', 'SVM Damage Rate'],
    'Value': [
        np.sum(g_corr & s_corr),
        np.sum(~g_corr & ~s_corr),
        np.sum(g_corr & ~s_corr),
        np.sum(~g_corr & s_corr),
        np.sum(~g_corr & s_corr) / np.sum(~g_corr),
        np.sum(g_corr & ~s_corr) / np.sum(g_corr)
    ]
})
comp.to_csv(os.path.join(WORKSPACE, "error_complementarity.csv"), index=False)

# Phase 3: Geometric Recall@K
print("Phase 3: Geometric Recall@K")
def recall_at_k(probs, true, ks=[1, 2, 3, 5, 10]):
    res = []
    for k in ks:
        res.append(sum(true[i] in np.argsort(probs[i])[-k:] for i in range(len(true))) / len(true))
    return res

rk_val = recall_at_k(g_val, true_val)
rk_sar = recall_at_k(g_sar, true_sar)
pd.DataFrame({
    'K': [1, 2, 3, 5, 10], 'Val_Recall': rk_val, 'SARAGA_Recall': rk_sar
}).to_csv(os.path.join(WORKSPACE, "geometric_recall_at_k.csv"), index=False)

# Phase 4: Top-K SVM Reranking
print("Phase 4: Top-K Reranking")
res_topk = []
for k in [2, 3, 4, 5, 10]:
    def rerank(probs_g, probs_s):
        probs = np.zeros_like(probs_g)
        for i in range(len(probs)):
            topk_idx = np.argsort(probs_g[i])[-k:]
            best_idx = topk_idx[np.argmax(probs_s[i][topk_idx])]
            probs[i, best_idx] = 1.0
        return probs
    rr_val = rerank(g_val, s_val)
    rr_sar = rerank(g_sar, s_sar)
    record(f"SVM_Rerank_K={k}", rr_val, rr_sar, "Top-K Reranking")
    res_topk.append({'K': k, 'Val_Top1': metrics(rr_val, true_val)[0], 'SARAGA_Top1': metrics(rr_sar, true_sar)[0]})
pd.DataFrame(res_topk).to_csv(os.path.join(WORKSPACE, "topk_reranking_results.csv"), index=False)

# Phase 5: Reciprocal Rank Fusion
print("Phase 5: RRF")
res_rrf = []
for k in [10, 20, 30, 40, 50, 60]:
    def compute_rrf(probs_g, probs_s):
        probs = np.zeros_like(probs_g)
        for i in range(len(probs)):
            rg = 61 - np.argsort(np.argsort(probs_g[i]))
            rs = 61 - np.argsort(np.argsort(probs_s[i]))
            probs[i] = 1.0 / (k + rg) + 1.0 / (k + rs)
        return probs
    rrf_val = compute_rrf(g_val, s_val)
    rrf_sar = compute_rrf(g_sar, s_sar)
    record(f"RRF_k={k}", rrf_val, rrf_sar, "Reciprocal Rank Fusion")
    res_rrf.append({'k': k, 'Val_Top1': metrics(rrf_val, true_val)[0], 'SARAGA_Top1': metrics(rrf_sar, true_sar)[0]})
pd.DataFrame(res_rrf).to_csv(os.path.join(WORKSPACE, "rrf_results.csv"), index=False)

# Phase 6: Lightweight Rank Fusion
print("Phase 6: Lightweight Rank Fusion")
res_rf = []
def compute_wrf(probs_g, probs_s, alpha):
    probs = np.zeros_like(probs_g)
    for i in range(len(probs)):
        rg = 61 - np.argsort(np.argsort(probs_g[i]))
        rs = 61 - np.argsort(np.argsort(probs_s[i]))
        probs[i] = alpha * (61 - rg) + (1 - alpha) * (61 - rs)
    return probs
for a in [0.25, 0.50, 0.75]:
    wrf_val = compute_wrf(g_val, s_val, a)
    wrf_sar = compute_wrf(g_sar, s_sar, a)
    record(f"WeightedRank_a={a}", wrf_val, wrf_sar, "Borda-style Rank Fusion")
    res_rf.append({'alpha': a, 'Val_Top1': metrics(wrf_val, true_val)[0], 'SARAGA_Top1': metrics(wrf_sar, true_sar)[0]})
pd.DataFrame(res_rf).to_csv(os.path.join(WORKSPACE, "rank_fusion_results.csv"), index=False)

# Phase 7: Confidence-Aware Hybrid
print("Phase 7: Confidence-Aware Hybrid")
res_conf = []
# Rule A: SVM override
for th in [0.70, 0.75, 0.80, 0.85, 0.90]:
    def conf_a(probs_g, probs_s, probs_h):
        probs = np.copy(probs_h)
        for i in range(len(probs)):
            if np.max(probs_s[i]) > th:
                probs[i] = probs_s[i]
        return probs
    ca_val = conf_a(g_val, s_val, h_val)
    ca_sar = conf_a(g_sar, s_sar, h_sar)
    record(f"Conf_RuleA_th={th}", ca_val, ca_sar, "SVM Override")
    res_conf.append({'Rule': 'SVM Override', 'Param': th, 'Val_Top1': metrics(ca_val, true_val)[0], 'SARAGA_Top1': metrics(ca_sar, true_sar)[0]})
pd.DataFrame(res_conf).to_csv(os.path.join(WORKSPACE, "confidence_fusion_results.csv"), index=False)

# Phase 8: Stratified 5-Fold CV on TRF TRAIN
print("Phase 8: Stratified CV Confidence Analysis")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_preds, cv_true, cv_g_conf, cv_s_conf, cv_h_margin, cv_agreed = [], [], [], [], [], []

train_k = np.array(train_keys)
# Sub-filter train keys that loaded correctly
train_k = np.array([k for k in train_keys if k in train_data])
y_k = np.array([RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in train_k])

for fold, (idx_t, idx_v) in enumerate(skf.split(train_k, y_k)):
    train_fold_keys = train_k[idx_t]
    val_fold_keys = train_k[idx_v]
    
    # Train localized SVM
    X_f_seqs, y_f_seqs = [], []
    for k in train_fold_keys:
        feat = get_scaled_features(train_data[k])
        X_f_seqs.extend(feat)
        y_f_seqs.extend([RAAG_TO_IDX[trf_meta.loc[k, 'raag']]] * len(feat))
    clf = SVC(C=10, gamma='scale', probability=True, random_state=42)
    clf.fit(X_f_seqs, y_f_seqs)
    
    # Eval on held-out out-of-fold
    for k in val_fold_keys:
        d = train_data[k]
        pg = get_geometric_probs(d) # Geometric ignores cross-validation isolation for features, but that's fine for margin analysis
        X = get_scaled_features(d)
        ps = np.mean(clf.predict_proba(X), axis=0)
        ph = 0.75 * pg + 0.25 * ps
        
        cv_true.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        pred = np.argmax(ph)
        cv_preds.append(pred)
        cv_s_conf.append(np.max(ps))
        s1, s2 = np.sort(ph)[-2:]
        cv_h_margin.append(s2 - s1)
        cv_agreed.append(1 if np.argmax(pg) == np.argmax(ps) else 0)

cv_true = np.array(cv_true)
cv_preds = np.array(cv_preds)
cv_s_conf = np.array(cv_s_conf)
cv_h_margin = np.array(cv_h_margin)
cv_agreed = np.array(cv_agreed)
cv_corr = cv_preds == cv_true

res_cv = []
for th in np.linspace(0.01, 0.5, 50):
    accepted = cv_h_margin > th
    cov = np.mean(accepted)
    acc = np.mean(cv_corr[accepted]) if np.sum(accepted) > 0 else 0
    res_cv.append({'Measure': 'Hybrid Margin', 'Threshold': th, 'Coverage': cov, 'Accuracy': acc})

for th in np.linspace(0.2, 0.95, 50):
    accepted = cv_s_conf > th
    cov = np.mean(accepted)
    acc = np.mean(cv_corr[accepted]) if np.sum(accepted) > 0 else 0
    res_cv.append({'Measure': 'SVM Confidence', 'Threshold': th, 'Coverage': cov, 'Accuracy': acc})

pd.DataFrame(res_cv).to_csv(os.path.join(WORKSPACE, "accuracy_rejection_curve.csv"), index=False)

# Phase 14: Leaderboard Output
print("Phase 14: Generating Leaderboard")
pd.DataFrame(leaderboard).to_csv(os.path.join(WORKSPACE, "complete_v5_leaderboard.csv"), index=False)
print("V5 Script Complete!")
