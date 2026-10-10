
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
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_final_classical_models")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")

# 1. Load Data
trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]

ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}

def compute_n3(cents_seq):
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    return n3 / (np.sum(n3) + 1e-9)

def load_data(k):
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}_trans.npy"), os.path.join(OLD_FEAT_DIR, f"{k}_pc.npy"), os.path.join(FEAT_DIR, f"{k}_f0.npy")
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

train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
test_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TEST'] # UNSEALING TEST

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

def get_scaled_features(d):
    return np.hstack([0.75 * d['t'], 0.20 * d['p'], 0.05 * d['n3']])

X_train_seqs, y_train_seqs = [], []
for k in train_keys:
    if k in train_data:
        d = train_data[k]
        feat = get_scaled_features(d)
        X_train_seqs.extend(feat)
        y_train_seqs.extend([RAAG_TO_IDX[trf_meta.loc[k, 'raag']]] * len(feat))
X_train_seqs = np.array(X_train_seqs)
y_train_seqs = np.array(y_train_seqs)

# Train the winning SVM
print("Training final Frozen SVM on TRAIN...")
svm = SVC(C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_seqs, y_train_seqs)

# Evaluate on SEALED TEST
print("Evaluating on SEALED TRF TEST...")
true, c_probs, s_probs = [], [], []

for k in test_keys:
    d = load_data(k)
    if d is None: continue
    
    true.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
    
    # Control Probability
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
    c_prob = e_x / e_x.sum()
    c_probs.append(c_prob)
    
    # SVM Probability
    X = get_scaled_features(d)
    seq_probs = svm.predict_proba(X)
    s_prob = np.mean(seq_probs, axis=0)
    s_probs.append(s_prob)

true = np.array(true)
c_probs = np.array(c_probs)
s_probs = np.array(s_probs)

def get_metrics(probs):
    preds = np.argmax(probs, axis=1)
    top1 = np.mean(preds == true)
    top3 = sum(true[i] in np.argsort(probs[i])[-3:] for i in range(len(true))) / len(true)
    f1 = f1_score(true, preds, average='macro')
    margins = np.mean([np.sort(p)[-1] - np.sort(p)[-2] for p in probs])
    return float(top1), float(top3), float(f1), float(margins)

ct1, ct3, cf1, cm = get_metrics(c_probs)
# Hybrid 75/25
h_probs = 0.75 * c_probs + 0.25 * s_probs
ht1, ht3, hf1, hm = get_metrics(h_probs)

# Bootstrap CI for Hybrid
n_boot = 1000
h_preds = np.argmax(h_probs, axis=1)
boot_accs = []
for _ in range(n_boot):
    idx = np.random.randint(0, len(true), len(true))
    boot_accs.append(np.mean(h_preds[idx] == true[idx]))
ci_lower, ci_upper = np.percentile(boot_accs, [2.5, 97.5])

result = {
    'Control_Test_Top1': ct1,
    'Hybrid_Test_Top1': ht1,
    'Hybrid_Test_Top3': ht3,
    'Hybrid_Test_Macro_F1': hf1,
    'Hybrid_Test_Margin': hm,
    'Hybrid_95_CI': [float(ci_lower), float(ci_upper)]
}

if ht1 > ct1:
    result['STATUS'] = "HYBRID WINS - NEW FROZEN MODEL"
else:
    result['STATUS'] = "HYBRID FAILED ON TEST - RETAIN 43.10% CONTROL"
    
with open(os.path.join(WORKSPACE, "sealed_test_results.json"), "w") as f:
    json.dump(result, f, indent=4)

print("Control Top-1:", ct1)
print("Hybrid Top-1:", ht1)
print("Status:", result['STATUS'])
