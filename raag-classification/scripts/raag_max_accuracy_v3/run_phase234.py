
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_max_accuracy_v3")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]
sar_map = pd.read_csv(SARAGA_MAPPING)
if 'evaluation_eligibility' in sar_map.columns:
    sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
else:
    sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]

ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}

def compute_features(cents_seq):
    voiced_ratio = np.sum(~np.isnan(cents_seq)) / float(len(cents_seq))
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    
    # Stability
    diffs = np.diff(cents)
    stability = 1.0 / (np.std(diffs) + 1.0)
    
    # Quantized
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    
    # N3, N4
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    n3 = n3 / (np.sum(n3) + 1e-9)
    
    n4 = np.zeros(12**4, dtype=np.float32)
    for i in range(len(pc) - 3): n4[pc[i]*1728 + pc[i+1]*144 + pc[i+2]*12 + pc[i+3]] += 1
    n4 = n4 / (np.sum(n4) + 1e-9)
    
    # Phrase Direction
    asc = np.sum(diffs > 50)
    desc = np.sum(diffs < -50)
    stable = np.sum(np.abs(diffs) <= 50)
    total = len(diffs) + 1e-9
    dirs = np.array([asc/total, desc/total, stable/total], dtype=np.float32)
    
    return {
        'n3': n3, 'n4': n4, 'dirs': dirs,
        'quality_score': voiced_ratio * stability
    }

def load_data(k, is_saraga=False):
    st = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    sp = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    sf = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    
    t_path = os.path.join(OLD_FEAT_DIR, f"{k}{st}")
    p_path = os.path.join(OLD_FEAT_DIR, f"{k}{sp}")
    f_path = os.path.join(FEAT_DIR, f"{k}{sf}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    feats = [compute_features(seq) for seq in f0]
    
    n3_l, n4_l, dir_l, qs_l = [], [], [], []
    valid_idx = []
    for i, f in enumerate(feats):
        if f is not None:
            n3_l.append(f['n3']); n4_l.append(f['n4']); dir_l.append(f['dirs']); qs_l.append(f['quality_score'])
            valid_idx.append(i)
            
    min_len = min(len(t), len(p), len(feats))
    t = t[:min_len]
    p = p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    n4_l = [n4_l[i] for i in range(len(valid_idx))]
    dir_l = [dir_l[i] for i in range(len(valid_idx))]
    qs_l = [qs_l[i] for i in range(len(valid_idx))]
    
    if not n3_l: return None
    
    t = t[valid_idx]
    p = p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    
    return {
        't': t, 'p': p,
        'n3': np.array(n3_l), 'n4': np.array(n4_l), 'dirs': np.array(dir_l),
        'qs': np.array(qs_l)
    }

train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

print("Loading data...")
ref_t, ref_p, ref_n3, ref_n4, ref_dirs, ref_r = [], [], [], [], [], []
for k in train_keys:
    d = load_data(k)
    if d is not None:
        ref_t.append(d['t'].mean(axis=0)); ref_p.append(d['p'].mean(axis=0))
        ref_n3.append(d['n3'].mean(axis=0)); ref_n4.append(d['n4'].mean(axis=0))
        ref_dirs.append(d['dirs'].mean(axis=0))
        ref_r.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])

ref_t = np.array(ref_t) / (np.linalg.norm(ref_t, axis=1, keepdims=True) + 1e-9)
ref_p = np.array(ref_p) / (np.linalg.norm(ref_p, axis=1, keepdims=True) + 1e-9)
ref_n3 = np.array(ref_n3) / (np.linalg.norm(ref_n3, axis=1, keepdims=True) + 1e-9)
ref_n4 = np.array(ref_n4) / (np.linalg.norm(ref_n4, axis=1, keepdims=True) + 1e-9)
ref_dirs = np.array(ref_dirs) / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-9)
ref_r = np.array(ref_r)

def get_sims(q, ref):
    if len(q) == 0: return np.zeros((0, 61))
    s = cosine_similarity(q, ref)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(ref_r == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
    return zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)

def evaluate(keys, w_t, w_p, w_n3, w_n4, w_dir, agg_method, is_saraga=False):
    true = []
    sims = []
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        
        sim_t = get_sims(d['t'], ref_t)
        sim_p = get_sims(d['p'], ref_p)
        fused = w_t * sim_t + w_p * sim_p
        
        if w_n3 > 0: fused += w_n3 * get_sims(d['n3'], ref_n3)
        if w_n4 > 0: fused += w_n4 * get_sims(d['n4'], ref_n4)
        if w_dir > 0: fused += w_dir * get_sims(d['dirs'], ref_dirs)
        
        if agg_method == 'Top-50%':
            num_seq = max(1, int(len(fused) * 0.5))
            if len(fused) <= num_seq: final_sim = np.mean(fused, axis=0)
            else: final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
        elif agg_method == 'Soft-Quality':
            qs = d['qs'] / (np.sum(d['qs']) + 1e-9)
            final_sim = np.sum(fused * qs[:, None], axis=0)
            
        sims.append(final_sim)
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        true.append(RAAG_TO_IDX[mapped])
        
    return np.mean(np.argmax(sims, axis=1) == true)

print("Running Temporal/N-Gram/Direction evaluations...")
results = []
# Control Baseline
v_base = evaluate(val_keys, 0.75, 0.20, 0.05, 0, 0, 'Top-50%', False)
s_base = evaluate(sar_est_keys, 0.75, 0.20, 0.05, 0, 0, 'Top-50%', True)
results.append({'Config': '75T+20P+5N3 (Top-50%)', 'Val': v_base, 'SAR': s_base})

# Soft Weighting
v_soft = evaluate(val_keys, 0.75, 0.20, 0.05, 0, 0, 'Soft-Quality', False)
s_soft = evaluate(sar_est_keys, 0.75, 0.20, 0.05, 0, 0, 'Soft-Quality', True)
results.append({'Config': '75T+20P+5N3 (Soft-Quality)', 'Val': v_soft, 'SAR': s_soft})

# Longer N-grams (N4)
for w4 in [0.05, 0.10]:
    v = evaluate(val_keys, 0.75, 0.20-w4, 0.05, w4, 0, 'Top-50%', False)
    s = evaluate(sar_est_keys, 0.75, 0.20-w4, 0.05, w4, 0, 'Top-50%', True)
    results.append({'Config': f'75T+{int((0.20-w4)*100)}P+5N3+{int(w4*100)}N4 (Top-50%)', 'Val': v, 'SAR': s})
    
# Direction features
for wd in [0.05, 0.10]:
    v = evaluate(val_keys, 0.75, 0.20-wd, 0.05, 0, wd, 'Top-50%', False)
    s = evaluate(sar_est_keys, 0.75, 0.20-wd, 0.05, 0, wd, 'Top-50%', True)
    results.append({'Config': f'75T+{int((0.20-wd)*100)}P+5N3+{int(wd*100)}Dir (Top-50%)', 'Val': v, 'SAR': s})

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv(os.path.join(WORKSPACE, "phrase_feature_results.csv"), index=False)
