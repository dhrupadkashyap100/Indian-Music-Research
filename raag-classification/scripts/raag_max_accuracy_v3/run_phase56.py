
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
    diffs = np.diff(cents)
    stability = 1.0 / (np.std(diffs) + 1.0)
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    n3 = n3 / (np.sum(n3) + 1e-9)
    n4 = np.zeros(12**4, dtype=np.float32)
    for i in range(len(pc) - 3): n4[pc[i]*1728 + pc[i+1]*144 + pc[i+2]*12 + pc[i+3]] += 1
    n4 = n4 / (np.sum(n4) + 1e-9)
    return {'n3': n3, 'n4': n4, 'quality_score': voiced_ratio * stability}

def load_data(k, is_saraga=False):
    st = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    sp = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    sf = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}{st}"), os.path.join(OLD_FEAT_DIR, f"{k}{sp}"), os.path.join(FEAT_DIR, f"{k}{sf}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    feats = [compute_features(seq) for seq in f0]
    n3_l, n4_l, qs_l, valid_idx = [], [], [], []
    for i, f in enumerate(feats):
        if f is not None:
            n3_l.append(f['n3']); n4_l.append(f['n4']); qs_l.append(f['quality_score'])
            valid_idx.append(i)
    min_len = min(len(t), len(p), len(feats))
    t, p = t[:min_len], p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    if not valid_idx: return None
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    n4_l = [n4_l[i] for i in range(len(valid_idx))]
    qs_l = [qs_l[i] for i in range(len(valid_idx))]
    t, p = t[valid_idx], p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return {'t': t, 'p': p, 'n3': np.array(n3_l), 'n4': np.array(n4_l), 'qs': np.array(qs_l)}

train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

ref_data = {}
for k in train_keys:
    d = load_data(k)
    if d is not None:
        ref_data[k] = d

# Phase 5: Reference Quality Filtering
def build_refs(discard_bottom=0.0):
    rt, rp, rn3, rn4, rr = [], [], [], [], []
    for k, d in ref_data.items():
        if discard_bottom > 0:
            threshold = np.percentile(d['qs'], discard_bottom * 100)
            valid = d['qs'] >= threshold
            if np.sum(valid) == 0: valid = np.ones(len(d['qs']), dtype=bool)
        else:
            valid = np.ones(len(d['qs']), dtype=bool)
            
        rt.append(d['t'][valid].mean(axis=0))
        rp.append(d['p'][valid].mean(axis=0))
        rn3.append(d['n3'][valid].mean(axis=0))
        rn4.append(d['n4'][valid].mean(axis=0))
        rr.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        
    rt = np.array(rt) / (np.linalg.norm(rt, axis=1, keepdims=True) + 1e-9)
    rp = np.array(rp) / (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-9)
    rn3 = np.array(rn3) / (np.linalg.norm(rn3, axis=1, keepdims=True) + 1e-9)
    rn4 = np.array(rn4) / (np.linalg.norm(rn4, axis=1, keepdims=True) + 1e-9)
    return rt, rp, rn3, rn4, np.array(rr)

def get_sims(q, refs, rraags, calibrate=False):
    if len(q) == 0: return np.zeros((0, 61))
    s = cosine_similarity(q, refs)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(rraags == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
        
    # Standard query-wise Z-score normalization (existing control model uses this!)
    rs_norm = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)
    
    # Optional Raag-wise calibration (Phase 6)
    if calibrate:
        # Normalize each Raag column independently across all sequences
        rs_norm = zscore(rs_norm, axis=0)
        
    return rs_norm

def evaluate(keys, refs, w_t=0.75, w_p=0.15, w_n3=0.05, w_n4=0.05, is_saraga=False, calibrate=False):
    rt, rp, rn3, rn4, rr = refs
    true, sims = [], []
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        sim_t = get_sims(d['t'], rt, rr, calibrate)
        sim_p = get_sims(d['p'], rp, rr, calibrate)
        sim_n3 = get_sims(d['n3'], rn3, rr, calibrate)
        sim_n4 = get_sims(d['n4'], rn4, rr, calibrate)
        fused = w_t * sim_t + w_p * sim_p + w_n3 * sim_n3 + w_n4 * sim_n4
        
        # Top-50% aggregation
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq: final_sim = np.mean(fused, axis=0)
        else: final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        sims.append(final_sim)
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        true.append(RAAG_TO_IDX[mapped])
        
    return np.mean(np.argmax(sims, axis=1) == true)

results = []

print("Evaluating Reference Quality Filtering...")
refs_0 = build_refs(0.0)
v0 = evaluate(val_keys, refs_0, is_saraga=False)
s0 = evaluate(sar_est_keys, refs_0, is_saraga=True)
results.append({'Config': 'No Filtering', 'Val': v0, 'SARAGA': s0})

refs_5 = build_refs(0.05)
v5 = evaluate(val_keys, refs_5, is_saraga=False)
s5 = evaluate(sar_est_keys, refs_5, is_saraga=True)
results.append({'Config': 'Discard Bottom 5%', 'Val': v5, 'SARAGA': s5})

refs_10 = build_refs(0.10)
v10 = evaluate(val_keys, refs_10, is_saraga=False)
s10 = evaluate(sar_est_keys, refs_10, is_saraga=True)
results.append({'Config': 'Discard Bottom 10%', 'Val': v10, 'SARAGA': s10})

print("Evaluating Class-Specific Calibration...")
vc = evaluate(val_keys, refs_0, is_saraga=False, calibrate=True)
sc = evaluate(sar_est_keys, refs_0, is_saraga=True, calibrate=True)
results.append({'Config': 'Class Calibration', 'Val': vc, 'SARAGA': sc})

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv(os.path.join(WORKSPACE, "ref_calib_results.csv"), index=False)
