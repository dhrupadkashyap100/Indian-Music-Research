
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.utils import resample
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
    return {'n3': n3, 'n4': n4, 'qs': voiced_ratio * stability}

def load_data(k, is_saraga=False, use_gt=False):
    s_t = f"_GT_trans.npy" if (is_saraga and use_gt) else (f"_EST_trans.npy" if is_saraga else f"_trans.npy")
    s_p = f"_GT_pc.npy" if (is_saraga and use_gt) else (f"_EST_pc.npy" if is_saraga else f"_pc.npy")
    s_f = f"_GT_f0.npy" if (is_saraga and use_gt) else (f"_EST_f0.npy" if is_saraga else f"_f0.npy")
    
    t_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_t}")
    p_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_p}")
    f_path = os.path.join(FEAT_DIR, f"{k}{s_f}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    feats = [compute_features(seq) for seq in f0]
    n3_l, n4_l, qs_l, valid_idx = [], [], [], []
    for i, f in enumerate(feats):
        if f is not None:
            n3_l.append(f['n3']); n4_l.append(f['n4']); qs_l.append(f['qs']); valid_idx.append(i)
            
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
test_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TEST']
sar_est_keys = sar_mapped['saraga_recording_id'].values

rt, rp, rn3, rn4, rr = [], [], [], [], []
for k in train_keys:
    d = load_data(k)
    if d is not None:
        valid = d['qs'] >= np.percentile(d['qs'], 5)
        if np.sum(valid) == 0: valid = np.ones(len(d['qs']), dtype=bool)
        rt.append(d['t'][valid].mean(axis=0))
        rp.append(d['p'][valid].mean(axis=0))
        rn3.append(d['n3'][valid].mean(axis=0))
        rn4.append(d['n4'][valid].mean(axis=0))
        rr.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])

rt = np.array(rt) / (np.linalg.norm(rt, axis=1, keepdims=True) + 1e-9)
rp = np.array(rp) / (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-9)
rn3 = np.array(rn3) / (np.linalg.norm(rn3, axis=1, keepdims=True) + 1e-9)
rn4 = np.array(rn4) / (np.linalg.norm(rn4, axis=1, keepdims=True) + 1e-9)
rr = np.array(rr)

def get_sims(q, refs):
    if len(q) == 0: return np.zeros((0, 61))
    s = cosine_similarity(q, refs)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(rr == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
    return zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)

def evaluate_recs(keys, is_saraga=False, use_gt=False):
    recs = []
    for k in keys:
        d = load_data(k, is_saraga, use_gt)
        if d is None: continue
        
        sim_t = get_sims(d['t'], rt)
        sim_p = get_sims(d['p'], rp)
        sim_n3 = get_sims(d['n3'], rn3)
        sim_n4 = get_sims(d['n4'], rn4)
        
        fused = 0.75 * sim_t + 0.15 * sim_p + 0.05 * sim_n3 + 0.05 * sim_n4
        
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq: final_sim = np.mean(fused, axis=0)
        else: final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        recs.append({'id': k, 'true': RAAG_TO_IDX[mapped], 'sim': final_sim})
    return recs

print("Evaluating Final SEALED Test and SARAGA...")
test_recs = evaluate_recs(test_keys)
sar_est_recs = evaluate_recs(sar_est_keys, True, False)
sar_gt_recs = evaluate_recs(sar_est_keys, True, True)

def calc_metrics(recs):
    if not recs: return {}
    true = np.array([r['true'] for r in recs])
    sims = np.array([r['sim'] for r in recs])
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true)
    top3 = sum(true[i] in np.argsort(sims[i])[-3:] for i in range(len(true))) / len(true)
    f1 = f1_score(true, preds, average='macro')
    margins = [np.sort(s)[-1] - np.sort(s)[-2] for s in sims]
    return {'Top-1': float(top1), 'Top-3': float(top3), 'Macro_F1': float(f1), 'Mean_Margin': float(np.mean(margins))}

def boot(recs):
    if not recs: return [0, 0]
    true = np.array([r['true'] for r in recs])
    sims = np.array([r['sim'] for r in recs])
    n = len(true)
    s = []
    for _ in range(1000):
        idx = resample(np.arange(n))
        s.append(np.mean(np.argmax(sims[idx], axis=1) == true[idx]))
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]

res = {
    'Method': '75T + 15PC + 5N3 + 5N4, Top-50%, MAX Ref(Discard Bottom 5%)',
    'TRF_Test': calc_metrics(test_recs),
    'SARAGA_EST': calc_metrics(sar_est_recs),
    'SARAGA_GT': calc_metrics(sar_gt_recs),
    'TRF_Test_CI': boot(test_recs)
}
with open(os.path.join(WORKSPACE, "sealed_test_results.json"), "w") as f:
    json.dump(res, f, indent=4)
print("Sealed test done.")
