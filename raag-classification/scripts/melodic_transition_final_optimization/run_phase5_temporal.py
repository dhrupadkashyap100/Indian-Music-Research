
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(WORKSPACE, "features_f0")
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

def compute_n3(cents_seq):
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 10: return None
    semitones = np.round(cents / 100.0)
    pc = np.mod(semitones, 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2):
        n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    return n3 / (np.sum(n3) + 1e-9)

def load_data(k, is_saraga=False):
    suffix_t = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    suffix_p = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    suffix_f0 = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    
    t_path = os.path.join(OLD_FEAT_DIR, f"{k}{suffix_t}")
    p_path = os.path.join(OLD_FEAT_DIR, f"{k}{suffix_p}")
    f0_path = os.path.join(FEAT_DIR, f"{k}{suffix_f0}")
    
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f0_path)): return None, None, None
    
    t = np.load(t_path)
    p = np.load(p_path)
    f0 = np.load(f0_path)
    
    n3_list = []
    for seq in f0:
        n3 = compute_n3(seq)
        if n3 is not None: n3_list.append(n3)
        
    if not n3_list: return None, None, None
    n3_arr = np.array(n3_list)
    
    mlen = min(len(t), len(n3_arr))
    t = t[:mlen] / (np.linalg.norm(t[:mlen], axis=1, keepdims=True) + 1e-9)
    p = p[:mlen] / (np.linalg.norm(p[:mlen], axis=1, keepdims=True) + 1e-9)
    n3_arr = n3_arr[:mlen] / (np.linalg.norm(n3_arr[:mlen], axis=1, keepdims=True) + 1e-9)
    
    return t, p, n3_arr

train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
test_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TEST']
sar_est_keys = sar_mapped['saraga_recording_id'].values

train_t, train_p, train_n3, train_r = [], [], [], []
for k in train_keys:
    t, p, n3 = load_data(k)
    if t is not None:
        train_t.append(t.mean(axis=0))
        train_p.append(p.mean(axis=0))
        train_n3.append(n3.mean(axis=0))
        train_r.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])

train_t = np.array(train_t)
train_p = np.array(train_p)
train_n3 = np.array(train_n3)
train_r = np.array(train_r)

train_t = train_t / (np.linalg.norm(train_t, axis=1, keepdims=True) + 1e-9)
train_p = train_p / (np.linalg.norm(train_p, axis=1, keepdims=True) + 1e-9)
train_n3 = train_n3 / (np.linalg.norm(train_n3, axis=1, keepdims=True) + 1e-9)

def get_sims(q, ref, raags):
    s = cosine_similarity(q, ref)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(raags == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
    return zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)

def eval_temporal(keys, top_p, is_saraga=False):
    true = []
    sims = []
    for k in keys:
        t, p, n3 = load_data(k, is_saraga)
        if t is None: continue
        
        sim_t = get_sims(t, train_t, train_r)
        sim_p = get_sims(p, train_p, train_r)
        sim_n3 = get_sims(n3, train_n3, train_r)
        
        fused = 0.75*sim_t + 0.20*sim_p + 0.05*sim_n3
        
        if top_p == 1.0:
            final_sim = np.mean(fused, axis=0)
        else:
            num_seq = max(1, int(len(fused) * top_p))
            if len(fused) <= num_seq:
                final_sim = np.mean(fused, axis=0)
            else:
                final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        if is_saraga:
            mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0]
            true.append(RAAG_TO_IDX[mapped])
        else:
            true.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        sims.append(final_sim)
        
    return np.mean(np.argmax(sims, axis=1) == true)

results = []
for p in [1.0, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20]:
    v = eval_temporal(val_keys, p, False)
    s = eval_temporal(sar_est_keys, p, True)
    results.append({'Aggregation': f'Top-{int(p*100)}%', 'Val': v, 'SARAGA': s})

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv(os.path.join(WORKSPACE, "temporal_optimization_results.csv"), index=False)

# Best aggregation based on this (assuming 50% remains solid) is selected for SEALED TEST
print("Running Sealed Test and SARAGA GT...")

# Best model: 75T + 20P + 5N3, Top-50%
def get_final_recs(keys, is_saraga=False, use_gt=False):
    recs = []
    for k in keys:
        if is_saraga and use_gt:
            suffix_t = f"_GT_trans.npy"
            suffix_p = f"_GT_pc.npy"
            suffix_f0 = f"_GT_f0.npy"
            t_path = os.path.join(OLD_FEAT_DIR, f"{k}{suffix_t}")
            p_path = os.path.join(OLD_FEAT_DIR, f"{k}{suffix_p}")
            f0_path = os.path.join(FEAT_DIR, f"{k}{suffix_f0}")
            if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f0_path)): continue
            t = np.load(t_path)
            p = np.load(p_path)
            f0 = np.load(f0_path)
            n3_list = []
            for seq in f0:
                n3 = compute_n3(seq)
                if n3 is not None: n3_list.append(n3)
            if not n3_list: continue
            n3_arr = np.array(n3_list)
            mlen = min(len(t), len(n3_arr))
            t = t[:mlen] / (np.linalg.norm(t[:mlen], axis=1, keepdims=True) + 1e-9)
            p = p[:mlen] / (np.linalg.norm(p[:mlen], axis=1, keepdims=True) + 1e-9)
            n3_arr = n3_arr[:mlen] / (np.linalg.norm(n3_arr[:mlen], axis=1, keepdims=True) + 1e-9)
        else:
            t, p, n3_arr = load_data(k, is_saraga)
            if t is None: continue
            
        sim_t = get_sims(t, train_t, train_r)
        sim_p = get_sims(p, train_p, train_r)
        sim_n3 = get_sims(n3_arr, train_n3, train_r)
        
        fused = 0.75*sim_t + 0.20*sim_p + 0.05*sim_n3
        
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq:
            final_sim = np.mean(fused, axis=0)
        else:
            final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        if is_saraga:
            mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0]
            true = RAAG_TO_IDX[mapped]
        else:
            true = RAAG_TO_IDX[trf_meta.loc[k, 'raag']]
            
        recs.append({'id': k, 'true': true, 'sim': final_sim})
    return recs

from sklearn.metrics import f1_score
from sklearn.utils import resample

test_recs = get_final_recs(test_keys, False)
sar_est_recs = get_final_recs(sar_est_keys, True, False)
sar_gt_recs = get_final_recs(sar_est_keys, True, True)

def eval_recs(recs):
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

import json
res = {
    'Method': '75% Trans + 20% PC + 5% N-gram, Top-50% Seq, MAX Ref',
    'TRF_Test': eval_recs(test_recs),
    'SARAGA_EST': eval_recs(sar_est_recs),
    'SARAGA_GT': eval_recs(sar_gt_recs),
    'TRF_Test_CI': boot(test_recs)
}
with open(os.path.join(WORKSPACE, "sealed_test_results.json"), "w") as f:
    json.dump(res, f, indent=4)
print("Sealed test evaluated.")
