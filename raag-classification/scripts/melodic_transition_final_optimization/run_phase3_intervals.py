
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
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

def load_feat(k, ftype, is_saraga=False):
    suffix = f"_EST_{ftype}.npy" if is_saraga else f"_{ftype}.npy"
    path = os.path.join(OLD_FEAT_DIR, f"{k}{suffix}")
    if not os.path.exists(path): return None
    arr = np.load(path)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

train_trans, train_pc, train_int, valid_raags = [], [], [], []
for k in train_keys:
    t, p, i = load_feat(k, 'trans'), load_feat(k, 'pc'), load_feat(k, 'int')
    if t is not None and p is not None and i is not None:
        train_trans.append(t.mean(axis=0))
        train_pc.append(p.mean(axis=0))
        train_int.append(i.mean(axis=0))
        valid_raags.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        
train_trans = np.array(train_trans)
train_pc = np.array(train_pc)
train_int = np.array(train_int)
train_raags = np.array(valid_raags)
train_trans = train_trans / (np.linalg.norm(train_trans, axis=1, keepdims=True) + 1e-9)
train_pc = train_pc / (np.linalg.norm(train_pc, axis=1, keepdims=True) + 1e-9)
train_int = train_int / (np.linalg.norm(train_int, axis=1, keepdims=True) + 1e-9)

def get_sims(q, ref, raags):
    s = cosine_similarity(q, ref)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(raags == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
    return zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)

def eval_config(keys, wt, wp, wi, is_saraga=False):
    true = []
    sims = []
    for k in keys:
        t, p, i = load_feat(k, 'trans', is_saraga), load_feat(k, 'pc', is_saraga), load_feat(k, 'int', is_saraga)
        if t is None: continue
        
        sim_t = get_sims(t, train_trans, train_raags)
        sim_p = get_sims(p, train_pc, train_raags)
        sim_i = get_sims(i, train_int, train_raags)
        
        fused = wt*sim_t + wp*sim_p + wi*sim_i
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq:
            final_sim = np.mean(fused, axis=0)
        else:
            final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        if is_saraga:
            mapped_raag = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0]
            true.append(RAAG_TO_IDX[mapped_raag])
        else:
            true.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        sims.append(final_sim)
        
    return np.mean(np.argmax(sims, axis=1) == true)

results = []
results.append({'Method': '75T + 25P', 'Val': eval_config(val_keys, 0.75, 0.25, 0), 'SAR': eval_config(sar_est_keys, 0.75, 0.25, 0, True)})
results.append({'Method': '75T + 20P + 5Int', 'Val': eval_config(val_keys, 0.75, 0.20, 0.05), 'SAR': eval_config(sar_est_keys, 0.75, 0.20, 0.05, True)})
results.append({'Method': '75T + 15P + 10Int', 'Val': eval_config(val_keys, 0.75, 0.15, 0.10), 'SAR': eval_config(sar_est_keys, 0.75, 0.15, 0.10, True)})
results.append({'Method': '70T + 20P + 10Int', 'Val': eval_config(val_keys, 0.70, 0.20, 0.10), 'SAR': eval_config(sar_est_keys, 0.70, 0.20, 0.10, True)})

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv(os.path.join(WORKSPACE, "interval_fusion_results.csv"), index=False)
