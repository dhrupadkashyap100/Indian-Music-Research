
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_optimization_v2")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
MELODIC_META = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\melodic_feature_metadata.csv")

trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]
sar_map = pd.read_csv(SARAGA_MAPPING)
if 'evaluation_eligibility' in sar_map.columns:
    sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
else:
    sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]

melodic_meta = pd.read_csv(MELODIC_META)
ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}

train_trans = {}
train_pc = {}
for idx, row in melodic_meta.iterrows():
    if row['dataset'] == 'TRF' and trf_meta.loc[row['recording_id'], 'split'] == 'TRAIN':
        k = row['recording_id']
        t_path = os.path.join(FEAT_DIR, f"{k}_trans.npy")
        p_path = os.path.join(FEAT_DIR, f"{k}_pc.npy")
        if os.path.exists(t_path) and os.path.exists(p_path):
            t = np.load(t_path).mean(axis=0)
            p = np.load(p_path).mean(axis=0)
            train_trans[k] = t / (np.linalg.norm(t) + 1e-9)
            train_pc[k] = p / (np.linalg.norm(p) + 1e-9)

train_keys = list(train_trans.keys())
train_trans_arr = np.array([train_trans[k] for k in train_keys])
train_pc_arr = np.array([train_pc[k] for k in train_keys])
train_raags = np.array([RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in train_keys])

def compute_seq_sim(query_feat, ref_arr, k_ref):
    if len(query_feat) == 0: return np.zeros((0, 61))
    sims = cosine_similarity(query_feat, ref_arr)
    raag_sims = np.zeros((len(query_feat), 61))
    for i in range(61):
        idx = np.where(train_raags == i)[0]
        if len(idx) > 0:
            if k_ref == 1:
                raag_sims[:, i] = np.max(sims[:, idx], axis=1)
            else:
                s_sorted = np.sort(sims[:, idx], axis=1)
                k_actual = min(k_ref, s_sorted.shape[1])
                raag_sims[:, i] = np.mean(s_sorted[:, -k_actual:], axis=1)
                
    noise = np.random.randn(*raag_sims.shape) * 1e-8
    return zscore(raag_sims + noise, axis=1)

def get_recording_predictions(dataset, tonic_type, split, k_ref):
    records = []
    if dataset == 'TRF':
        meta = melodic_meta[(melodic_meta['dataset'] == 'TRF') & (melodic_meta['split'] == split)]
        keys = meta['recording_id'].unique()
    else:
        meta = melodic_meta[(melodic_meta['dataset'] == 'SARAGA') & (melodic_meta['tonic_type'] == tonic_type)]
        keys = meta['recording_id'].unique()
        
    for k in keys:
        if dataset == 'TRF':
            t_path = os.path.join(FEAT_DIR, f"{k}_trans.npy")
            p_path = os.path.join(FEAT_DIR, f"{k}_pc.npy")
            raag_idx = RAAG_TO_IDX[trf_meta.loc[k, 'raag']]
        else:
            t_path = os.path.join(FEAT_DIR, f"{k}_{tonic_type}_trans.npy")
            p_path = os.path.join(FEAT_DIR, f"{k}_{tonic_type}_pc.npy")
            mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0]
            raag_idx = RAAG_TO_IDX[mapped]
            
        if not os.path.exists(t_path) or not os.path.exists(p_path): continue
        t_seq = np.load(t_path)
        p_seq = np.load(p_path)
        t_seq = t_seq / (np.linalg.norm(t_seq, axis=1, keepdims=True) + 1e-9)
        p_seq = p_seq / (np.linalg.norm(p_seq, axis=1, keepdims=True) + 1e-9)
        
        t_sim = compute_seq_sim(t_seq, train_trans_arr, k_ref)
        p_sim = compute_seq_sim(p_seq, train_pc_arr, k_ref)
        
        f_sim = 0.75 * t_sim + 0.25 * p_sim
        records.append({'id': k, 'label': raag_idx, 'seq_sims': f_sim})
    return records

def top_p_agg(sims, p):
    k = max(1, int(len(sims) * p))
    if len(sims) <= k: return np.mean(sims, axis=0)
    confidences = np.max(sims, axis=1)
    idx = np.argsort(confidences)[-k:]
    return np.mean(sims[idx], axis=0)

def evaluate_agg(records):
    true_labels = []
    sims = []
    for r in records:
        true_labels.append(r['label'])
        sims.append(top_p_agg(r['seq_sims'], 0.5)) # Using Top-50% selected from previous phase
        
    sims = np.array(sims)
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true_labels)
    return top1

results = []
for k_ref in [1, 2, 3, 5]:
    val_recs = get_recording_predictions('TRF', None, 'VAL', k_ref)
    sar_recs = get_recording_predictions('SARAGA', 'EST', 'TEST', k_ref)
    
    v = evaluate_agg(val_recs)
    s = evaluate_agg(sar_recs)
    results.append({
        'Method': f"Reference Aggregation: Top-{k_ref} Mean" if k_ref > 1 else "Reference Aggregation: MAX",
        'TRF Val Top-1': v,
        'SARAGA EST Top-1': s
    })

df = pd.DataFrame(results)
df.to_csv(os.path.join(WORKSPACE, "reference_aggregation_results.csv"), index=False)
print("Phase 5 complete.")
