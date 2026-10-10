
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

# For the reference bank, we will still use the Early-Fusion (mean of features) as it's the stable "concept" of the training recording.
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

def compute_seq_sim(query_feat, ref_arr):
    if len(query_feat) == 0: return np.zeros((0, 61))
    sims = cosine_similarity(query_feat, ref_arr)
    raag_sims = np.zeros((len(query_feat), 61))
    for i in range(61):
        idx = np.where(train_raags == i)[0]
        if len(idx) > 0:
            raag_sims[:, i] = np.max(sims[:, idx], axis=1)
    
    noise = np.random.randn(*raag_sims.shape) * 1e-8
    return zscore(raag_sims + noise, axis=1)

def get_recording_predictions(dataset, tonic_type, split):
    records = []
    
    # Identify recordings
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
        
        # Normalize sequences
        t_seq = t_seq / (np.linalg.norm(t_seq, axis=1, keepdims=True) + 1e-9)
        p_seq = p_seq / (np.linalg.norm(p_seq, axis=1, keepdims=True) + 1e-9)
        
        # Sim per sequence
        t_sim = compute_seq_sim(t_seq, train_trans_arr)
        p_sim = compute_seq_sim(p_seq, train_pc_arr)
        
        # 75/25 Fusion at sequence level
        f_sim = 0.75 * t_sim + 0.25 * p_sim
        
        records.append({
            'id': k,
            'label': raag_idx,
            'seq_sims': f_sim,
            'early_fusion_sim': 0.75 * compute_seq_sim([t_seq.mean(axis=0)], train_trans_arr)[0] + 0.25 * compute_seq_sim([p_seq.mean(axis=0)], train_pc_arr)[0]
        })
    return records

val_recs = get_recording_predictions('TRF', None, 'VAL')
sar_recs = get_recording_predictions('SARAGA', 'EST', 'TEST')

def evaluate_agg(records, agg_fn):
    true_labels = []
    sims = []
    for r in records:
        true_labels.append(r['label'])
        sims.append(agg_fn(r['seq_sims'], r['early_fusion_sim']))
        
    sims = np.array(sims)
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true_labels)
    return top1

def eval_all(name, agg_fn):
    v = evaluate_agg(val_recs, agg_fn)
    s = evaluate_agg(sar_recs, agg_fn)
    return {'Method': name, 'TRF Val Top-1': v, 'SARAGA EST Top-1': s}

results = []
results.append(eval_all("0. Early Fusion (Baseline)", lambda sims, ef: ef))
results.append(eval_all("1. Mean of sequence similarities", lambda sims, ef: np.mean(sims, axis=0)))

def top_k_agg(sims, k):
    if len(sims) <= k: return np.mean(sims, axis=0)
    confidences = np.max(sims, axis=1)
    idx = np.argsort(confidences)[-k:]
    return np.mean(sims[idx], axis=0)

results.append(eval_all("2. Top-3 sequences", lambda sims, ef: top_k_agg(sims, 3)))
results.append(eval_all("3. Top-5 sequences", lambda sims, ef: top_k_agg(sims, 5)))
results.append(eval_all("4. Top-10 sequences", lambda sims, ef: top_k_agg(sims, 10)))

def top_p_agg(sims, p):
    k = max(1, int(len(sims) * p))
    return top_k_agg(sims, k)

results.append(eval_all("5. Top-20% sequences", lambda sims, ef: top_p_agg(sims, 0.2)))
results.append(eval_all("6. Top-30% sequences", lambda sims, ef: top_p_agg(sims, 0.3)))
results.append(eval_all("7. Top-50% sequences", lambda sims, ef: top_p_agg(sims, 0.5)))

df = pd.DataFrame(results)
df.to_csv(os.path.join(WORKSPACE, "temporal_aggregation_results.csv"), index=False)
print("Phase 4 complete.")
