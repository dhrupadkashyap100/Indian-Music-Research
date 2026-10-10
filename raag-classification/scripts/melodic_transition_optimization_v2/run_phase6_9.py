
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, confusion_matrix
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
IDX_TO_RAAG = {i: r for r, i in RAAG_TO_IDX.items()}

# Load train references
train_trans, train_pc = {}, {}
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
            raag_sims[:, i] = np.max(sims[:, idx], axis=1) # MAX reference aggregation
    noise = np.random.randn(*raag_sims.shape) * 1e-8
    return zscore(raag_sims + noise, axis=1)

def get_predictions(dataset, tonic_type, split):
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
        
        t_sim = compute_seq_sim(t_seq, train_trans_arr)
        p_sim = compute_seq_sim(p_seq, train_pc_arr)
        
        f_sim = 0.75 * t_sim + 0.25 * p_sim
        
        # Top-50% sequence aggregation
        num_seq = max(1, int(len(f_sim) * 0.5))
        if len(f_sim) <= num_seq:
            final_sim = np.mean(f_sim, axis=0)
        else:
            conf = np.max(f_sim, axis=1)
            idx = np.argsort(conf)[-num_seq:]
            final_sim = np.mean(f_sim[idx], axis=0)
            
        records.append({'id': k, 'label': raag_idx, 'sim': final_sim})
    return records

print("Running Val, Test, SARAGA EST, SARAGA GT...")
val_recs = get_predictions('TRF', None, 'VAL')
test_recs = get_predictions('TRF', None, 'TEST')
sar_est_recs = get_predictions('SARAGA', 'EST', 'TEST')
sar_gt_recs = get_predictions('SARAGA', 'GT', 'TEST')

def evaluate_recs(records):
    true_labels = [r['label'] for r in records]
    sims = np.array([r['sim'] for r in records])
    preds = np.argmax(sims, axis=1)
    
    top1 = np.mean(preds == true_labels)
    top3 = sum(true_labels[i] in np.argsort(s)[-3:] for i, s in enumerate(sims)) / len(sims)
    f1 = f1_score(true_labels, preds, average='macro')
    
    margins, sep_corr, sep_incorr = [], [], []
    for i, s in enumerate(sims):
        s_sorted = np.sort(s)
        margin = s_sorted[-1] - s_sorted[-2]
        margins.append(margin)
        if preds[i] == true_labels[i]:
            sep_corr.append(margin)
        else:
            sep_incorr.append(s_sorted[-1] - s[true_labels[i]])
            
    return {
        'Top-1': float(top1), 'Top-3': float(top3), 'Macro_F1': float(f1),
        'mean_margin': float(np.mean(margins)), 'median_margin': float(np.median(margins)),
        'margin_sep': float(np.mean(sep_corr) - np.mean(sep_incorr)) if sep_incorr and sep_corr else 0.0,
        'true': true_labels, 'preds': preds, 'sims': sims
    }

val_res = evaluate_recs(val_recs)
test_res = evaluate_recs(test_recs)
sar_est_res = evaluate_recs(sar_est_recs)
sar_gt_res = evaluate_recs(sar_gt_recs)

# Export sealed test results
sealed_json = {
    'Method': 'Optimized Melodic (75% Trans + 25% PC, Top-50% Seq, MAX Ref)',
    'TRF_Validation': {k: v for k, v in val_res.items() if k not in ['true', 'preds', 'sims']},
    'TRF_Test': {k: v for k, v in test_res.items() if k not in ['true', 'preds', 'sims']},
    'SARAGA_EST': {k: v for k, v in sar_est_res.items() if k not in ['true', 'preds', 'sims']},
    'SARAGA_GT': {k: v for k, v in sar_gt_res.items() if k not in ['true', 'preds', 'sims']}
}
with open(os.path.join(WORKSPACE, "sealed_test_results.json"), "w") as f:
    json.dump(sealed_json, f, indent=4)

# Phase 7: Per-Raag Analysis on Validation
raag_stats = []
sims = val_res['sims']
true_labels = np.array(val_res['true'])
preds = np.array(val_res['preds'])

for r_idx in range(61):
    idx = np.where(true_labels == r_idx)[0]
    n_val = len(idx)
    if n_val == 0: continue
    
    r_sims = sims[idx]
    r_preds = preds[idx]
    
    top1 = np.mean(r_preds == r_idx)
    top3 = sum(r_idx in np.argsort(s)[-3:] for s in r_sims) / n_val
    corr = sum(r_preds == r_idx)
    incorr = n_val - corr
    
    margins = [np.sort(s)[-1] - np.sort(s)[-2] for s in r_sims]
    
    train_count = sum(train_raags == r_idx)
    
    raag_stats.append({
        'Raag': IDX_TO_RAAG[r_idx],
        'Train_References': train_count,
        'Val_Recordings': n_val,
        'Top-1': top1,
        'Top-3': top3,
        'Mean_Margin': np.mean(margins),
        'Correct': corr,
        'Incorrect': incorr
    })

pd.DataFrame(raag_stats).to_csv(os.path.join(WORKSPACE, "per_raag_performance.csv"), index=False)

# Confusion Matrix
cm = confusion_matrix(true_labels, preds, labels=range(61))
cm_df = pd.DataFrame(cm, index=ALL_RAAGS, columns=ALL_RAAGS)
cm_df.to_csv(os.path.join(WORKSPACE, "confusion_matrix.csv"))

print("Phase 6-9 complete. Model selected and tested.")
