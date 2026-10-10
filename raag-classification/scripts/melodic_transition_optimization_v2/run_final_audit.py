
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore
from sklearn.utils import resample

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_optimization_v2")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")
MELODIC_META = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\melodic_feature_metadata.csv")

# ====================================================================
# PHASE 1: DATA-LEAKAGE AUDIT
# ====================================================================

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

train_keys = set(trf_meta[trf_meta['split'] == 'TRAIN'].index)
val_keys = set(trf_meta[trf_meta['split'] == 'VAL'].index)
test_keys = set(trf_meta[trf_meta['split'] == 'TEST'].index)
saraga_keys = set(sar_mapped['saraga_recording_id'].values)

leakage_report = {
    "TRF_Train_Count": len(train_keys),
    "TRF_Val_Count": len(val_keys),
    "TRF_Test_Count": len(test_keys),
    "Train_Val_Overlap": len(train_keys.intersection(val_keys)),
    "Train_Test_Overlap": len(train_keys.intersection(test_keys)),
    "Val_Test_Overlap": len(val_keys.intersection(test_keys)),
    "SARAGA_TRF_Overlap": len(saraga_keys.intersection(train_keys.union(val_keys).union(test_keys))),
    "PCA_or_Learned_Norm_Used": False,
    "Leakage_Status": "PASS"
}

if leakage_report["Train_Val_Overlap"] > 0 or leakage_report["Train_Test_Overlap"] > 0 or leakage_report["Val_Test_Overlap"] > 0 or leakage_report["SARAGA_TRF_Overlap"] > 0:
    leakage_report["Leakage_Status"] = "FAIL"
    
with open(os.path.join(WORKSPACE, "leakage_audit.json"), "w") as f:
    json.dump(leakage_report, f, indent=4)

# ====================================================================
# LOAD FEATURES
# ====================================================================

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

train_keys_list = list(train_trans.keys())
train_trans_arr = np.array([train_trans[k] for k in train_keys_list])
train_pc_arr = np.array([train_pc[k] for k in train_keys_list])
train_raags = np.array([RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in train_keys_list])

# Reference Coverage
ref_coverage = []
for r_idx, r_name in IDX_TO_RAAG.items():
    ref_coverage.append({
        'Raag': r_name,
        'Train_References': sum(train_raags == r_idx)
    })
pd.DataFrame(ref_coverage).to_csv(os.path.join(WORKSPACE, "reference_coverage_audit.csv"), index=False)


# ====================================================================
# PHASE 2-5: EVALUATION CORE
# ====================================================================

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

def get_predictions(dataset, tonic_type, split, model_type="optimized"):
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
        
        if model_type == "baseline":
            # Early fusion baseline (mean over sequences)
            t_mean = t_seq.mean(axis=0, keepdims=True)
            final_sim = compute_seq_sim(t_mean, train_trans_arr)[0]
        else:
            # Optimized
            p_sim = compute_seq_sim(p_seq, train_pc_arr)
            f_sim = 0.75 * t_sim + 0.25 * p_sim
            
            num_seq = max(1, int(len(f_sim) * 0.5))
            if len(f_sim) <= num_seq:
                final_sim = np.mean(f_sim, axis=0)
            else:
                conf = np.max(f_sim, axis=1)
                idx = np.argsort(conf)[-num_seq:]
                final_sim = np.mean(f_sim[idx], axis=0)
            
        records.append({'id': k, 'label': raag_idx, 'sim': final_sim})
    return records

def evaluate_recs(records):
    true_labels = [r['label'] for r in records]
    sims = np.array([r['sim'] for r in records])
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true_labels)
    top3 = sum(true_labels[i] in np.argsort(s)[-3:] for i, s in enumerate(sims)) / len(sims)
    f1 = f1_score(true_labels, preds, average='macro')
    
    margins = []
    for i, s in enumerate(sims):
        s_sorted = np.sort(s)
        margins.append(s_sorted[-1] - s_sorted[-2])
            
    return {
        'Top-1': float(top1), 'Top-3': float(top3), 'Macro_F1': float(f1),
        'mean_margin': float(np.mean(margins)),
        'true': true_labels, 'preds': preds, 'sims': sims, 'margins': margins
    }

print("Running baseline...")
base_val = evaluate_recs(get_predictions('TRF', None, 'VAL', 'baseline'))
base_test = evaluate_recs(get_predictions('TRF', None, 'TEST', 'baseline'))
base_sar_est = evaluate_recs(get_predictions('SARAGA', 'EST', 'TEST', 'baseline'))

print("Running optimized...")
opt_val = evaluate_recs(get_predictions('TRF', None, 'VAL', 'optimized'))
opt_test = evaluate_recs(get_predictions('TRF', None, 'TEST', 'optimized'))
opt_sar_est = evaluate_recs(get_predictions('SARAGA', 'EST', 'TEST', 'optimized'))
opt_sar_gt = evaluate_recs(get_predictions('SARAGA', 'GT', 'TEST', 'optimized'))


# ====================================================================
# PHASE 6: STATISTICAL CONTEXT (BOOTSTRAP)
# ====================================================================

def bootstrap_ci(true_labels, sims, metric_fn, n_bootstraps=1000):
    true_labels = np.array(true_labels)
    sims = np.array(sims)
    n = len(true_labels)
    scores = []
    for _ in range(n_bootstraps):
        idx = resample(np.arange(n))
        scores.append(metric_fn(true_labels[idx], sims[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def top1_metric(t, s): return np.mean(np.argmax(s, axis=1) == t)
def top3_metric(t, s): return sum(t[i] in np.argsort(s[i])[-3:] for i in range(len(t))) / len(t)

opt_test_top1_ci = bootstrap_ci(opt_test['true'], opt_test['sims'], top1_metric)
opt_test_top3_ci = bootstrap_ci(opt_test['true'], opt_test['sims'], top3_metric)
opt_sar_est_ci = bootstrap_ci(opt_sar_est['true'], opt_sar_est['sims'], top1_metric)

print(f"TRF Test Top-1 CI: {opt_test_top1_ci}")


# ====================================================================
# PHASE 7: PER-RAAG ANALYSIS (Validation)
# ====================================================================
raag_stats = []
sims = opt_val['sims']
true_labels = np.array(opt_val['true'])
preds = np.array(opt_val['preds'])

for r_idx in range(61):
    idx = np.where(true_labels == r_idx)[0]
    n_val = len(idx)
    if n_val == 0: continue
    
    r_sims = sims[idx]
    r_preds = preds[idx]
    
    top1 = np.mean(r_preds == r_idx)
    top3 = sum(r_idx in np.argsort(s)[-3:] for s in r_sims) / n_val
    corr = sum(r_preds == r_idx)
    
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
        'Incorrect': n_val - corr
    })

pd.DataFrame(raag_stats).to_csv(os.path.join(WORKSPACE, "per_raag_final_audit.csv"), index=False)


# ====================================================================
# PHASE 8: CONFIDENCE CALIBRATION
# ====================================================================

margins = np.array(opt_val['margins'])
corrects = (np.array(opt_val['preds']) == np.array(opt_val['true']))
sims_val = np.array(opt_val['sims'])
top3s = np.array([opt_val['true'][i] in np.argsort(s)[-3:] for i, s in enumerate(sims_val)])

bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 100.0)]
cal_results = []
for b_low, b_high in bins:
    idx = np.where((margins >= b_low) & (margins < b_high))[0]
    n_preds = len(idx)
    if n_preds == 0: continue
    acc1 = np.mean(corrects[idx])
    acc3 = np.mean(top3s[idx])
    mean_m = np.mean(margins[idx])
    cal_results.append({
        'Margin_Bin': f"{b_low}-{b_high}",
        'Count': n_preds,
        'Top-1': acc1,
        'Top-3': acc3,
        'Mean_Margin': mean_m
    })

pd.DataFrame(cal_results).to_csv(os.path.join(WORKSPACE, "confidence_calibration.csv"), index=False)

# Phase 9: Model is GO WITH CONFIDENCE FILTER based on manual analysis.

print("Final audit complete.")
