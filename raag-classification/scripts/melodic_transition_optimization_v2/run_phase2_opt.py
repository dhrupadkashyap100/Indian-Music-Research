
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

# Load raw base matrices
raw_feats = {}
for idx, row in melodic_meta.iterrows():
    rec_id = row['recording_id']
    suffix = f"_{row['tonic_type']}" if row['dataset'] == 'SARAGA' else ""
    k = f"{rec_id}{suffix}"
    trans_path = os.path.join(FEAT_DIR, f"{rec_id}{suffix}_trans.npy")
    if os.path.exists(trans_path):
        trans = np.load(trans_path).mean(axis=0)
        raw_feats[k] = trans.reshape(12, 12)

train_keys = [k for k in trf_meta[trf_meta['split'] == 'TRAIN'].index if k in raw_feats]
val_keys = [k for k in trf_meta[trf_meta['split'] == 'VAL'].index if k in raw_feats]
sar_keys_est = [f"{r}_EST" for r in sar_mapped['saraga_recording_id'] if f"{r}_EST" in raw_feats]
sar_keys_gt = [f"{r}_GT" for r in sar_mapped['saraga_recording_id'] if f"{r}_GT" in raw_feats]

val_labels = [RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in val_keys]
sar_est_labels = [RAAG_TO_IDX[sar_mapped[sar_mapped['saraga_recording_id'] == k.replace("_EST", "")]['mapped_trf_raag'].values[0]] for k in sar_keys_est]
sar_gt_labels = [RAAG_TO_IDX[sar_mapped[sar_mapped['saraga_recording_id'] == k.replace("_GT", "")]['mapped_trf_raag'].values[0]] for k in sar_keys_gt]

def compute_similarity(queries, refs, ref_keys):
    if len(queries) == 0 or len(refs) == 0:
        return np.zeros((len(queries), 61))
    sims = cosine_similarity(queries, refs)
    raag_sims = np.zeros((len(queries), 61))
    ref_raags = np.array([RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in ref_keys])
    for i in range(61):
        idx = np.where(ref_raags == i)[0]
        if len(idx) > 0:
            raag_sims[:, i] = np.max(sims[:, idx], axis=1)
    noise = np.random.randn(*raag_sims.shape) * 1e-8
    return zscore(raag_sims + noise, axis=1)

def evaluate(sims, true_labels):
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
        'margin_sep': float(np.mean(sep_corr) - np.mean(sep_incorr)) if sep_incorr and sep_corr else 0.0
    }

def run_experiment(name, transform_fn):
    feat_dict = {}
    for k, mat in raw_feats.items():
        transformed = transform_fn(mat)
        flat = transformed.flatten()
        feat_dict[k] = flat / (np.linalg.norm(flat) + 1e-9)
        
    train_feats = np.array([feat_dict[k] for k in train_keys])
    val_sims = compute_similarity(np.array([feat_dict[k] for k in val_keys]), train_feats, train_keys)
    sar_est_sims = compute_similarity(np.array([feat_dict[k] for k in sar_keys_est]), train_feats, train_keys)
    sar_gt_sims = compute_similarity(np.array([feat_dict[k] for k in sar_keys_gt]), train_feats, train_keys)
    
    v_res = evaluate(val_sims, val_labels)
    s_res = evaluate(sar_est_sims, sar_est_labels)
    s_gt_res = evaluate(sar_gt_sims, sar_gt_labels)
    
    return {
        'Method': name,
        'TRF Val Top-1': v_res['Top-1'], 'TRF Val Top-3': v_res['Top-3'], 'TRF Val F1': v_res['Macro_F1'],
        'TRF Val Margin Sep': v_res['margin_sep'], 'TRF Val Mean Margin': v_res['mean_margin'], 'TRF Val Median Margin': v_res['median_margin'],
        'SARAGA EST Top-1': s_res['Top-1'], 'SARAGA GT Top-1': s_gt_res['Top-1']
    }

results = []

results.append(run_experiment("A. Raw baseline", lambda x: x))
results.append(run_experiment("B. Log-prob (eps=1e-6)", lambda x: np.log(x + 1e-6)))
results.append(run_experiment("B. Log-prob (eps=1e-4)", lambda x: np.log(x + 1e-4)))
results.append(run_experiment("B. Log-prob (eps=1e-3)", lambda x: np.log(x + 1e-3)))
results.append(run_experiment("C. Sqrt / Hellinger", lambda x: np.sqrt(x)))
results.append(run_experiment("D. Strict Row-normalized", lambda x: x / (np.sum(x, axis=1, keepdims=True) + 1e-9)))
results.append(run_experiment("E. Global normalized sum=1", lambda x: x / (np.sum(x) + 1e-9)))
results.append(run_experiment("F. Symmetrized (P + P.T)/2", lambda x: (x + x.T) / 2))

def concat_sym(x):
    sym = (x + x.T) / 2
    return np.concatenate([x.flatten(), sym.flatten()])
# Need a wrapper because concat_sym returns flattened, but we flatten in run_experiment. We'll adjust:
results.append(run_experiment("G. Original + Symmetrized Concat", lambda x: np.reshape(np.concatenate([x.flatten(), ((x + x.T) / 2).flatten()]), (-1, 1))))

def supp_diag(x):
    y = x.copy()
    np.fill_diagonal(y, 0)
    return y / (np.sum(y, axis=1, keepdims=True) + 1e-9)
results.append(run_experiment("H. Diagonal suppression", supp_diag))

df = pd.DataFrame(results)
df.to_csv(os.path.join(WORKSPACE, "melodic_optimization_results.csv"), index=False)
print("Phase 2 complete.")
