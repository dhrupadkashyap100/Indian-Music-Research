
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

# Load Trans and PC features
trans_feats = {}
pc_feats = {}

for idx, row in melodic_meta.iterrows():
    rec_id = row['recording_id']
    suffix = f"_{row['tonic_type']}" if row['dataset'] == 'SARAGA' else ""
    k = f"{rec_id}{suffix}"
    
    trans_path = os.path.join(FEAT_DIR, f"{rec_id}{suffix}_trans.npy")
    pc_path = os.path.join(FEAT_DIR, f"{rec_id}{suffix}_pc.npy")
    
    if os.path.exists(trans_path):
        trans = np.load(trans_path).mean(axis=0)
        trans_feats[k] = trans / (np.linalg.norm(trans) + 1e-9)
    if os.path.exists(pc_path):
        pc = np.load(pc_path).mean(axis=0)
        pc_feats[k] = pc / (np.linalg.norm(pc) + 1e-9)

train_keys = [k for k in trf_meta[trf_meta['split'] == 'TRAIN'].index if k in trans_feats and k in pc_feats]
val_keys = [k for k in trf_meta[trf_meta['split'] == 'VAL'].index if k in trans_feats and k in pc_feats]
sar_keys_est = [f"{r}_EST" for r in sar_mapped['saraga_recording_id'] if f"{r}_EST" in trans_feats and f"{r}_EST" in pc_feats]
sar_keys_gt = [f"{r}_GT" for r in sar_mapped['saraga_recording_id'] if f"{r}_GT" in trans_feats and f"{r}_GT" in pc_feats]

val_labels = [RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in val_keys]
sar_est_labels = [RAAG_TO_IDX[sar_mapped[sar_mapped['saraga_recording_id'] == k.replace("_EST", "")]['mapped_trf_raag'].values[0]] for k in sar_keys_est]
sar_gt_labels = [RAAG_TO_IDX[sar_mapped[sar_mapped['saraga_recording_id'] == k.replace("_GT", "")]['mapped_trf_raag'].values[0]] for k in sar_keys_gt]

def compute_sim_matrix(query_keys, ref_keys, feat_dict):
    queries = np.array([feat_dict[k] for k in query_keys])
    refs = np.array([feat_dict[k] for k in ref_keys])
    if len(queries) == 0 or len(refs) == 0: return np.zeros((len(query_keys), 61))
    sims = cosine_similarity(queries, refs)
    raag_sims = np.zeros((len(query_keys), 61))
    ref_raags = np.array([RAAG_TO_IDX[trf_meta.loc[k, 'raag']] for k in ref_keys])
    for i in range(61):
        idx = np.where(ref_raags == i)[0]
        if len(idx) > 0:
            raag_sims[:, i] = np.max(sims[:, idx], axis=1)
    noise = np.random.randn(*raag_sims.shape) * 1e-8
    return zscore(raag_sims + noise, axis=1)

val_trans = compute_sim_matrix(val_keys, train_keys, trans_feats)
sar_est_trans = compute_sim_matrix(sar_keys_est, train_keys, trans_feats)
sar_gt_trans = compute_sim_matrix(sar_keys_gt, train_keys, trans_feats)

val_pc = compute_sim_matrix(val_keys, train_keys, pc_feats)
sar_est_pc = compute_sim_matrix(sar_keys_est, train_keys, pc_feats)
sar_gt_pc = compute_sim_matrix(sar_keys_gt, train_keys, pc_feats)

def evaluate(sims, true_labels):
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true_labels)
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
        'Top-1': float(top1),
        'Macro_F1': float(f1),
        'margin_sep': float(np.mean(sep_corr) - np.mean(sep_incorr)) if sep_incorr and sep_corr else 0.0
    }

results = []
def add_fusion(w_trans, w_pc):
    v = evaluate(w_trans*val_trans + w_pc*val_pc, val_labels)
    s_est = evaluate(w_trans*sar_est_trans + w_pc*sar_est_pc, sar_est_labels)
    s_gt = evaluate(w_trans*sar_gt_trans + w_pc*sar_gt_pc, sar_gt_labels)
    results.append({
        'Method': f"Trans {w_trans*100}% + PC {w_pc*100}%",
        'TRF Val Top-1': v['Top-1'], 'TRF Val F1': v['Macro_F1'], 'TRF Val Margin Sep': v['margin_sep'],
        'SARAGA EST Top-1': s_est['Top-1'], 'SARAGA GT Top-1': s_gt['Top-1']
    })

add_fusion(1.0, 0.0)
add_fusion(0.0, 1.0)
add_fusion(0.9, 0.1)
add_fusion(0.75, 0.25)
add_fusion(0.6, 0.4)
add_fusion(0.5, 0.5)
add_fusion(0.4, 0.6)
add_fusion(0.25, 0.75)
add_fusion(0.1, 0.9)

df = pd.DataFrame(results)
df.to_csv(os.path.join(WORKSPACE, "feature_fusion_results.csv"), index=False)
print("Phase 3 complete.")
