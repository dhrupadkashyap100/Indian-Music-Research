
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_metric_learning")
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

def compute_n3(cents_seq):
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    return n3 / (np.sum(n3) + 1e-9)

def load_data(k, is_saraga=False):
    s_t = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    s_p = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    s_f = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    
    t_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_t}")
    p_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_p}")
    f_path = os.path.join(FEAT_DIR, f"{k}{s_f}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    n3_l, valid_idx = [], []
    for i, seq in enumerate(f0):
        n3 = compute_n3(seq)
        if n3 is not None:
            n3_l.append(n3); valid_idx.append(i)
            
    min_len = min(len(t), len(p), len(f0))
    t, p = t[:min_len], p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    if not valid_idx: return None
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    t, p = t[valid_idx], p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return {'t': t, 'p': p, 'n3': np.array(n3_l)}

print("Loading dataset...")
train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

rt_list, rp_list, rn3_list, rr_list = [], [], [], []
train_seqs = []

# Feature inventory variables
all_t, all_p, all_n3 = [], [], []

for k in train_keys:
    d = load_data(k)
    if d is not None:
        rt_list.append(d['t'].mean(axis=0))
        rp_list.append(d['p'].mean(axis=0))
        rn3_list.append(d['n3'].mean(axis=0))
        rr_list.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
        
        all_t.append(d['t'])
        all_p.append(d['p'])
        all_n3.append(d['n3'])
        
        for i in range(len(d['t'])):
            train_seqs.append((d['t'][i], d['p'][i], d['n3'][i], RAAG_TO_IDX[trf_meta.loc[k, 'raag']]))

rt = np.array(rt_list) / (np.linalg.norm(rt_list, axis=1, keepdims=True) + 1e-9)
rp = np.array(rp_list) / (np.linalg.norm(rp_list, axis=1, keepdims=True) + 1e-9)
rn3 = np.array(rn3_list) / (np.linalg.norm(rn3_list, axis=1, keepdims=True) + 1e-9)
rr = np.array(rr_list)

# Generate Phase 1 Feature Inventory
flat_t = np.vstack(all_t)
flat_p = np.vstack(all_p)
flat_n3 = np.vstack(all_n3)
inventory = pd.DataFrame({
    'Feature': ['Transition Matrix', 'Pitch-Class', 'N=3 Trigram', 'Concatenated'],
    'Dimensions': [flat_t.shape[1], flat_p.shape[1], flat_n3.shape[1], flat_t.shape[1] + flat_p.shape[1] + flat_n3.shape[1]],
    'Num_Sequences_Train': [len(flat_t)] * 4,
    'Sparsity_%': [
        np.mean(flat_t == 0) * 100,
        np.mean(flat_p == 0) * 100,
        np.mean(flat_n3 == 0) * 100,
        np.mean(np.hstack([flat_t, flat_p, flat_n3]) == 0) * 100
    ],
    'Has_NaNs': [np.isnan(flat_t).any(), np.isnan(flat_p).any(), np.isnan(flat_n3).any(), False],
    'Has_Infs': [np.isinf(flat_t).any(), np.isinf(flat_p).any(), np.isinf(flat_n3).any(), False]
})
inventory.to_csv(os.path.join(WORKSPACE, "feature_inventory.csv"), index=False)
print(inventory)

# Build Sequence-level training data for PCA/LDA
X_seq_t = np.array([x[0] for x in train_seqs])
X_seq_p = np.array([x[1] for x in train_seqs])
X_seq_n3 = np.array([x[2] for x in train_seqs])
y_seq = np.array([x[3] for x in train_seqs])
X_seq_concat = np.hstack([X_seq_t, X_seq_p, X_seq_n3])
X_seq_concat = X_seq_concat / (np.linalg.norm(X_seq_concat, axis=1, keepdims=True) + 1e-9)

def get_sims(q, refs):
    if len(q) == 0: return np.zeros((0, 61))
    s = cosine_similarity(q, refs)
    rs = np.zeros((len(q), 61))
    for r in range(61):
        idx = np.where(rr == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
    return zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)

def evaluate_baseline(method, model=None, keys=[], is_saraga=False):
    true, sims = [], []
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        
        if method == 'Control':
            sim_t = get_sims(d['t'], rt)
            sim_p = get_sims(d['p'], rp)
            sim_n3 = get_sims(d['n3'], rn3)
            fused = 0.75 * sim_t + 0.20 * sim_p + 0.05 * sim_n3
        else:
            q_concat = np.hstack([d['t'], d['p'], d['n3']])
            q_concat = q_concat / (np.linalg.norm(q_concat, axis=1, keepdims=True) + 1e-9)
            
            if method == 'Raw_Concat':
                r_concat = np.hstack([rt, rp, rn3])
                r_concat = r_concat / (np.linalg.norm(r_concat, axis=1, keepdims=True) + 1e-9)
                fused = get_sims(q_concat, r_concat)
            elif method == 'PCA' or method == 'LDA':
                q_proj = model.transform(q_concat)
                q_proj = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-9)
                
                # Transform train references
                r_concat = np.hstack([rt, rp, rn3])
                r_concat = r_concat / (np.linalg.norm(r_concat, axis=1, keepdims=True) + 1e-9)
                r_proj = model.transform(r_concat)
                r_proj = r_proj / (np.linalg.norm(r_proj, axis=1, keepdims=True) + 1e-9)
                
                fused = get_sims(q_proj, r_proj)
                
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq: final_sim = np.mean(fused, axis=0)
        else: final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        true.append(RAAG_TO_IDX[mapped])
        sims.append(final_sim)
        
    true = np.array(true)
    sims = np.array(sims)
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true)
    top3 = sum(true[i] in np.argsort(sims[i])[-3:] for i in range(len(true))) / len(true)
    f1 = f1_score(true, preds, average='macro')
    margins = [np.sort(s)[-1] - np.sort(s)[-2] for s in sims]
    
    return float(top1), float(top3), float(f1), float(np.mean(margins))

results = []
print("\nEvaluating Control...")
t1, t3, f1, m = evaluate_baseline('Control', keys=val_keys)
st1, st3, sf1, sm = evaluate_baseline('Control', keys=sar_est_keys, is_saraga=True)
results.append({'Method': 'Control (43.10%)', 'Val_Top1': t1, 'Val_Top3': t3, 'Val_F1': f1, 'SARAGA_EST': st1})

print("Evaluating Raw Concatenation...")
t1, t3, f1, m = evaluate_baseline('Raw_Concat', keys=val_keys)
st1, st3, sf1, sm = evaluate_baseline('Raw_Concat', keys=sar_est_keys, is_saraga=True)
results.append({'Method': 'Raw Concatenation', 'Val_Top1': t1, 'Val_Top3': t3, 'Val_F1': f1, 'SARAGA_EST': st1})

for dim in [64, 128, 256]:
    print(f"Evaluating PCA {dim}...")
    pca = PCA(n_components=dim)
    pca.fit(X_seq_concat) # FIT ONLY ON TRAIN
    t1, t3, f1, m = evaluate_baseline('PCA', model=pca, keys=val_keys)
    st1, st3, sf1, sm = evaluate_baseline('PCA', model=pca, keys=sar_est_keys, is_saraga=True)
    results.append({'Method': f'PCA-{dim}', 'Val_Top1': t1, 'Val_Top3': t3, 'Val_F1': f1, 'SARAGA_EST': st1})

print("Evaluating LDA (Supervised Linear Projection)...")
lda = LDA()
lda.fit(X_seq_concat, y_seq) # FIT ONLY ON TRAIN
t1, t3, f1, m = evaluate_baseline('LDA', model=lda, keys=val_keys)
st1, st3, sf1, sm = evaluate_baseline('LDA', model=lda, keys=sar_est_keys, is_saraga=True)
results.append({'Method': 'LDA (Supervised)', 'Val_Top1': t1, 'Val_Top3': t3, 'Val_F1': f1, 'SARAGA_EST': st1})

res_df = pd.DataFrame(results)
print(res_df.to_string())
res_df.to_csv(os.path.join(WORKSPACE, "baseline_comparison.csv"), index=False)
