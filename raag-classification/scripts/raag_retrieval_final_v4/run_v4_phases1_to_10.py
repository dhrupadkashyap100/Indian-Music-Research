
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore
from sklearn.cluster import KMeans

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_retrieval_final_v4")
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

def compute_feats(cents_seq):
    voiced_ratio = np.sum(~np.isnan(cents_seq)) / float(len(cents_seq))
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 20: return None
    diffs = np.diff(cents)
    stability = 1.0 / (np.std(diffs) + 1.0)
    pc = np.mod(np.round(cents / 100.0), 12).astype(int)
    n3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2): n3[pc[i]*144 + pc[i+1]*12 + pc[i+2]] += 1
    n3 = n3 / (np.sum(n3) + 1e-9)
    return {'n3': n3, 'qs': voiced_ratio * stability}

def load_data(k, is_saraga=False):
    s_t = f"_EST_trans.npy" if is_saraga else f"_trans.npy"
    s_p = f"_EST_pc.npy" if is_saraga else f"_pc.npy"
    s_f = f"_EST_f0.npy" if is_saraga else f"_f0.npy"
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_t}"), os.path.join(OLD_FEAT_DIR, f"{k}{s_p}"), os.path.join(FEAT_DIR, f"{k}{s_f}")
    if not (os.path.exists(t_path) and os.path.exists(p_path) and os.path.exists(f_path)): return None
    
    t, p, f0 = np.load(t_path), np.load(p_path), np.load(f_path)
    n3_l, qs_l, valid_idx = [], [], []
    for i, seq in enumerate(f0):
        f = compute_feats(seq)
        if f is not None:
            n3_l.append(f['n3']); qs_l.append(f['qs']); valid_idx.append(i)
    min_len = min(len(t), len(p), len(f0))
    t, p = t[:min_len], p[:min_len]
    valid_idx = [i for i in valid_idx if i < min_len]
    if not valid_idx: return None
    n3_l = [n3_l[i] for i in range(len(valid_idx))]
    qs_l = [qs_l[i] for i in range(len(valid_idx))]
    t, p = t[valid_idx], p[valid_idx]
    
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return {'t': t, 'p': p, 'n3': np.array(n3_l), 'qs': np.array(qs_l).mean()}

print("Loading dataset...")
train_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'TRAIN']
val_keys = [k for k in trf_meta.index if trf_meta.loc[k, 'split'] == 'VAL']
sar_est_keys = sar_mapped['saraga_recording_id'].values

train_data = {}
for k in train_keys:
    d = load_data(k)
    if d is not None:
        train_data[k] = d

def build_refs(geom='max', prune_bottom=0.0):
    rt, rp, rn3, rr = [], [], [], []
    
    # 1. Pruning
    qs_all = np.array([train_data[k]['qs'] for k in train_data])
    qs_thresh = np.percentile(qs_all, prune_bottom * 100) if prune_bottom > 0 else -1
    
    # 2. Geometry
    for r_idx in range(61):
        r_keys = [k for k in train_data if RAAG_TO_IDX[trf_meta.loc[k, 'raag']] == r_idx and train_data[k]['qs'] > qs_thresh]
        if not r_keys: r_keys = [k for k in train_data if RAAG_TO_IDX[trf_meta.loc[k, 'raag']] == r_idx] # Fallback if pruned entirely
        
        t_arr = np.array([train_data[k]['t'].mean(axis=0) for k in r_keys])
        p_arr = np.array([train_data[k]['p'].mean(axis=0) for k in r_keys])
        n3_arr = np.array([train_data[k]['n3'].mean(axis=0) for k in r_keys])
        
        if geom == 'max':
            rt.extend(t_arr)
            rp.extend(p_arr)
            rn3.extend(n3_arr)
            rr.extend([r_idx] * len(r_keys))
        elif geom == 'mean':
            rt.append(t_arr.mean(axis=0))
            rp.append(p_arr.mean(axis=0))
            rn3.append(n3_arr.mean(axis=0))
            rr.append(r_idx)
        elif geom == 'medoid':
            mean_c = np.hstack([t_arr, p_arr, n3_arr]).mean(axis=0)
            sims = cosine_similarity(np.hstack([t_arr, p_arr, n3_arr]), [mean_c])
            idx = np.argmax(sims)
            rt.append(t_arr[idx]); rp.append(p_arr[idx]); rn3.append(n3_arr[idx]); rr.append(r_idx)
        elif geom.startswith('k='):
            k = int(geom.split('=')[1])
            if len(r_keys) < k:
                rt.extend(t_arr); rp.extend(p_arr); rn3.extend(n3_arr); rr.extend([r_idx] * len(r_keys))
            else:
                combined = np.hstack([t_arr, p_arr, n3_arr])
                kmeans = KMeans(n_clusters=k, random_state=42).fit(combined)
                for c in kmeans.cluster_centers_:
                    rt.append(c[:144])
                    rp.append(c[144:156])
                    rn3.append(c[156:])
                    rr.append(r_idx)
                    
    rt = np.array(rt) / (np.linalg.norm(rt, axis=1, keepdims=True) + 1e-9)
    rp = np.array(rp) / (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-9)
    rn3 = np.array(rn3) / (np.linalg.norm(rn3, axis=1, keepdims=True) + 1e-9)
    rr = np.array(rr)
    return rt, rp, rn3, rr

def eval_config(refs, wt=0.75, wp=0.20, wn3=0.05, agg='max', norm='zscore', consensus=1.0, seq_pen=0.0, keys=[], is_saraga=False):
    rt, rp, rn3, rr = refs
    true, sims_l = [], []
    
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        
        sim_t = cosine_similarity(d['t'], rt)
        sim_p = cosine_similarity(d['p'], rp)
        sim_n3 = cosine_similarity(d['n3'], rn3)
        raw_sim = wt * sim_t + wp * sim_p + wn3 * sim_n3
        
        rs = np.zeros((len(d['t']), 61))
        for r in range(61):
            idx = np.where(rr == r)[0]
            if len(idx) > 0:
                if agg == 'max' or len(idx) == 1:
                    rs[:, r] = np.max(raw_sim[:, idx], axis=1)
                elif agg.startswith('top-'):
                    k_agg = int(agg.split('-')[1])
                    if len(idx) <= k_agg: rs[:, r] = np.mean(raw_sim[:, idx], axis=1)
                    else:
                        sorted_sims = np.sort(raw_sim[:, idx], axis=1)
                        rs[:, r] = np.mean(sorted_sims[:, -k_agg:], axis=1)
                elif agg == 'mean':
                    rs[:, r] = np.mean(raw_sim[:, idx], axis=1)
                    
                # Hybrid consensus
                if consensus < 1.0 and len(idx) > 1:
                    med_sim = np.median(raw_sim[:, idx], axis=1)
                    rs[:, r] = consensus * rs[:, r] + (1 - consensus) * med_sim
                    
        # Normalization (applied query-wise or raag-wise)
        if norm == 'zscore':
            rs = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1) # standard query-wise
        elif norm == 'global_raag':
            rs = zscore(rs, axis=0) # Normalize across raags (highly risky)
        elif norm == 'iqr':
            q25, q75 = np.percentile(rs, [25, 75], axis=0)
            iqr = q75 - q25 + 1e-9
            rs = (rs - np.median(rs, axis=0)) / iqr
            
        # Temporal Aggregation
        num_seq = max(1, int(len(rs) * 0.5))
        if len(rs) <= num_seq: final_sim = np.mean(rs, axis=0)
        else:
            top_idx = np.argsort(np.max(rs, axis=1))[-num_seq:]
            sel = rs[top_idx]
            final_sim = np.mean(sel, axis=0)
            if seq_pen > 0:
                var = np.var(sel, axis=0)
                final_sim -= seq_pen * var
                
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        true.append(RAAG_TO_IDX[mapped])
        sims_l.append(final_sim)
        
    true = np.array(true)
    sims = np.array(sims_l)
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true)
    top3 = sum(true[i] in np.argsort(sims[i])[-3:] for i in range(len(true))) / len(true)
    f1 = f1_score(true, preds, average='macro')
    margins = [np.sort(s)[-1] - np.sort(s)[-2] for s in sims]
    
    return float(top1), float(top3), float(f1), float(np.mean(margins))

def run_experiment(name, refs, wt=0.75, wp=0.20, wn3=0.05, agg='max', norm='zscore', consensus=1.0, seq_pen=0.0):
    t1, t3, f1, m = eval_config(refs, wt, wp, wn3, agg, norm, consensus, seq_pen, val_keys, False)
    st1, st3, sf1, sm = eval_config(refs, wt, wp, wn3, agg, norm, consensus, seq_pen, sar_est_keys, True)
    return {'Candidate': name, 'Val_Top1': t1, 'Val_Top3': t3, 'Val_F1': f1, 'SARAGA_EST': st1}

leaderboard = []

print("Building references...")
refs_max = build_refs('max')
refs_max_p5 = build_refs('max', 0.05)
refs_max_p10 = build_refs('max', 0.10)
refs_max_p15 = build_refs('max', 0.15)
refs_mean = build_refs('mean')
refs_medoid = build_refs('medoid')
refs_k2 = build_refs('k=2')
refs_k3 = build_refs('k=3')

# Phase 1: Control
print("Evaluating Phase 1 (Control)...")
ctrl = run_experiment('1_Control (43.10%)', refs_max)
leaderboard.append(ctrl)
with open(os.path.join(WORKSPACE, "control_reproduction.json"), "w") as f:
    json.dump(ctrl, f, indent=4)

# Phase 2: Geometries
for name, refs in [('2_MeanPrototype', refs_mean), ('2_Medoid', refs_medoid), ('2_K=2', refs_k2), ('2_K=3', refs_k3)]:
    leaderboard.append(run_experiment(name, refs))

# Phase 3: Aggregation
for agg in ['top-2', 'top-3', 'top-5', 'mean']:
    leaderboard.append(run_experiment(f'3_RefAgg({agg})', refs_max, agg=agg))
    
# Phase 4: Pruning
leaderboard.append(run_experiment('4_PruneBottom5%', refs_max_p5))
leaderboard.append(run_experiment('4_PruneBottom10%', refs_max_p10))
leaderboard.append(run_experiment('4_PruneBottom15%', refs_max_p15))

# Phase 5: Adaptive Fusion
for wt, wp, wn3 in [(0.80, 0.15, 0.05), (0.75, 0.15, 0.10), (0.70, 0.20, 0.10), (0.70, 0.25, 0.05)]:
    leaderboard.append(run_experiment(f'5_Fusion({wt}/{wp}/{wn3})', refs_max, wt=wt, wp=wp, wn3=wn3))

# Phase 6: Normalization
leaderboard.append(run_experiment('6_Norm(GlobalRaagZ)', refs_max, norm='global_raag'))
leaderboard.append(run_experiment('6_Norm(IQR)', refs_max, norm='iqr'))

# Phase 7: Consensus
for c in [0.90, 0.75, 0.50]:
    leaderboard.append(run_experiment(f'7_Consensus(alpha={c})', refs_max, consensus=c))

# Phase 8: Sequence Consistency
for pen in [0.1, 0.5, 1.0]:
    leaderboard.append(run_experiment(f'8_SeqPenalty({pen})', refs_max, seq_pen=pen))

# Phase 9: Difficult Raag Analysis (performed using Control matrix)
print("Evaluating Phase 9 Analysis...")
true, sims = [], []
for k in val_keys:
    d = load_data(k)
    if d is None: continue
    sim_t = cosine_similarity(d['t'], refs_max[0])
    sim_p = cosine_similarity(d['p'], refs_max[1])
    sim_n3 = cosine_similarity(d['n3'], refs_max[2])
    raw = 0.75 * sim_t + 0.20 * sim_p + 0.05 * sim_n3
    rs = np.zeros((len(d['t']), 61))
    for r in range(61):
        idx = np.where(refs_max[3] == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(raw[:, idx], axis=1)
    rs = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)
    num_seq = max(1, int(len(rs) * 0.5))
    final = np.mean(rs[np.argsort(np.max(rs, axis=1))[-num_seq:]], axis=0) if len(rs) > num_seq else np.mean(rs, axis=0)
    true.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])
    sims.append(final)
true = np.array(true); sims = np.array(sims); preds = np.argmax(sims, axis=1)

raag_stats = []
for r_idx, r_name in enumerate(ALL_RAAGS):
    mask = (true == r_idx)
    if np.sum(mask) == 0: continue
    t1 = np.mean(preds[mask] == true[mask])
    # strongest confusion
    errs = preds[mask][preds[mask] != true[mask]]
    conf = ALL_RAAGS[np.bincount(errs).argmax()] if len(errs) > 0 else "None"
    raag_stats.append({'Raag': r_name, 'Val_Count': np.sum(mask), 'Top1': t1, 'Confusion': conf})
pd.DataFrame(raag_stats).to_csv(os.path.join(WORKSPACE, "per_raag_retrieval_analysis.csv"), index=False)

# Phase 10: Leaderboard
df = pd.DataFrame(leaderboard)
print(df.to_string())
df.to_csv(os.path.join(WORKSPACE, "complete_v4_leaderboard.csv"), index=False)
