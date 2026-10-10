
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization")
FEAT_DIR = os.path.join(WORKSPACE, "features_f0")
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
IDX_TO_RAAG = {i: r for r, i in RAAG_TO_IDX.items()}

# ==========================================================
# 1. COMPUTE AND CACHE REPRESENTATIONS
# ==========================================================
def compute_f0_representations(cents_seq):
    # Filter unvoiced
    cents = cents_seq[~np.isnan(cents_seq)]
    if len(cents) < 10:
        return None
    
    # Quantized contour (semitones)
    semitones = np.round(cents / 100.0)
    pc = np.mod(semitones, 12).astype(int)
    
    # A. N-grams (n=3)
    n = 3
    ngram3 = np.zeros(12**3, dtype=np.float32)
    for i in range(len(pc) - 2):
        idx = pc[i]*144 + pc[i+1]*12 + pc[i+2]
        ngram3[idx] += 1
    ngram3_dist = ngram3 / (np.sum(ngram3) + 1e-9)
    
    # B & D. Local Stats & Directions
    diffs = np.diff(cents)
    asc = np.sum(diffs > 50)
    desc = np.sum(diffs < -50)
    stable = np.sum(np.abs(diffs) <= 50)
    total = len(diffs) + 1e-9
    
    large_jumps = np.sum(np.abs(diffs) > 250) / total
    mean_step = np.mean(np.abs(diffs))
    
    stats = np.array([asc/total, desc/total, stable/total, large_jumps, mean_step/1200.0], dtype=np.float32)
    
    return {'ngram3': ngram3_dist, 'stats': stats}

def load_or_compute_features():
    train_keys = []
    val_keys = []
    sar_est_keys = []
    
    data = {}
    
    # TRF
    for k in trf_meta.index:
        split = trf_meta.loc[k, 'split']
        f0_path = os.path.join(FEAT_DIR, f"{k}_f0.npy")
        if not os.path.exists(f0_path): continue
        if split == 'TEST': continue # SEALED
        
        try:
            f0_arr = np.load(f0_path)
            seqs_n3, seqs_st = [], []
            for seq in f0_arr:
                rep = compute_f0_representations(seq)
                if rep is not None:
                    seqs_n3.append(rep['ngram3'])
                    seqs_st.append(rep['stats'])
            if not seqs_n3: continue
            
            data[k] = {
                'ngram3': np.array(seqs_n3),
                'stats': np.array(seqs_st),
                'raag_idx': RAAG_TO_IDX[trf_meta.loc[k, 'raag']],
                'type': 'TRF',
                'split': split
            }
            if split == 'TRAIN': train_keys.append(k)
            elif split == 'VAL': val_keys.append(k)
        except: pass
        
    # SARAGA EST
    for k in sar_mapped['saraga_recording_id'].values:
        f0_path = os.path.join(FEAT_DIR, f"{k}_EST_f0.npy")
        if not os.path.exists(f0_path): continue
        mapped_raag = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0]
        
        try:
            f0_arr = np.load(f0_path)
            seqs_n3, seqs_st = [], []
            for seq in f0_arr:
                rep = compute_f0_representations(seq)
                if rep is not None:
                    seqs_n3.append(rep['ngram3'])
                    seqs_st.append(rep['stats'])
            if not seqs_n3: continue
            
            data[k] = {
                'ngram3': np.array(seqs_n3),
                'stats': np.array(seqs_st),
                'raag_idx': RAAG_TO_IDX[mapped_raag],
                'type': 'SARAGA',
                'split': 'TEST'
            }
            sar_est_keys.append(k)
        except: pass

    return data, train_keys, val_keys, sar_est_keys

print("Computing/Loading representations...")
data_f0, train_keys, val_keys, sar_est_keys = load_or_compute_features()
print(f"Loaded {len(train_keys)} TRAIN, {len(val_keys)} VAL, {len(sar_est_keys)} SARAGA EST")

# Build Reference Arrays (Mean across sequences for reference bank)
train_n3 = np.array([np.mean(data_f0[k]['ngram3'], axis=0) for k in train_keys])
train_st = np.array([np.mean(data_f0[k]['stats'], axis=0) for k in train_keys])
train_raags = np.array([data_f0[k]['raag_idx'] for k in train_keys])

train_n3 = train_n3 / (np.linalg.norm(train_n3, axis=1, keepdims=True) + 1e-9)
train_st = train_st / (np.linalg.norm(train_st, axis=1, keepdims=True) + 1e-9)

# Load existing Transition/PC baseline for fusion
def load_baseline(k, is_saraga=False):
    t_path = os.path.join(OLD_FEAT_DIR, f"{k}_EST_trans.npy" if is_saraga else f"{k}_trans.npy")
    p_path = os.path.join(OLD_FEAT_DIR, f"{k}_EST_pc.npy" if is_saraga else f"{k}_pc.npy")
    if not os.path.exists(t_path): return None, None
    t = np.load(t_path)
    p = np.load(p_path)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
    return t, p

train_trans, train_pc = [], []
valid_train_idx = []
for i, k in enumerate(train_keys):
    t, p = load_baseline(k)
    if t is not None:
        train_trans.append(t.mean(axis=0))
        train_pc.append(p.mean(axis=0))
        valid_train_idx.append(i)

train_trans = np.array(train_trans)
train_pc = np.array(train_pc)
train_trans = train_trans / (np.linalg.norm(train_trans, axis=1, keepdims=True) + 1e-9)
train_pc = train_pc / (np.linalg.norm(train_pc, axis=1, keepdims=True) + 1e-9)
train_raags_base = train_raags[valid_train_idx]

def get_sims(query_feat, ref_arr, raag_labels):
    if len(query_feat) == 0: return np.zeros((0, 61))
    s = cosine_similarity(query_feat, ref_arr)
    raag_s = np.zeros((len(query_feat), 61))
    for r in range(61):
        idx = np.where(raag_labels == r)[0]
        if len(idx) > 0:
            raag_s[:, r] = np.max(s[:, idx], axis=1)
    return zscore(raag_s + np.random.randn(*raag_s.shape)*1e-8, axis=1)

def eval_config(keys, w_t, w_p, w_n3, w_st, is_saraga=False):
    true_labels = []
    sims = []
    for k in keys:
        t, p = load_baseline(k, is_saraga)
        if t is None: continue
        
        q_n3 = data_f0[k]['ngram3'] / (np.linalg.norm(data_f0[k]['ngram3'], axis=1, keepdims=True) + 1e-9)
        q_st = data_f0[k]['stats'] / (np.linalg.norm(data_f0[k]['stats'], axis=1, keepdims=True) + 1e-9)
        
        # Align lengths (some sequences might have been dropped due to <10 frames)
        min_len = min(len(t), len(q_n3))
        t, p, q_n3, q_st = t[:min_len], p[:min_len], q_n3[:min_len], q_st[:min_len]
        
        sim_t = get_sims(t, train_trans, train_raags_base)
        sim_p = get_sims(p, train_pc, train_raags_base)
        sim_n3 = get_sims(q_n3, train_n3, train_raags)
        sim_st = get_sims(q_st, train_st, train_raags)
        
        fused = w_t * sim_t + w_p * sim_p + w_n3 * sim_n3 + w_st * sim_st
        
        # Top-50%
        num_seq = max(1, int(len(fused) * 0.5))
        if len(fused) <= num_seq:
            final_sim = np.mean(fused, axis=0)
        else:
            final_sim = np.mean(fused[np.argsort(np.max(fused, axis=1))[-num_seq:]], axis=0)
            
        true_labels.append(data_f0[k]['raag_idx'])
        sims.append(final_sim)
        
    sims = np.array(sims)
    preds = np.argmax(sims, axis=1)
    acc = np.mean(preds == true_labels)
    return float(acc)

print("Evaluating Baseline (75T + 25P)...")
base_val = eval_config(val_keys, 0.75, 0.25, 0, 0, False)
base_sar = eval_config(sar_est_keys, 0.75, 0.25, 0, 0, True)

results = [{'Method': 'Baseline (75T+25P)', 'Val_Top1': base_val, 'SARAGA_Top1': base_sar}]

print("Evaluating N-gram Fusions...")
for w_n3 in [0.05, 0.10, 0.15]:
    w_t = 0.75
    w_p = 1.0 - w_t - w_n3
    if w_p < 0: continue
    
    v = eval_config(val_keys, w_t, w_p, w_n3, 0, False)
    s = eval_config(sar_est_keys, w_t, w_p, w_n3, 0, True)
    results.append({'Method': f'{int(w_t*100)}T + {int(w_p*100)}P + {int(w_n3*100)}N3', 'Val_Top1': v, 'SARAGA_Top1': s})

print("Evaluating Local Stats Fusions...")
for w_st in [0.05, 0.10, 0.15]:
    w_t = 0.75
    w_p = 1.0 - w_t - w_st
    if w_p < 0: continue
    
    v = eval_config(val_keys, w_t, w_p, 0, w_st, False)
    s = eval_config(sar_est_keys, w_t, w_p, 0, w_st, True)
    results.append({'Method': f'{int(w_t*100)}T + {int(w_p*100)}P + {int(w_st*100)}Stats', 'Val_Top1': v, 'SARAGA_Top1': s})

df = pd.DataFrame(results)
df.to_csv(os.path.join(WORKSPACE, "f0_representation_results.csv"), index=False)
print(df.to_string())
print("Done Phase 2/3")
