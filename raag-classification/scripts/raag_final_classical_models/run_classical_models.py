
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import warnings

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_final_classical_models")
OLD_FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\cross_dataset_melodic_optimization\features")
FEAT_DIR = os.path.join(PROJECT_ROOT, r"scripts\melodic_transition_final_optimization\features_f0")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

# 1. Load Data
trf = pd.read_csv(TRF_MANIFEST)
trf_meta = trf.drop_duplicates('source_recording').set_index('source_recording')[['split', 'raag']]
sar_map = pd.read_csv(SARAGA_MAPPING)
if 'evaluation_eligibility' in sar_map.columns:
    sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True]
else:
    sar_mapped = sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]

ALL_RAAGS = sorted(trf_meta['raag'].unique().tolist())
RAAG_TO_IDX = {r: i for i, r in enumerate(ALL_RAAGS)}
NUM_CLASSES = len(ALL_RAAGS)

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
    t_path, p_path, f_path = os.path.join(OLD_FEAT_DIR, f"{k}{s_t}"), os.path.join(OLD_FEAT_DIR, f"{k}{s_p}"), os.path.join(FEAT_DIR, f"{k}{s_f}")
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

# Precompute Reference Bank for Control
rt_l, rp_l, rn3_l, rr_l = [], [], [], []
train_data = {}
for k in train_keys:
    d = load_data(k)
    if d is not None:
        train_data[k] = d
        rt_l.append(d['t'].mean(axis=0))
        rp_l.append(d['p'].mean(axis=0))
        rn3_l.append(d['n3'].mean(axis=0))
        rr_l.append(RAAG_TO_IDX[trf_meta.loc[k, 'raag']])

rt = np.array(rt_l) / (np.linalg.norm(rt_l, axis=1, keepdims=True) + 1e-9)
rp = np.array(rp_l) / (np.linalg.norm(rp_l, axis=1, keepdims=True) + 1e-9)
rn3 = np.array(rn3_l) / (np.linalg.norm(rn3_l, axis=1, keepdims=True) + 1e-9)
rr = np.array(rr_l)

def get_control_probs(d):
    sim_t = cosine_similarity(d['t'], rt)
    sim_p = cosine_similarity(d['p'], rp)
    sim_n3 = cosine_similarity(d['n3'], rn3)
    raw = 0.75 * sim_t + 0.20 * sim_p + 0.05 * sim_n3
    rs = np.zeros((len(d['t']), 61))
    for r in range(61):
        idx = np.where(rr == r)[0]
        if len(idx) > 0: rs[:, r] = np.max(raw[:, idx], axis=1)
    rs = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)
    num_seq = max(1, int(len(rs) * 0.5))
    final = np.mean(rs[np.argsort(np.max(rs, axis=1))[-num_seq:]], axis=0) if len(rs) > num_seq else np.mean(rs, axis=0)
    # Convert z-score to pseudo-probability via softmax
    e_x = np.exp(final - np.max(final))
    return e_x / e_x.sum()

def get_scaled_features(d, use_t=True, use_p=True, use_n3=True):
    parts = []
    if use_t: parts.append(0.75 * d['t'])
    if use_p: parts.append(0.20 * d['p'])
    if use_n3: parts.append(0.05 * d['n3'])
    return np.hstack(parts)

# Construct Training Dataset for Classifiers
print("Building Sequence-Level Training Data...")
X_train_seqs, y_train_seqs = [], []
for k in train_keys:
    if k in train_data:
        d = train_data[k]
        feat = get_scaled_features(d)
        X_train_seqs.extend(feat)
        y_train_seqs.extend([RAAG_TO_IDX[trf_meta.loc[k, 'raag']]] * len(feat))
X_train_seqs = np.array(X_train_seqs)
y_train_seqs = np.array(y_train_seqs)

def eval_model(model_pred_fn, keys, is_saraga=False):
    true, preds, confs, top3s, all_probs = [], [], [], [], []
    for k in keys:
        d = load_data(k, is_saraga)
        if d is None: continue
        probs = model_pred_fn(d)
        mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
        true.append(RAAG_TO_IDX[mapped])
        preds.append(np.argmax(probs))
        confs.append(np.max(probs))
        top3s.append(np.argsort(probs)[-3:].tolist())
        all_probs.append(probs)
    
    true = np.array(true)
    preds = np.array(preds)
    top1 = float(np.mean(preds == true))
    top3 = float(sum(true[i] in top3s[i] for i in range(len(true))) / len(true))
    f1 = float(f1_score(true, preds, average='macro'))
    margins = float(np.mean([np.sort(p)[-1] - np.sort(p)[-2] for p in all_probs]))
    return top1, top3, f1, margins, true, preds, top3s, confs, all_probs

# Phase 1: Control Reproduction
print("Evaluating Phase 1 (Control)...")
vt1, vt3, vf1, vm, v_true, v_cpreds, v_ctop3, v_cconf, v_cprob = eval_model(get_control_probs, val_keys, False)
st1, st3, sf1, sm, s_true, s_cpreds, s_ctop3, s_cconf, s_cprob = eval_model(get_control_probs, sar_est_keys, True)
tt1, tt3, tf1, tm, _, _, _, _, _ = eval_model(get_control_probs, train_keys, False)
control_results = {'Model': 'Control (43.10%)', 'Train_Acc': tt1, 'Val_Top1': vt1, 'Val_Top3': vt3, 'Val_F1': vf1, 'SARAGA_EST': st1, 'Margin': vm}
with open(os.path.join(WORKSPACE, "control_reproduction.json"), "w") as f:
    json.dump(control_results, f, indent=4)

leaderboard = [control_results]

# Store models to reuse the best one later
trained_models = {}

def get_classifier_probs(model, use_t, use_p, use_n3):
    def fn(d):
        X = get_scaled_features(d, use_t, use_p, use_n3)
        seq_probs = model.predict_proba(X)
        # Average probability across sequences
        final_probs = np.mean(seq_probs, axis=0)
        return final_probs
    return fn

def train_and_eval(name, clf, use_t=True, use_p=True, use_n3=True):
    print(f"Training {name}...")
    
    # Ablation support
    X_tr = get_scaled_features({'t': X_train_seqs[:, :144]*1.333, 'p': X_train_seqs[:, 144:156]*5, 'n3': X_train_seqs[:, 156:]*20}, use_t, use_p, use_n3)
    
    clf.fit(X_tr, y_train_seqs)
    pred_fn = get_classifier_probs(clf, use_t, use_p, use_n3)
    
    tt1, tt3, tf1, tm, _, _, _, _, _ = eval_model(pred_fn, train_keys, False)
    vt1, vt3, vf1, vm, v_true, v_preds, v_top3, v_conf, v_prob = eval_model(pred_fn, val_keys, False)
    st1, st3, sf1, sm, s_true, s_preds, s_top3, s_conf, s_prob = eval_model(pred_fn, sar_est_keys, True)
    
    # Agreement metrics on Validation
    both_correct = np.sum((v_preds == v_true) & (v_cpreds == v_true))
    control_corr_clf_wrong = np.sum((v_cpreds == v_true) & (v_preds != v_true))
    control_wrong_clf_corr = np.sum((v_cpreds != v_true) & (v_preds == v_true))
    agreement = np.mean(v_preds == v_cpreds)
    
    gap = tt1 - vt1
    status = "OVERFIT" if gap > 0.40 and vt1 < control_results['Val_Top1'] else "PROMISING" if vt1 > control_results['Val_Top1'] else "NO_IMPROVEMENT"
    
    res = {
        'Model': name, 'Train_Acc': tt1, 'Val_Top1': vt1, 'Val_Top3': vt3, 'Val_F1': vf1, 
        'SARAGA_EST': st1, 'Margin': vm, 'Gap': gap, 
        'Agreement': agreement, 'Ctrl_Corr_Clf_Wrong': control_corr_clf_wrong, 
        'Ctrl_Wrong_Clf_Corr': control_wrong_clf_corr, 'Status': status,
        'v_preds': v_preds, 'v_prob': v_prob, 'v_conf': v_conf,
        's_preds': s_preds, 's_prob': s_prob, 's_conf': s_conf
    }
    leaderboard.append({k: v for k, v in res.items() if not isinstance(v, (np.ndarray, list))})
    trained_models[name] = res
    return res

# Phase 3: SVM
for c in [0.1, 1, 10]:
    for g in ['scale', 0.001]:
        train_and_eval(f'SVM_C={c}_g={g}', SVC(C=c, gamma=g, probability=True, random_state=42))

# Phase 4: ExtraTrees
for n in [200, 500]:
    for d in [None, 10, 20]:
        train_and_eval(f'ET_{n}_d={d}', ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=42))

# Phase 5: LightGBM
if LGBM_AVAILABLE:
    for lr in [0.05, 0.1]:
        for nl in [15, 31]:
            train_and_eval(f'LGBM_lr={lr}_nl={nl}', lgb.LGBMClassifier(learning_rate=lr, num_leaves=nl, random_state=42, n_estimators=100))

# Find best model
best_name = None
best_val = -1
for name in trained_models:
    if trained_models[name]['Val_Top1'] > best_val:
        best_val = trained_models[name]['Val_Top1']
        best_name = name

print(f"\nBest Classifier: {best_name} (Val: {best_val:.3f})")

# Phase 6: Feature Ablation on Best Classifier Model Type
if best_name.startswith('SVM'): base_clf = SVC(C=float(best_name.split('_C=')[1].split('_')[0]), gamma=best_name.split('_g=')[1], probability=True, random_state=42)
elif best_name.startswith('ET'): base_clf = ExtraTreesClassifier(n_estimators=int(best_name.split('ET_')[1].split('_')[0]), max_depth=None if 'None' in best_name else int(best_name.split('_d=')[1]), random_state=42)
elif best_name.startswith('LGBM'): base_clf = lgb.LGBMClassifier(learning_rate=float(best_name.split('_lr=')[1].split('_')[0]), num_leaves=int(best_name.split('_nl=')[1]), random_state=42, n_estimators=100)

train_and_eval(f'{best_name}_Abl_T_Only', base_clf, True, False, False)
train_and_eval(f'{best_name}_Abl_T_PC', base_clf, True, True, False)
train_and_eval(f'{best_name}_Abl_T_N3', base_clf, True, False, True)

# Phase 7: Hybrid Retrieval
best = trained_models[best_name]
def eval_hybrid(name, v_hpreds, s_hpreds):
    vt1 = np.mean(v_hpreds == v_true)
    st1 = np.mean(s_hpreds == s_true)
    gap = 0
    status = "HYBRID_PROMISING" if vt1 > control_results['Val_Top1'] else "NO_IMPROVEMENT"
    leaderboard.append({'Model': name, 'Val_Top1': float(vt1), 'SARAGA_EST': float(st1), 'Status': status})

# Hybrid A: Confidence
for th in [0.70, 0.80, 0.85, 0.90]:
    v_hpreds = np.where(np.array(best['v_conf']) > th, best['v_preds'], v_cpreds)
    s_hpreds = np.where(np.array(best['s_conf']) > th, best['s_preds'], s_cpreds)
    eval_hybrid(f'HybridA_th={th}', v_hpreds, s_hpreds)

# Hybrid B: Alpha mix
for a in [0.25, 0.50, 0.75]:
    v_hprob = a * np.array(v_cprob) + (1 - a) * np.array(best['v_prob'])
    s_hprob = a * np.array(s_cprob) + (1 - a) * np.array(best['s_prob'])
    eval_hybrid(f'HybridB_a={a}', np.argmax(v_hprob, axis=1), np.argmax(s_hprob, axis=1))

# Hybrid C: Reranking Control Top-3
v_hpreds = np.copy(v_cpreds)
for i in range(len(v_true)):
    if best['v_preds'][i] in v_ctop3[i]: v_hpreds[i] = best['v_preds'][i]
s_hpreds = np.copy(s_cpreds)
for i in range(len(s_true)):
    if best['s_preds'][i] in s_ctop3[i]: s_hpreds[i] = best['s_preds'][i]
eval_hybrid('HybridC_RerankTop3', v_hpreds, s_hpreds)

df = pd.DataFrame(leaderboard)
df.to_csv(os.path.join(WORKSPACE, "classical_model_leaderboard.csv"), index=False)
print(df[['Model', 'Train_Acc', 'Val_Top1', 'SARAGA_EST', 'Status']].to_string())

