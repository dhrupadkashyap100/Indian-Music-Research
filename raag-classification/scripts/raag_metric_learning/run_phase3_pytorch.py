
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from scipy.stats import zscore

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_metric_learning")
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

train_data = {}
for k in train_keys:
    d = load_data(k)
    if d is not None:
        train_data[k] = {
            'seqs': np.hstack([d['t'], d['p'], d['n3']]),
            'raag': RAAG_TO_IDX[trf_meta.loc[k, 'raag']],
            't': d['t'].mean(axis=0),
            'p': d['p'].mean(axis=0),
            'n3': d['n3'].mean(axis=0)
        }

# Precompute Train-Train similarity using the control method for Hard Negative Mining
train_keys_valid = list(train_data.keys())
train_rr = np.array([train_data[k]['raag'] for k in train_keys_valid])
rt = np.array([train_data[k]['t'] for k in train_keys_valid])
rp = np.array([train_data[k]['p'] for k in train_keys_valid])
rn3 = np.array([train_data[k]['n3'] for k in train_keys_valid])

control_sim_matrix = 0.75 * cosine_similarity(rt, rt) + 0.20 * cosine_similarity(rp, rp) + 0.05 * cosine_similarity(rn3, rn3)
np.fill_diagonal(control_sim_matrix, -1) # don't match self

class TripletDataset(Dataset):
    def __init__(self, data_dict, keys, rraags, sim_matrix, mining_strategy='random'):
        self.data_dict = data_dict
        self.keys = keys
        self.rraags = rraags
        self.sim_matrix = sim_matrix
        self.mining = mining_strategy
        
        self.raag_to_keys = {r: [] for r in range(61)}
        for i, k in enumerate(keys):
            self.raag_to_keys[rraags[i]].append((i, k))
            
    def __len__(self):
        return len(self.keys) * 5 # epoch size
        
    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        anchor_k = self.keys[idx]
        anchor_raag = self.rraags[idx]
        
        # Positive
        pos_candidates = [k for i, k in self.raag_to_keys[anchor_raag] if k != anchor_k]
        if not pos_candidates: pos_candidates = [anchor_k]
        pos_k = np.random.choice(pos_candidates)
        
        # Negative
        if self.mining == 'random':
            neg_candidates = [k for i, k in enumerate(self.keys) if self.rraags[i] != anchor_raag]
            neg_k = np.random.choice(neg_candidates)
        elif self.mining == 'hard':
            # Hardest negative recording
            sims = self.sim_matrix[idx].copy()
            sims[self.rraags == anchor_raag] = -1 # mask out same raag
            neg_idx = np.argmax(sims)
            neg_k = self.keys[neg_idx]
        elif self.mining == 'semi-hard':
            # Randomly pick from top 10 hardest negatives
            sims = self.sim_matrix[idx].copy()
            sims[self.rraags == anchor_raag] = -1
            top_10 = np.argsort(sims)[-10:]
            neg_idx = np.random.choice(top_10)
            neg_k = self.keys[neg_idx]
            
        a_seq = self.data_dict[anchor_k]['seqs']
        p_seq = self.data_dict[pos_k]['seqs']
        n_seq = self.data_dict[neg_k]['seqs']
        
        # Randomly sample one sequence from the recording
        a_feat = a_seq[np.random.randint(len(a_seq))]
        p_feat = p_seq[np.random.randint(len(p_seq))]
        n_feat = n_seq[np.random.randint(len(n_seq))]
        
        # L2 norm inputs
        a_feat = a_feat / (np.linalg.norm(a_feat) + 1e-9)
        p_feat = p_feat / (np.linalg.norm(p_feat) + 1e-9)
        n_feat = n_feat / (np.linalg.norm(n_feat) + 1e-9)
        
        return torch.FloatTensor(a_feat), torch.FloatTensor(p_feat), torch.FloatTensor(n_feat)

class MetricNet(nn.Module):
    def __init__(self, in_dim=1884, out_dim=128, mlp=False):
        super().__init__()
        self.mlp = mlp
        if not mlp:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, out_dim)
            )
            
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)

def evaluate_model(model, keys, is_saraga=False):
    model.eval()
    
    # Build reference bank (MEAN over valid sequences for each TRAIN recording)
    ref_embs, ref_raags = [], []
    with torch.no_grad():
        for k in train_keys_valid:
            seqs = train_data[k]['seqs']
            seqs = seqs / (np.linalg.norm(seqs, axis=1, keepdims=True) + 1e-9)
            e = model(torch.FloatTensor(seqs)).numpy()
            ref_embs.append(np.mean(e, axis=0))
            ref_raags.append(train_data[k]['raag'])
            
    ref_embs = np.array(ref_embs)
    ref_embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-9)
    ref_raags = np.array(ref_raags)
    
    true, sims = [], []
    with torch.no_grad():
        for k in keys:
            d = load_data(k, is_saraga)
            if d is None: continue
            
            seqs = np.hstack([d['t'], d['p'], d['n3']])
            seqs = seqs / (np.linalg.norm(seqs, axis=1, keepdims=True) + 1e-9)
            q_e = model(torch.FloatTensor(seqs)).numpy()
            q_e = q_e / (np.linalg.norm(q_e, axis=1, keepdims=True) + 1e-9)
            
            # Retrieval
            s = cosine_similarity(q_e, ref_embs)
            rs = np.zeros((len(q_e), 61))
            for r in range(61):
                idx = np.where(ref_raags == r)[0]
                if len(idx) > 0: rs[:, r] = np.max(s[:, idx], axis=1)
            rs = zscore(rs + np.random.randn(*rs.shape)*1e-8, axis=1)
            
            # Top-50%
            num_seq = max(1, int(len(rs) * 0.5))
            if len(rs) <= num_seq: final_sim = np.mean(rs, axis=0)
            else: final_sim = np.mean(rs[np.argsort(np.max(rs, axis=1))[-num_seq:]], axis=0)
                
            mapped = sar_mapped[sar_mapped['saraga_recording_id'] == k]['mapped_trf_raag'].values[0] if is_saraga else trf_meta.loc[k, 'raag']
            true.append(RAAG_TO_IDX[mapped])
            sims.append(final_sim)
            
    true = np.array(true)
    sims = np.array(sims)
    preds = np.argmax(sims, axis=1)
    top1 = np.mean(preds == true)
    return float(top1)

def train_eval_config(mining, mlp, out_dim=128):
    print(f"Training: Mining={mining}, MLP={mlp}, Dim={out_dim}")
    model = MetricNet(out_dim=out_dim, mlp=mlp)
    dataset = TripletDataset(train_data, train_keys_valid, train_rr, control_sim_matrix, mining)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.2)
    
    best_val = 0
    for epoch in range(15): # extremely fast early stopping
        model.train()
        for a, p, n in loader:
            optimizer.zero_grad()
            ea, ep, en = model(a), model(p), model(n)
            loss = criterion(ea, ep, en)
            loss.backward()
            optimizer.step()
            
        val_acc = evaluate_model(model, val_keys, False)
        if val_acc > best_val:
            best_val = val_acc
            
    # Final Eval
    sar_acc = evaluate_model(model, sar_est_keys, True)
    params = sum(p.numel() for p in model.parameters())
    return best_val, sar_acc, params

results = []

# Phase 3 & 6: Linear with different mining
for strategy in ['random', 'semi-hard', 'hard']:
    v, s, p = train_eval_config(mining=strategy, mlp=False, out_dim=128)
    results.append({'Arch': 'Linear(128)', 'Mining': strategy, 'Params': p, 'Val_Top1': v, 'SARAGA_EST': s})
    
# Phase 4: Tiny MLP
for strategy in ['random', 'semi-hard', 'hard']:
    v, s, p = train_eval_config(mining=strategy, mlp=True, out_dim=128)
    results.append({'Arch': 'MLP(128->128)', 'Mining': strategy, 'Params': p, 'Val_Top1': v, 'SARAGA_EST': s})

res_df = pd.DataFrame(results)
print(res_df.to_string())
res_df.to_csv(os.path.join(WORKSPACE, "metric_learning_results.csv"), index=False)
