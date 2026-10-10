
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter

EXP_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier")
DATA_DIR = os.path.join(EXP_DIR, "data")
EMB_DIR = os.path.join(EXP_DIR, "embeddings")
TRAIN_DIR = os.path.join(EXP_DIR, "training")

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.mlp(x)

class RaagSequenceDataset(Dataset):
    def __init__(self, sequences, labels, rec_ids):
        self.sequences = sequences
        self.labels = labels
        self.rec_ids = rec_ids
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): 
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.rec_ids[idx]

df_meta = pd.read_csv(os.path.join(DATA_DIR, "embedding_metadata.csv"))
df_manifest = pd.read_csv(os.path.join(DATA_DIR, "trf_training_manifest.csv"))
all_raags = sorted(df_manifest['raag'].unique())
label2id = {r: i for i, r in enumerate(all_raags)}

df_split = pd.read_csv(os.path.join(TRAIN_DIR, "recording_split.csv"))
train_recs = df_split[df_split['split'] == 'train']['source_recording'].tolist()

print("Loading embeddings for train...")
rec_to_emb = {}
for rec in train_recs:
    npy_path = os.path.join(EMB_DIR, f"{rec}.npy")
    if os.path.exists(npy_path):
        rec_to_emb[rec] = np.load(npy_path)

seqs, labels, rids = [], [], []
for rec in train_recs:
    if rec not in rec_to_emb: continue
    emb = rec_to_emb[rec]
    num_seqs = len(emb) // 6
    if num_seqs == 0: continue
    label = label2id[df_manifest[df_manifest['source_recording'] == rec].iloc[0]['raag']]
    for i in range(num_seqs):
        seqs.append(emb[i*6:(i+1)*6])
        labels.append(label)
        rids.append(rec)

loader = DataLoader(RaagSequenceDataset(seqs, labels, rids), batch_size=128, shuffle=True)

class_counts = Counter(labels)
weights = torch.zeros(len(all_raags))
for i in range(len(all_raags)):
    count = class_counts.get(i, 0)
    weights[i] = len(labels) / (len(all_raags) * count) if count > 0 else 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaselineModel(61).to(device)
crit = nn.CrossEntropyLoss(weight=weights.to(device))
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

model.train()
for i, (X, y, _) in enumerate(loader):
    X, y = X.to(device), y.to(device)
    
    if torch.isnan(X).any():
        print(f"Batch {i}: X contains NaN!")
        break
        
    opt.zero_grad()
    out = model(X)
    
    if torch.isnan(out).any():
        print(f"Batch {i}: out contains NaN!")
        break
        
    loss = crit(out, y)
    
    if torch.isnan(loss):
        print(f"Batch {i}: Loss is NaN!")
        # Debug why loss is NaN
        print("Outputs:", out)
        print("Labels:", y)
        print("Weights for labels:", weights[y])
        print("Sum of weights:", weights[y].sum())
        break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    if i % 10 == 0:
        print(f"Batch {i} | Loss: {loss.item():.4f}")

print("Debug complete.")
