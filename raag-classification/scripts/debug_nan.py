
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Load a few embeddings
EXP_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier")
EMB_DIR = os.path.join(EXP_DIR, "embeddings")
test_file = os.listdir(EMB_DIR)[0]
emb = np.load(os.path.join(EMB_DIR, test_file))

print(f"Embedding shape: {emb.shape}")
print(f"Embedding min: {emb.min()}, max: {emb.max()}, mean: {emb.mean()}, std: {emb.std()}")

if np.isnan(emb).any():
    print("WARNING: NaN in embeddings!")

# Test Model A forward pass
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

model = BaselineModel(61)
model.train()

# Mock batch
X = torch.tensor(emb[:6]).unsqueeze(0).float() # [1, 6, 768]
# Duplicate to batch size 2 for batchnorm
X = torch.cat([X, X], dim=0)

out = model(X)
print(f"Output min: {out.min().item()}, max: {out.max().item()}")

y = torch.tensor([0, 1])
crit = nn.CrossEntropyLoss()
loss = crit(out, y)
print(f"Initial loss: {loss.item()}")
loss.backward()
print("Backward passed without error.")
