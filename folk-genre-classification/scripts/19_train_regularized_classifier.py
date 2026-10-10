"""
=========================================================
Train Regularized Multi-Feature Classifier
=========================================================

Project:
AI-Based Analysis of Indian Traditional Folk Music

Description
-----------
An isolated experimental branch to train the MultiFeatureEfficientNet
with combined regularization techniques (SpecAugment, Label Smoothing, 
Weight Decay, increased Dropout, and lowered LRs).
=========================================================
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.multi_feature_config import (
    EPOCHS, LINE_SEPARATOR, SMALL_SEPARATOR,
    FREEZE_EPOCHS, UNFREEZE_BLOCKS
)
from utils.multi_feature_dataloader import create_multi_dataloaders
from models.multi_feature_efficientnet import MultiFeatureEfficientNet


# ==========================================================
# ISOLATED EXPERIMENTAL DIRECTORIES
# ==========================================================
EXP_DIR = PROJECT_ROOT / "outputs" / "combined_regularization"
EXP_MODEL_DIR = EXP_DIR / "models"
EXP_LOG_DIR = EXP_DIR / "logs"

for d in [EXP_DIR, EXP_MODEL_DIR, EXP_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXP_MODEL_NAME = "regularized_classifier.pth"
BEST_EXP_MODEL_NAME = "best_regularized_classifier.pth"
EXP_TRAINING_LOG = EXP_LOG_DIR / "regularized_training_log.csv"
EXP_TRAIN_STATS_FILE = "regularized_train_statistics.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device           : CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("  Device           : CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    fast_dev_run: bool = False
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    pbar = tqdm(dataloader, desc=f"  Epoch [{epoch:03d}/{total_epochs:03d}] Train", unit="batch")
    for mel_batch, chroma_batch, batch_labels, _ in pbar:
        mel_batch = mel_batch.to(device, non_blocking=True)
        chroma_batch = chroma_batch.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(mel_batch, chroma_batch)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(mel_batch, chroma_batch)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * mel_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_labels).sum().item()
        total += mel_batch.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(correct/total)*100:.2f}%"})
        
        if fast_dev_run and total > 50:
            break

    end_time = time.perf_counter()
    return running_loss/total, (correct/total)*100.0, total/(end_time - start_time)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    fast_dev_run: bool = False
) -> Tuple[float, float, float, float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"  Epoch [{epoch:03d}/{total_epochs:03d}] Val  ", unit="batch")
    for mel_batch, chroma_batch, batch_labels, _ in pbar:
        mel_batch = mel_batch.to(device, non_blocking=True)
        chroma_batch = chroma_batch.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        outputs = model(mel_batch, chroma_batch)
        loss = criterion(outputs, batch_labels)

        running_loss += loss.item() * mel_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_labels).sum().item()
        total += mel_batch.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{(correct/total)*100:.2f}%"})
        
        if fast_dev_run and total > 50:
            break

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    
    return running_loss/total, (correct/total)*100.0, precision, recall, f1


def initialise_training_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Epoch", "Train Loss", "Train Acc (%)",
            "Val Loss", "Val Acc (%)", "Val Precision", "Val Recall", "Val F1",
            "LR", "Epoch Dur (s)", "Samples/sec", "GPU Mem (MB)", "Timestamp"
        ])

def append_training_log(
    log_path: Path, epoch: int, t_loss: float, t_acc: float,
    v_loss: float, v_acc: float, v_prec: float, v_rec: float, v_f1: float,
    lr: float, dur: float, sps: float, gpu_mem: str
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            epoch, f"{t_loss:.6f}", f"{t_acc:.4f}",
            f"{v_loss:.6f}", f"{v_acc:.4f}", f"{v_prec:.4f}", f"{v_rec:.4f}", f"{v_f1:.4f}",
            f"{lr:.8f}", f"{dur:.2f}", f"{sps:.1f}", gpu_mem, timestamp
        ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run only a few batches for testing")
    
    # Regularization Arguments
    parser.add_argument("--specaugment", action="store_true", default=True, help="Enable SpecAugment on Mel")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label Smoothing factor")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.6, help="Classifier dropout rate")
    parser.add_argument("--cls-lr", type=float, default=5e-4, help="Classifier learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-5, help="Backbone learning rate")
    
    args = parser.parse_args()

    print(LINE_SEPARATOR)
    print("  COMBINED REGULARIZATION EXPERIMENT — TRAINING PIPELINE")
    print(LINE_SEPARATOR)
    print(f"  SpecAugment       : {'Enabled' if args.specaugment else 'Disabled'}")
    print(f"  Label Smoothing   : {args.label_smoothing}")
    print(f"  Weight Decay      : {args.weight_decay}")
    print(f"  Dropout           : {args.dropout}")
    print(f"  Classifier LR     : {args.cls_lr}")
    print(f"  Backbone LR       : {args.backbone_lr}")
    print(LINE_SEPARATOR)

    set_seed(42)
    device = get_device()

    # Pass enable_specaugment to DataLoader
    train_loader, val_loader, test_loader, genre_mapping = create_multi_dataloaders(
        batch_size=args.batch_size, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available(), 
        enable_specaugment=args.specaugment,
        verbose=False
    )
    num_classes = len(genre_mapping)
    
    print(f"  Train Samples    : {len(train_loader.dataset)}")
    print(f"  Val Samples      : {len(val_loader.dataset)}")
    print(f"  Batch Size       : {args.batch_size}")

    # Initialize model with parameterized dropout
    model = MultiFeatureEfficientNet(num_classes=num_classes, dropout_rate=args.dropout).to(device)
    model.freeze_backbone()
    backbone_frozen = True
    print(f"  Stage 1: Backbone frozen. Encoders/Fusion/Classifier training.")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = optim.AdamW([
        {'params': model.mel_encoder.parameters(), 'lr': args.cls_lr},
        {'params': model.chroma_encoder.parameters(), 'lr': args.cls_lr},
        {'params': model.fusion.parameters(), 'lr': args.cls_lr},
        {'params': model.backbone.classifier.parameters(), 'lr': args.cls_lr},
        {'params': model.backbone.features.parameters(), 'lr': args.backbone_lr}
    ], weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_paths = [EXP_MODEL_DIR / BEST_EXP_MODEL_NAME]
    latest_paths = [EXP_MODEL_DIR / EXP_MODEL_NAME]

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc, best_val_f1 = 0.0, 0.0

    if args.resume and latest_paths[0].exists():
        print(f"  Resuming from {latest_paths[0]}...")
        ckpt = torch.load(latest_paths[0], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        if start_epoch > FREEZE_EPOCHS:
            model.unfreeze_backbone(num_blocks=UNFREEZE_BLOCKS)
            backbone_frozen = False
    else:
        initialise_training_log(EXP_TRAINING_LOG)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        
        if epoch == FREEZE_EPOCHS + 1 and backbone_frozen:
            print("\n  [+] Initiating fine-tuning phase. Unfreezing EfficientNet.")
            model.unfreeze_backbone(num_blocks=UNFREEZE_BLOCKS)
            backbone_frozen = False
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
            epochs_without_improvement = 0

        train_loss, train_acc, sps = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler, args.fast_dev_run)
        val_loss, val_acc, v_prec, v_rec, v_f1 = validate_epoch(model, val_loader, criterion, device, epoch, args.epochs, args.fast_dev_run)

        duration = time.perf_counter() - epoch_start_time
        scheduler.step(val_loss)

        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = v_f1
            epochs_without_improvement = 0
            improved = True
            
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": val_loss, "best_val_acc": val_acc,
                "genre_mapping": genre_mapping
            }
            for p in best_paths:
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, p)
        else:
            epochs_without_improvement += 1

        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss, "genre_mapping": genre_mapping
        }
        for p in latest_paths:
            torch.save(ckpt, p)

        gpu_mem = f"{torch.cuda.max_memory_allocated(device)/(1024**2):.1f}" if device.type=='cuda' else "N/A"
        append_training_log(EXP_TRAINING_LOG, epoch, train_loss, train_acc, val_loss, val_acc, v_prec, v_rec, v_f1, optimizer.param_groups[0]["lr"], duration, sps, gpu_mem)

        status_flag = "[BEST]" if improved else f"({epochs_without_improvement}/12)"
        print(f"  -> Epoch [{epoch:03d}] T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | V-Acc: {val_acc:.2f}% | V-F1: {v_f1:.4f} {status_flag}")

        if epochs_without_improvement >= 12:
            print("  [EARLY STOPPING]")
            break

    print(LINE_SEPARATOR)
    print("  REGULARIZED TRAINING COMPLETED")
    print(f"  Best Val Loss : {best_val_loss:.6f}")
    print(f"  Best Val Acc  : {best_val_acc:.2f}%")
    
    stats_data = {
        "model": "MultiFeatureEfficientNet_Regularized",
        "best_validation_loss": round(best_val_loss, 6),
        "best_validation_accuracy": round(best_val_acc, 4),
        "best_validation_f1": round(best_val_f1, 4),
        "total_epochs_trained": epoch,
        "genre_mapping": genre_mapping
    }
    with open(EXP_MODEL_DIR / EXP_TRAIN_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)

if __name__ == "__main__":
    main()
