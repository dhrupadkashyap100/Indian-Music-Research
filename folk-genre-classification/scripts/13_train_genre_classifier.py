"""
=========================================================
Train Genre Classifier
=========================================================

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap

Description
-----------
Builds, trains, and validates a 2D Convolutional Neural Network (CNN)
for 10-class folk music genre classification using log-mel spectrograms.

Features:
- Loads DataLoaders directly via scripts.12_prepare_genre_data
- Deterministic seeding across PyTorch, NumPy, and Python random
- GPU acceleration support (CUDA) with fallback to CPU
- 4-block CNN architecture with AdaptiveAvgPool2d & Batch Normalization
- Complete training and validation loops with metric calculation
- Learning rate scheduling (ReduceLROnPlateau) and Early Stopping
- CSV training logging (training_log.csv) and JSON statistics (train_statistics.json)
- Checkpoint persistence across both outputs/models/ and checkpoints/models/

Usage:
    python scripts/13_train_genre_classifier.py [--epochs 100] [--batch-size 32] [--lr 0.001]
=========================================================
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in path for module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Configure clean UTF-8 output on Windows console if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from utils.config import (
    BATCH_SIZE,
    BEST_MODEL_NAME,
    EPOCHS,
    GENRES,
    LEARNING_RATE,
    LINE_SEPARATOR,
    LOG_DIR,
    MODEL_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    PIN_MEMORY,
    SMALL_SEPARATOR,
    TRAIN_STATS_FILE,
    TRAINING_LOG,
)
import importlib

_module_12 = importlib.import_module("scripts.12_prepare_genre_data")
create_dataloaders = _module_12.create_dataloaders


# ==========================================================
# DETERMINISTIC SEEDING
# ==========================================================

def set_seed(seed: int = 42) -> None:
    """
    Set deterministic random seeds across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================
# DEVICE SETUP
# ==========================================================

def get_device() -> torch.device:
    """
    Determine and log the execution device (CUDA GPU vs CPU).

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device           : CUDA GPU ({gpu_name})")
    else:
        device = torch.device("cpu")
        print("  Device           : CPU (CUDA not available)")
    return device


# ==========================================================
# CNN MODEL ARCHITECTURE
# ==========================================================

class GenreClassifierCNN(nn.Module):
    """
    2D Convolutional Neural Network for Log-Mel Spectrogram Genre Classification.

    Input tensor shape: (Batch, Channels=1, n_mels=128, n_frames=938)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(GenreClassifierCNN, self).__init__()

        # Block 1: (B, 1, 128, 938) -> (B, 32, 64, 469)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.30)
        )

        # Block 2: (B, 32, 64, 469) -> (B, 64, 32, 234)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.35)
        )

        # Block 3: (B, 64, 32, 234) -> (B, 128, 16, 117)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.40)
        )

        # Block 4: (B, 128, 16, 117) -> (B, 256, 1, 1) via AdaptiveAvgPool
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.45)
        )

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, 1, 128, 938).

        Returns
        -------
        torch.Tensor
            Logits of shape (Batch, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# ==========================================================
# TRAINING AND VALIDATION LOGIC
# ==========================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float, float]:
    """
    Execute one full training epoch.

    Returns
    -------
    (train_loss, train_accuracy, samples_per_sec) : Tuple[float, float, float]
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    pbar = tqdm(dataloader, desc=f"  Epoch [{epoch:03d}/{total_epochs:03d}] Train", unit="batch")
    for batch_features, batch_labels, _ in pbar:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
            
            scaler.scale(loss).backward()
            
            # Unscale the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * batch_features.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_labels).sum().item()
        total += batch_features.size(0)

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct / total) * 100:.2f}%"
        })

    end_time = time.perf_counter()
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    samples_per_sec = total / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
    return epoch_loss, epoch_acc, samples_per_sec


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation dataset without gradient tracking.

    Returns
    -------
    (val_loss, val_accuracy) : Tuple[float, float]
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"  Epoch [{epoch:03d}/{total_epochs:03d}] Val  ", unit="batch")
    for batch_features, batch_labels, _ in pbar:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        running_loss += loss.item() * batch_features.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_labels).sum().item()
        total += batch_features.size(0)

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct / total) * 100:.2f}%"
        })

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return val_loss, val_acc


# ==========================================================
# METRIC LOGGING AND PERSISTENCE
# ==========================================================

def initialise_training_log(log_path: Path) -> None:
    """
    Initialize CSV training log file with header.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Epoch", "Train Loss", "Train Accuracy (%)",
            "Validation Loss", "Validation Accuracy (%)",
            "Learning Rate", "Epoch Duration (s)", 
            "Best Val Loss", "Best Val Acc (%)", "Samples/sec", 
            "GPU Memory (MB)", "Timestamp"
        ])


def append_training_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    duration: float,
    best_val_loss: float,
    best_val_acc: float,
    samples_per_sec: float,
    gpu_mem: str,
) -> None:
    """
    Append metrics for the current epoch to the CSV training log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
            f"{val_loss:.6f}", f"{val_acc:.4f}",
            f"{lr:.8f}", f"{duration:.2f}",
            f"{best_val_loss:.6f}", f"{best_val_acc:.4f}",
            f"{samples_per_sec:.1f}", gpu_mem, timestamp
        ])


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    genre_mapping: Dict[str, int],
    target_paths: List[Path],
) -> None:
    """
    Save model weights and training metadata across all requested paths.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": val_loss,
        "best_val_acc": val_acc,
        "genre_mapping": genre_mapping,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


def save_training_statistics(
    stats_path: Path,
    best_epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    total_epochs_run: int,
    total_time: float,
    genre_mapping: Dict[str, int],
) -> None:
    """
    Persist summary training statistics as a JSON file.
    """
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_data = {
        "project": "AI-Based Analysis of Indian Traditional Folk Music",
        "model": "GenreClassifierCNN",
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_val_loss, 6),
        "best_validation_accuracy": round(best_val_acc, 4),
        "total_epochs_trained": total_epochs_run,
        "total_training_time_seconds": round(total_time, 2),
        "num_classes": len(genre_mapping),
        "genre_mapping": genre_mapping,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_data, fh, indent=4)


# ==========================================================
# MAIN TRAINING PIPELINE
# ==========================================================

def main() -> None:
    """
    Orchestrate model initialization, DataLoader setup, training loop,
    validation, early stopping, and checkpoint persistence.
    """
    parser = argparse.ArgumentParser(description="Train Folk Music Genre Classifier CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience in epochs")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    print()
    print(LINE_SEPARATOR)
    print("  FOLK MUSIC GENRE CLASSIFIER — TRAINING PIPELINE")
    print(LINE_SEPARATOR)

    # Set deterministic random seeds
    set_seed(args.seed)
    print(f"  Random Seed      : {args.seed}")

    # Determine execution device
    device = get_device()

    # Load DataLoaders via Script 12
    print(SMALL_SEPARATOR)
    print("  Loading Dataset & DataLoaders...")
    train_loader, val_loader, test_loader, genre_mapping = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        verbose=False
    )
    num_classes = len(genre_mapping)
    print(f"  Train Samples    : {len(train_loader.dataset):,}")
    print(f"  Validation       : {len(val_loader.dataset):,}")
    print(f"  Test Samples     : {len(test_loader.dataset):,}")
    print(f"  Num Classes      : {num_classes}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Max Epochs       : {args.epochs}")
    print(f"  Initial LR       : {args.lr}")
    print(f"  Early Stopping   : Patience = {args.patience}")

    # Build model, criterion, optimizer, and scheduler
    # Compute class weights automatically
    print("  Computing class weights...")
    class_counts = torch.zeros(num_classes)
    for _, labels, _ in train_loader:
        labels_bincount = torch.bincount(labels, minlength=num_classes)
        class_counts += labels_bincount
    
    if (class_counts.max() / class_counts.min()) > 1.1:
        class_weights = class_counts.sum() / (num_classes * class_counts)
        class_weights = class_weights.to(device)
        print(f"  Class Imbalance Detected. Using weights.")
    else:
        class_weights = None
        print("  Classes are balanced. Using normal loss.")

    model = GenreClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize AMP GradScaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # Initialize CSV log
    initialise_training_log(TRAINING_LOG)

    # Define checkpoint output paths
    # Saves both to outputs/models/ and checkpoints/models/ per project instructions
    checkpoints_dir = PROJECT_ROOT / "checkpoints" / "models"
    best_model_paths = [
        MODEL_DIR / BEST_MODEL_NAME,
        checkpoints_dir / BEST_MODEL_NAME
    ]
    latest_model_paths = [
        MODEL_DIR / MODEL_NAME,
        checkpoints_dir / MODEL_NAME
    ]

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    pipeline_start_time = time.perf_counter()

    print(LINE_SEPARATOR)
    print("  STARTING TRAINING")
    print(LINE_SEPARATOR)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        # Train one epoch
        train_loss, train_acc, samples_per_sec = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler
        )

        # Validate epoch
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        epoch_duration = time.perf_counter() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Note: Log appending moved lower to capture actual Best Val Loss

        # Learning rate step based on validation loss
        scheduler.step(val_loss)

        # Check for validation improvement
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            improved = True

            # Save best checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                genre_mapping, best_model_paths
            )
        else:
            epochs_without_improvement += 1

        # Always save latest model state
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            genre_mapping, latest_model_paths
        )

        gpu_mem = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f}" if device.type == 'cuda' else "N/A"
        
        # Log epoch metrics to CSV
        append_training_log(
            TRAINING_LOG, epoch, train_loss, train_acc,
            val_loss, val_acc, current_lr, epoch_duration,
            best_val_loss, best_val_acc, samples_per_sec, gpu_mem
        )

        # Print epoch summary
        status_flag = "✓ (BEST)" if improved else f"({epochs_without_improvement}/{args.patience})"
        print(
            f"  -> Epoch [{epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} {status_flag} | Speed: {samples_per_sec:.1f} samp/s | GPU Mem: {gpu_mem}MB"
        )

        # Check early stopping condition
        if epochs_without_improvement >= args.patience:
            print()
            print(SMALL_SEPARATOR)
            print(f"  [EARLY STOPPING] No validation loss improvement for {args.patience} epochs.")
            print(SMALL_SEPARATOR)
            break

    total_training_time = time.perf_counter() - pipeline_start_time

    # Save summary statistics JSON
    save_training_statistics(
        MODEL_DIR / TRAIN_STATS_FILE,
        best_epoch, best_val_loss, best_val_acc,
        epoch, total_training_time, genre_mapping
    )

    print()
    print(LINE_SEPARATOR)
    print("  TRAINING COMPLETED — SUMMARY")
    print(LINE_SEPARATOR)
    print(f"  Total Epochs Run : {epoch}")
    print(f"  Best Epoch       : {best_epoch}")
    print(f"  Best Val Loss    : {best_val_loss:.6f}")
    print(f"  Best Val Acc     : {best_val_acc:.2f}%")
    print(f"  Total Time       : {total_training_time / 60:.2f} mins ({total_training_time:.2f} s)")
    print(SMALL_SEPARATOR)
    print(f"  Best Model Saved -> {best_model_paths[0]}")
    print(f"  Checkpoints Copy -> {best_model_paths[1]}")
    print(f"  Training Log     -> {TRAINING_LOG}")
    print(f"  Train Statistics -> {MODEL_DIR / TRAIN_STATS_FILE}")
    print(LINE_SEPARATOR)
    print()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
