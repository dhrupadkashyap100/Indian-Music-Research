"""
=========================================================
Evaluate Regularized Multi-Feature Classifier
=========================================================
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.multi_feature_config import BATCH_SIZE, LINE_SEPARATOR, SMALL_SEPARATOR
from utils.multi_feature_dataloader import create_multi_dataloaders
from models.multi_feature_efficientnet import MultiFeatureEfficientNet

EXP_DIR = PROJECT_ROOT / "outputs" / "combined_regularization"
EXP_MODEL_DIR = EXP_DIR / "models"
EXP_LOG_DIR = EXP_DIR / "logs"
EXP_REPORT_DIR = EXP_DIR / "reports"

EXP_REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = EXP_MODEL_DIR / "best_regularized_classifier.pth"
TRAIN_LOG_PATH = EXP_LOG_DIR / "regularized_training_log.csv"

def plot_training_curves():
    if not TRAIN_LOG_PATH.exists():
        print("  Training log not found.")
        return

    df = pd.read_csv(TRAIN_LOG_PATH)
    
    plt.figure(figsize=(14, 6))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
    plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", marker='s')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(df["Epoch"], df["Train Acc (%)"], label="Train Acc (%)", marker='o')
    plt.plot(df["Epoch"], df["Val Acc (%)"], label="Val Acc (%)", marker='s')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = EXP_REPORT_DIR / "learning_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Learning Curves: {plot_path.name}")


@torch.no_grad()
def evaluate_model():
    if not BEST_MODEL_PATH.exists():
        print(f"  Model checkpoint not found: {BEST_MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"  Loading best checkpoint: {BEST_MODEL_PATH.name}")
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    genre_mapping = ckpt["genre_mapping"]
    num_classes = len(genre_mapping)
    inv_mapping = {v: k for k, v in genre_mapping.items()}
    genre_names = [inv_mapping[i] for i in range(num_classes)]

    model = MultiFeatureEfficientNet(num_classes=num_classes, verbose=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _ = create_multi_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, enable_specaugment=False, verbose=False
    )
    
    all_preds = []
    all_labels = []

    print("  Running inference on Test Set...")
    for mel_batch, chroma_batch, labels, _ in test_loader:
        mel_batch = mel_batch.to(device, non_blocking=True)
        chroma_batch = chroma_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(mel_batch, chroma_batch)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=genre_names, output_dict=True, zero_division=0)
    
    # Save Report
    report_path = EXP_REPORT_DIR / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
        
    print(LINE_SEPARATOR)
    print("  TEST EVALUATION RESULTS")
    print(SMALL_SEPARATOR)
    print(f"  Accuracy  : {report_dict['accuracy']*100:.2f}%")
    print(f"  Precision : {report_dict['macro avg']['precision']:.4f}")
    print(f"  Recall    : {report_dict['macro avg']['recall']:.4f}")
    print(f"  F1 Score  : {report_dict['macro avg']['f1-score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_names, yticklabels=genre_names)
    plt.title('Confusion Matrix - Regularized Multi-Feature Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = EXP_REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Confusion Matrix: {cm_path.name}")


def main():
    print(LINE_SEPARATOR)
    print("  EVALUATE REGULARIZED MULTI-FEATURE CLASSIFIER")
    print(LINE_SEPARATOR)
    
    plot_training_curves()
    evaluate_model()
    
    print(LINE_SEPARATOR)

if __name__ == "__main__":
    main()
