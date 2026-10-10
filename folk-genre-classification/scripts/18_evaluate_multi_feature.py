"""
=========================================================
Evaluate Multi-Feature EfficientNet Classifier
=========================================================
"""

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.multi_feature_config import (
    MODEL_DIR, REPORT_DIR, BEST_MULTI_FEATURE_MODEL_NAME,
    MULTI_FEATURE_EVAL_REPORT, MULTI_FEATURE_TRAIN_STATS_FILE,
    EFFICIENTNET_TRAIN_STATS_FILE, LINE_SEPARATOR, SMALL_SEPARATOR
)
from utils.multi_feature_dataloader import create_multi_dataloaders
from models.multi_feature_efficientnet import MultiFeatureEfficientNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print(LINE_SEPARATOR)
    print("  EVALUATING MULTI-FEATURE MODEL")
    print(LINE_SEPARATOR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, genre_mapping = create_multi_dataloaders(
        batch_size=16, num_workers=4, pin_memory=True, verbose=False
    )
    num_classes = len(genre_mapping)
    inv_map = {v: k for k, v in genre_mapping.items()}

    model = MultiFeatureEfficientNet(num_classes=num_classes).to(device)
    ckpt_path = MODEL_DIR / BEST_MULTI_FEATURE_MODEL_NAME
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find best model at {ckpt_path}")
        
    print(f"  Loading weights from {ckpt_path.name}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    print("  Running inference on Test Set...")
    with torch.no_grad():
        for mel_batch, chroma_batch, batch_labels, _ in tqdm(test_loader, unit="batch"):
            mel_batch = mel_batch.to(device)
            chroma_batch = chroma_batch.to(device)
            
            outputs = model(mel_batch, chroma_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    report_dict = classification_report(
        all_labels, all_preds, target_names=[inv_map[i] for i in range(num_classes)], output_dict=True
    )
    
    per_class_acc = {}
    for i in range(num_classes):
        genre = inv_map[i]
        per_class_acc[genre] = report_dict[genre]["recall"] * 100.0 # recall is per-class accuracy

    print(SMALL_SEPARATOR)
    print(f"  Test Accuracy  : {acc:.2f}%")
    print(f"  Test F1 Score  : {f1:.4f}")
    print(f"  Test Precision : {prec:.4f}")
    print(f"  Test Recall    : {rec:.4f}")
    print(SMALL_SEPARATOR)

    # Save to report
    out_data = {
        "model": "MultiFeatureEfficientNet",
        "test_accuracy": acc,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "classification_report": report_dict
    }
    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MULTI_FEATURE_EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)
        
    print(f"  Saved evaluation report to {MULTI_FEATURE_EVAL_REPORT}")
    
    # --------------------------------------------------------
    # RESEARCH COMPARISON
    # --------------------------------------------------------
    print(LINE_SEPARATOR)
    print("  RESEARCH COMPARISON: MULTI-FEATURE vs BASELINE")
    print(LINE_SEPARATOR)
    
    mf_stats_path = MODEL_DIR / MULTI_FEATURE_TRAIN_STATS_FILE
    base_stats_path = MODEL_DIR / EFFICIENTNET_TRAIN_STATS_FILE
    
    if mf_stats_path.exists() and base_stats_path.exists():
        with open(mf_stats_path, "r") as f: mf_stats = json.load(f)
        with open(base_stats_path, "r") as f: base_stats = json.load(f)
        
        mf_val_acc = mf_stats.get("best_validation_accuracy", 0)
        base_val_acc = base_stats.get("best_validation_accuracy", 0)
        diff_val = mf_val_acc - base_val_acc
        
        print(f"  Baseline Val Acc : {base_val_acc:.2f}%")
        print(f"  Multi-Feat Val Acc: {mf_val_acc:.2f}%")
        print(f"  Improvement      : {diff_val:+.2f}%")
    else:
        print("  [Notice] Train statistics missing, cannot generate comparative diff.")

if __name__ == "__main__":
    main()
