import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import OUTPUT_DIR, BEST_EFFICIENTNET_MODEL_NAME, MODEL_DIR, REPORT_DIR
import importlib

create_dataloaders = importlib.import_module("scripts.12_prepare_genre_data").create_dataloaders
EfficientNetClassifier = importlib.import_module("models.efficientnet_classifier").EfficientNetClassifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataloaders...")
    _, _, test_loader, genre_mapping = create_dataloaders(
        batch_size=32,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        verbose=False
    )
    
    num_classes = len(genre_mapping)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    model_path = MODEL_DIR / BEST_EFFICIENTNET_MODEL_NAME
    if not model_path.exists():
        model_path = PROJECT_ROOT / "outputs" / "models" / "best_efficientnet_classifier.pth"
    print(f"Loading checkpoint from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    best_epoch = checkpoint.get("epoch", "N/A")
    best_val_acc = checkpoint.get("best_val_acc", "N/A")
    print(f"Checkpoint Best Epoch: {best_epoch}")
    print(f"Checkpoint Best Val Acc: {best_val_acc}")
    
    model = EfficientNetClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    print("Running inference...")
    with torch.no_grad():
        for batch_features, batch_labels, _ in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item() * batch_features.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds) * 100.0
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Reverse mapping
    inv_mapping = {v: k for k, v in genre_mapping.items()}
    class_names = [inv_mapping[i] for i in range(num_classes)]
    
    # Classification report
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = PROJECT_ROOT / "efficientnet_classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report to {report_csv_path}")
    
    # JSON for easy parsing
    with open(PROJECT_ROOT / "efficientnet_eval_metrics.json", "w") as f:
        json.dump({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "classification_report": report_dict
        }, f, indent=4)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('EfficientNet-B0 Confusion Matrix')
    cm_path = PROJECT_ROOT / "efficientnet_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
if __name__ == "__main__":
    main()
