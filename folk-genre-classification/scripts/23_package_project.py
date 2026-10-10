import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "archive" / "development_tools"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_BASELINE = RESULTS_DIR / "baseline"
RESULTS_CBAM = RESULTS_DIR / "cbam"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_REPORTS = RESULTS_DIR / "reports"
RESULTS_METRICS = RESULTS_DIR / "metrics"
RESULTS_LOGS = RESULTS_DIR / "logs"

# Phase 2: Archive Development Files
def archive_files():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    files_to_archive = [
        "scratch_audit.py",
        "scratch_params.py",
        "scratch_edit.py",
        "scratch_edit2.py",
        "verify_cbam_forward.py",
        "gen_chroma.py"
    ]
    for file_name in files_to_archive:
        src = PROJECT_ROOT / file_name
        if src.exists():
            dst = ARCHIVE_DIR / file_name
            shutil.move(str(src), str(dst))
            print(f"Moved {file_name} to archive.")

def clean_pycache():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(pycache_path)
            print(f"Removed {pycache_path}")

# Phase 3: Organize Results
def setup_results():
    for d in [RESULTS_BASELINE, RESULTS_CBAM, RESULTS_FIGURES, RESULTS_REPORTS, RESULTS_METRICS, RESULTS_LOGS]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy baseline checkpoints
    src_baseline = PROJECT_ROOT / "checkpoints" / "models" / "best_multi_feature_classifier.pth"
    if src_baseline.exists():
        shutil.copy2(src_baseline, RESULTS_BASELINE / "best_multi_feature_classifier.pth")
    
    # Copy cbam checkpoints
    src_cbam = PROJECT_ROOT / "outputs" / "cbam_experiment" / "models" / "best_cbam_classifier.pth"
    if src_cbam.exists():
        shutil.copy2(src_cbam, RESULTS_CBAM / "best_cbam_classifier.pth")
        
    # Copy logs
    src_baseline_log = PROJECT_ROOT / "outputs" / "logs" / "multi_feature_training_log.csv"
    if src_baseline_log.exists():
        shutil.copy2(src_baseline_log, RESULTS_LOGS / "multi_feature_training_log.csv")
    src_cbam_log = PROJECT_ROOT / "outputs" / "cbam_experiment" / "logs" / "cbam_training_log.csv"
    if src_cbam_log.exists():
        shutil.copy2(src_cbam_log, RESULTS_LOGS / "cbam_training_log.csv")
        
    # Copy metrics and reports
    src_baseline_report = PROJECT_ROOT / "outputs" / "reports" / "multi_feature_eval_report.json"
    if src_baseline_report.exists():
        shutil.copy2(src_baseline_report, RESULTS_METRICS / "baseline_eval_report.json")
        
    src_cbam_report = PROJECT_ROOT / "outputs" / "cbam_experiment" / "evaluation" / "classification_report.json"
    if src_cbam_report.exists():
        shutil.copy2(src_cbam_report, RESULTS_METRICS / "cbam_classification_report.json")
        
    src_cbam_cm = PROJECT_ROOT / "outputs" / "cbam_experiment" / "evaluation" / "confusion_matrix.png"
    if src_cbam_cm.exists():
        shutil.copy2(src_cbam_cm, RESULTS_FIGURES / "confusion_matrix.png")

# Phase 4: Generate Research Figures
def generate_figures():
    baseline_log = RESULTS_LOGS / "multi_feature_training_log.csv"
    cbam_log = RESULTS_LOGS / "cbam_training_log.csv"
    
    if not (baseline_log.exists() and cbam_log.exists()):
        print("Logs not found. Cannot generate figures.")
        return

    df_base = pd.read_csv(baseline_log)
    df_cbam = pd.read_csv(cbam_log)

    sns.set_theme(style="whitegrid")
    
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(RESULTS_FIGURES / f"{name}.png", dpi=300)
        plt.savefig(RESULTS_FIGURES / f"{name}.pdf")
        plt.close()

    # Accuracy vs Epoch
    plt.figure(figsize=(8,6))
    plt.plot(df_base["Epoch"], df_base["Train Acc (%)"], label="Baseline Train Acc", linestyle="--")
    plt.plot(df_base["Epoch"], df_base["Val Acc (%)"], label="Baseline Val Acc", marker="o")
    plt.plot(df_cbam["Epoch"], df_cbam["Train Acc (%)"], label="CBAM Train Acc", linestyle="--")
    plt.plot(df_cbam["Epoch"], df_cbam["Val Acc (%)"], label="CBAM Val Acc", marker="s")
    plt.title("Accuracy vs Epoch (Baseline vs CBAM)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    save_fig("accuracy_vs_epoch_comparison")

    # Loss vs Epoch
    plt.figure(figsize=(8,6))
    plt.plot(df_base["Epoch"], df_base["Train Loss"], label="Baseline Train Loss", linestyle="--")
    plt.plot(df_base["Epoch"], df_base["Val Loss"], label="Baseline Val Loss", marker="o")
    plt.plot(df_cbam["Epoch"], df_cbam["Train Loss"], label="CBAM Train Loss", linestyle="--")
    plt.plot(df_cbam["Epoch"], df_cbam["Val Loss"], label="CBAM Val Loss", marker="s")
    plt.title("Loss vs Epoch (Baseline vs CBAM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_fig("loss_vs_epoch_comparison")

    # Metrics comparison bar chart
    metrics = ['Precision', 'Recall', 'F1-Score']
    baseline_vals = [0.6579, 0.6847, 0.6556]
    cbam_vals = [0.6783, 0.7108, 0.6788]
    
    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8,6))
    plt.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline')
    plt.bar([i + width/2 for i in x], cbam_vals, width, label='CBAM')
    plt.ylabel('Score')
    plt.title('Baseline vs CBAM Overall Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0.5, 0.8)
    save_fig("performance_comparison_bar")

    print("Figures generated successfully.")

if __name__ == "__main__":
    archive_files()
    clean_pycache()
    setup_results()
    generate_figures()
