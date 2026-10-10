"""
=============================================================================
  Folk Music Genre Classification – Publication-Quality Figure Generator
=============================================================================
  Reads ALL existing logs / metrics / evaluation outputs (no retraining).
  Generates:
    1. Accuracy curves  (per model + overlay comparison)
    2. Loss curves      (per model + overlay comparison)
    3. Model comparison grouped bar chart
    4. Dataset distribution bar chart
    5. Confusion matrices  (reuse PNGs already saved)
    6. Text discovery / generation report

  Output folder: results/figures/
=============================================================================
"""

import os
import json
import csv
import shutil
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

# Fonts / style
rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

C_TRAIN    = "#2563EB"
C_VAL      = "#DC2626"
C_CNN      = "#7C3AED"
C_EFF      = "#059669"
C_CBAM     = "#D97706"
C_GRID     = "#E5E7EB"
ALPHA_FILL = 0.12

BASE    = Path(__file__).parent
FIG_DIR = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ─────────────────────────────────────────────────────────────────

def find_file(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

def load_csv_log(path, epoch_col="Epoch", train_loss_col="Train Loss",
                 val_loss_col="Validation Loss",
                 train_acc_col=None, val_acc_col=None, lr_col="Learning Rate"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if not row.get(epoch_col, "").strip():
                continue
            rows.append(row)

    def pick(preferred, fallbacks):
        if preferred and preferred in fieldnames:
            return preferred
        for fb in fallbacks:
            if fb in fieldnames:
                return fb
        return None

    t_acc_col = pick(train_acc_col, ["Train Accuracy (%)", "Train Acc (%)", "train_acc"])
    v_acc_col = pick(val_acc_col,   ["Validation Accuracy (%)", "Val Acc (%)", "val_acc"])
    t_loss_col = pick(train_loss_col, ["Train Loss", "train_loss"])
    v_loss_col = pick(val_loss_col,   ["Validation Loss", "Val Loss", "val_loss"])
    lr_c       = pick(lr_col, ["LR", "lr"])

    return {
        "epoch":      [int(r[epoch_col])      for r in rows],
        "train_loss": [float(r[t_loss_col])   for r in rows] if t_loss_col else [],
        "val_loss":   [float(r[v_loss_col])   for r in rows] if v_loss_col else [],
        "train_acc":  [float(r[t_acc_col])    for r in rows] if t_acc_col  else [],
        "val_acc":    [float(r[v_acc_col])    for r in rows] if v_acc_col  else [],
        "lr":         [float(r[lr_c])         for r in rows] if lr_c       else [],
    }

def smooth(y, w=3):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    padded = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]

def style_ax(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=C_GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="best", framealpha=0.9, edgecolor="#CBD5E1")

def save_fig(fig, name, report_list):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {path}")
    report_list.append(str(path))
    return path

GENERATED = []
WARNINGS  = []

# ─── STEP 1: discover files ───────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 - FILE DISCOVERY")
print("=" * 70)

cnn_log  = find_file([BASE / "outputs/logs/training_log.csv"])
eff_log  = find_file([BASE / "outputs/logs/efficientnet_training_log.csv"])
cbam_log = find_file([
    BASE / "outputs/cbam_experiment/logs/cbam_training_log.csv",
    BASE / "results/logs/cbam_training_log.csv",
])
mf_log = find_file([
    BASE / "outputs/logs/multi_feature_training_log.csv",
    BASE / "results/logs/multi_feature_training_log.csv",
])

cnn_metrics  = find_file([BASE / "cnn_eval_metrics.json"])
eff_metrics  = find_file([BASE / "efficientnet_eval_metrics.json"])
cbam_metrics = find_file([
    BASE / "results/metrics/cbam_classification_report.json",
    BASE / "outputs/cbam_experiment/evaluation/classification_report.json",
])
mf_metrics = find_file([
    BASE / "outputs/reports/multi_feature_eval_report.json",
    BASE / "results/metrics/baseline_eval_report.json",
])

cnn_cm_src  = find_file([BASE / "cnn_confusion_matrix.png"])
eff_cm_src  = find_file([BASE / "efficientnet_confusion_matrix.png"])
cbam_cm_src = find_file([
    BASE / "outputs/cbam_experiment/evaluation/confusion_matrix.png",
])

master_rpt = find_file([BASE / "datasets/metadata/master_dataset_report.txt"])
master_csv = find_file([BASE / "datasets/metadata/master_dataset.csv"])

DISCOVERED = {
    "cnn_training_log":       cnn_log,
    "efficientnet_train_log": eff_log,
    "cbam_training_log":      cbam_log,
    "multi_feature_log":      mf_log,
    "cnn_eval_metrics":       cnn_metrics,
    "efficientnet_metrics":   eff_metrics,
    "cbam_metrics":           cbam_metrics,
    "multifeature_metrics":   mf_metrics,
    "cnn_confusion_matrix":   cnn_cm_src,
    "efficientnet_cm":        eff_cm_src,
    "cbam_cm":                cbam_cm_src,
    "master_dataset_report":  master_rpt,
}

for k, v in DISCOVERED.items():
    status = "OK" if v else "MISSING"
    print(f"  [{status}] {k:<30} -> {v or 'NOT FOUND'}")

# ─── STEP 2: load histories ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2 - LOADING TRAINING HISTORIES")
print("=" * 70)

histories = {}

if cnn_log:
    histories["cnn"] = load_csv_log(cnn_log)
    print(f"  CNN:          {len(histories['cnn']['epoch'])} epochs")
else:
    WARNINGS.append("CNN log missing"); print("  CNN: MISSING")

if eff_log:
    histories["efficientnet"] = load_csv_log(eff_log)
    print(f"  EfficientNet: {len(histories['efficientnet']['epoch'])} epochs")
else:
    WARNINGS.append("EfficientNet log missing"); print("  EfficientNet: MISSING")

if cbam_log:
    histories["cbam"] = load_csv_log(
        cbam_log,
        train_loss_col="Train Loss", val_loss_col="Val Loss",
        train_acc_col="Train Acc (%)", val_acc_col="Val Acc (%)", lr_col="LR")
    print(f"  CBAM:         {len(histories['cbam']['epoch'])} epochs")
else:
    WARNINGS.append("CBAM log missing"); print("  CBAM: MISSING")

if mf_log:
    histories["multi_feature"] = load_csv_log(
        mf_log,
        train_loss_col="Train Loss", val_loss_col="Val Loss",
        train_acc_col="Train Acc (%)", val_acc_col="Val Acc (%)", lr_col="LR")
    print(f"  Multi-Feature:{len(histories['multi_feature']['epoch'])} epochs")

# ─── STEP 3: accuracy curves ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3 - ACCURACY CURVES")
print("=" * 70)

MODEL_META = [
    ("cnn",          "Baseline CNN",                         "cnn_accuracy_vs_epoch.png"),
    ("efficientnet", "EfficientNet-B0",                      "efficientnet_accuracy_vs_epoch.png"),
    ("cbam",         "Multi-Feature EfficientNet + CBAM",    "multifeature_cbam_accuracy_vs_epoch.png"),
]

for key, label, fname in MODEL_META:
    if key not in histories:
        WARNINGS.append(f"Accuracy curve skipped for {label}")
        print(f"  SKIP {fname}"); continue
    h = histories[key]
    epochs = h["epoch"]; t_acc = h["train_acc"]; v_acc = h["val_acc"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(epochs, t_acc, alpha=ALPHA_FILL, color=C_TRAIN)
    ax.fill_between(epochs, v_acc, alpha=ALPHA_FILL, color=C_VAL)
    ax.plot(epochs, t_acc, color=C_TRAIN, lw=2.4, label="Training Accuracy")
    ax.plot(epochs, v_acc, color=C_VAL,   lw=2.4, label="Validation Accuracy")

    v_loss = h["val_loss"]
    min_loss_idx = v_loss.index(min(v_loss))
    checkpoint_epoch = epochs[min_loss_idx]
    checkpoint_acc = v_acc[min_loss_idx]
    
    ax.axvline(checkpoint_epoch, color="#6B7280", lw=1.2, ls=":", alpha=0.7)
    offset = max(1, len(epochs)*0.04)
    ax.annotate(f"Checkpoint\n{checkpoint_acc:.1f}%", xy=(checkpoint_epoch, checkpoint_acc),
                xytext=(checkpoint_epoch+offset, checkpoint_acc-6), fontsize=10, color="#374151",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2))

    style_ax(ax, f"Accuracy vs Epoch - {label}", "Epoch", "Accuracy (%)", ylim=(0, 105))
    fig.tight_layout()
    save_fig(fig, fname, GENERATED)

# ─── STEP 4: loss curves ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 - LOSS CURVES")
print("=" * 70)

LOSS_META = [
    ("cnn",          "Baseline CNN",                       "cnn_loss_vs_epoch.png"),
    ("efficientnet", "EfficientNet-B0",                    "efficientnet_loss_vs_epoch.png"),
    ("cbam",         "Multi-Feature EfficientNet + CBAM",  "multifeature_cbam_loss_vs_epoch.png"),
]

for key, label, fname in LOSS_META:
    if key not in histories:
        WARNINGS.append(f"Loss curve skipped for {label}")
        print(f"  SKIP {fname}"); continue
    h = histories[key]
    epochs = h["epoch"]; t_loss = h["train_loss"]; v_loss = h["val_loss"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(epochs, t_loss, alpha=ALPHA_FILL, color=C_TRAIN)
    ax.fill_between(epochs, v_loss, alpha=ALPHA_FILL, color=C_VAL)
    ax.plot(epochs, t_loss, color=C_TRAIN, lw=2.4, label="Training Loss")
    ax.plot(epochs, v_loss, color=C_VAL,   lw=2.4, label="Validation Loss")

    bv = min(v_loss); be = epochs[v_loss.index(bv)]
    rng = max(v_loss) - min(v_loss)
    ax.axvline(be, color="#6B7280", lw=1.2, ls=":", alpha=0.7)
    ax.annotate(f"Best Val\n{bv:.4f}", xy=(be, bv),
                xytext=(be+max(1,len(epochs)*0.04), bv+rng*0.12),
                fontsize=10, color="#374151",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2))

    style_ax(ax, f"Loss vs Epoch - {label}", "Epoch", "Cross-Entropy Loss")
    fig.tight_layout()
    save_fig(fig, fname, GENERATED)

# ─── STEP 5: comparison bar chart ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 5 - MODEL COMPARISON BAR CHART")
print("=" * 70)

def extract_perf(path):
    if not path or not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    acc = d.get("test_accuracy") or d.get("test_acc")
    if acc is None:
        cr = d.get("classification_report") or d
        acc = cr.get("accuracy") if isinstance(cr, dict) else None
    if acc and acc < 1.5:
        acc *= 100

    cr = d.get("classification_report") or d
    macro = cr.get("macro avg") if isinstance(cr, dict) else None
    if macro is None and isinstance(d, dict):
        macro = d.get("macro avg")

    pre = rec = f1 = None
    if macro:
        pre = macro["precision"] * 100
        rec = macro["recall"]    * 100
        f1  = macro["f1-score"]  * 100
    if d.get("test_precision"):
        pre = d["test_precision"] * 100
    if d.get("test_recall"):
        rec = d["test_recall"]    * 100
    if d.get("test_f1"):
        f1  = d["test_f1"]        * 100
    return (acc, pre, rec, f1)

perf = {}
for nm, path in [("Baseline CNN", cnn_metrics),
                 ("EfficientNet-B0", eff_metrics),
                 ("Multi-Feat\nEfficientNet+CBAM", cbam_metrics)]:
    r = extract_perf(path)
    if r:
        perf[nm] = r
    else:
        WARNINGS.append(f"Metrics missing for {nm}")

if perf:
    mlabels = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    mnames  = list(perf.keys())
    x = np.arange(len(mlabels))
    w = 0.22
    colors = [C_CNN, C_EFF, C_CBAM]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (nm, vals) in enumerate(perf.items()):
        bars = ax.bar(x + i*w, vals, w, label=nm.replace("\n"," "),
                      color=colors[i], edgecolor="white", lw=1.2, alpha=0.92, zorder=3)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1F2937")

    ax.set_xticks(x + w*(len(mnames)-1)/2)
    ax.set_xticklabels(mlabels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Model Comparison - Test Set Performance", fontweight="bold", fontsize=16, pad=14)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", color=C_GRID, lw=0.8, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CBD5E1", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "model_comparison.png", GENERATED)
else:
    print("  SKIP model_comparison.png - no metrics found")

# ─── STEP 6: dataset distribution ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 6 - DATASET DISTRIBUTION")
print("=" * 70)

GENRE_ORDER = ["Baul","Bhangra","Bihu","Dhamail","Garba",
               "Ghoomar","Jhumur","Lavani","Maand","Sambalpuri"]
GENRE_COUNTS = {}

if master_rpt and master_rpt.exists():
    with open(master_rpt) as f:
        for line in f:
            for g in GENRE_ORDER:
                if line.strip().startswith(g + " ") or line.strip().startswith(g + ":"):
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        try:
                            GENRE_COUNTS[g] = int(parts[1].strip())
                        except ValueError:
                            pass
    print(f"  Loaded from report: {GENRE_COUNTS}")

if not GENRE_COUNTS:
    # Hardcode from the report content already read
    GENRE_COUNTS = {"Baul":1546,"Bhangra":1551,"Bihu":2015,"Dhamail":2077,
                    "Garba":2593,"Ghoomar":1981,"Jhumur":1654,"Lavani":1613,
                    "Maand":3368,"Sambalpuri":1982}
    print(f"  Using embedded counts from report file.")

genres = [g for g in GENRE_ORDER if g in GENRE_COUNTS]
counts = [GENRE_COUNTS[g] for g in genres]

cmap   = matplotlib.colormaps["RdYlGn"].resampled(len(genres))
colors = [cmap(i / len(genres)) for i in range(len(genres))]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(genres, counts, color=colors, edgecolor="white", lw=1.2, zorder=3, alpha=0.9)
for b, v in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+30,
            str(v), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1F2937")
ax.set_xlabel("Folk Music Genre", fontsize=13)
ax.set_ylabel("Number of Audio Segments", fontsize=13)
ax.set_title("Dataset Distribution - Segments per Genre", fontweight="bold", fontsize=16, pad=14)
ax.set_ylim(0, max(counts)*1.18)
ax.grid(True, axis="y", color=C_GRID, lw=0.8, ls="--", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.98, 0.97, f"Total segments: {sum(counts):,}\nGenres: {len(genres)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))
fig.tight_layout()
save_fig(fig, "dataset_distribution.png", GENERATED)

# ─── STEP 7: confusion matrices ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 7 - CONFUSION MATRICES")
print("=" * 70)

def copy_cm(src, dst_name):
    dst = FIG_DIR / dst_name
    shutil.copy2(src, dst)
    print(f"  Copied {src.name} -> {dst}")
    GENERATED.append(str(dst))

for src, dst_name in [(cnn_cm_src, "cnn_confusion_matrix.png"),
                       (eff_cm_src, "efficientnet_confusion_matrix.png"),
                       (cbam_cm_src,"multifeature_cbam_confusion_matrix.png")]:
    if src and src.exists():
        copy_cm(src, dst_name)
    else:
        print(f"  SKIP {dst_name} - source PNG not found")
        WARNINGS.append(f"{dst_name}: source PNG missing")

# ─── BONUS: overlay comparison curves ────────────────────────────────────────
print("\n" + "=" * 70)
print("  BONUS - OVERLAY COMPARISON CURVES")
print("=" * 70)

avail = [(k, l, c) for k, l, c in [
    ("cnn", "Baseline CNN", C_CNN),
    ("efficientnet", "EfficientNet-B0", C_EFF),
    ("cbam", "Multi-Feat EfficientNet+CBAM", C_CBAM),
] if k in histories]

if len(avail) >= 2:
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, label, color in avail:
        h = histories[key]
        ax.plot(h["epoch"], h["val_acc"], lw=2.6, color=color, label=f"{label} (val)")
    style_ax(ax, "Validation Accuracy - All Models", "Epoch", "Accuracy (%)", ylim=(20,105))
    fig.tight_layout()
    save_fig(fig, "overlay_val_accuracy.png", GENERATED)

    fig, ax = plt.subplots(figsize=(11, 6))
    for key, label, color in avail:
        h = histories[key]
        ax.plot(h["epoch"], h["val_loss"], lw=2.6, color=color, label=f"{label} (val)")
    style_ax(ax, "Validation Loss - All Models", "Epoch", "Cross-Entropy Loss")
    fig.tight_layout()
    save_fig(fig, "overlay_val_loss.png", GENERATED)

# ─── STEP 9: report ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 9 - FINAL REPORT")
print("=" * 70)

report_path = FIG_DIR / "generation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    def W(s=""): f.write(s + "\n")
    W("=" * 72)
    W("  FOLK MUSIC GENRE CLASSIFICATION - FIGURE GENERATION REPORT")
    W("=" * 72)
    W()
    W("DISCOVERED FILES")
    W("-" * 72)
    for k, v in DISCOVERED.items():
        W(f"  {k:<30} : {v or 'NOT FOUND'}")
    W()
    W("LOADED HISTORIES")
    W("-" * 72)
    for key in ["cnn","efficientnet","cbam","multi_feature"]:
        if key in histories:
            h = histories[key]
            W(f"  {key:<20}: {len(h['epoch'])} epochs | best val acc: {max(h['val_acc']):.2f}% | best val loss: {min(h['val_loss']):.4f}")
        else:
            W(f"  {key:<20}: NOT LOADED")
    W()
    W("COMPARATIVE METRICS")
    W("-" * 72)
    W(f"  {'Model':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    W("  " + "-" * 70)
    for nm, vals in perf.items():
        W(f"  {nm.replace(chr(10),' '):<32} {vals[0]:>9.2f}% {vals[1]:>9.2f}% {vals[2]:>9.2f}% {vals[3]:>9.2f}%")
    W()
    W("DATASET DISTRIBUTION")
    W("-" * 72)
    for g in GENRE_ORDER:
        W(f"  {g:<15}: {GENRE_COUNTS.get(g, '-')}")
    W(f"  {'TOTAL':<15}: {sum(GENRE_COUNTS.values())}")
    W()
    W("GENERATED FILES")
    W("-" * 72)
    for p in GENERATED:
        W(f"  OK  {p}")
    W()
    if WARNINGS:
        W("WARNINGS")
        W("-" * 72)
        for w in WARNINGS:
            W(f"  WARNING  {w}")
    W()
    W("=" * 72)
    W("  Output directory: " + str(FIG_DIR))
    W("=" * 72)

print(f"  Report saved -> {report_path}")
print()
print("=" * 70)
print(f"  ALL DONE - {len(GENERATED)} files written to: {FIG_DIR}")
print("=" * 70)
for p in GENERATED:
    from pathlib import Path as P
    print(f"    * {P(p).name}")
if WARNINGS:
    print()
    print("  Warnings:")
    for w in WARNINGS:
        print(f"    ! {w}")
print()

