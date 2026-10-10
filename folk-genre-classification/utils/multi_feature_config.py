"""
=========================================================
Multi-Feature Configuration
=========================================================
"""

from utils.config import *
from pathlib import Path

# ==========================================================
# MULTI-FEATURE MODEL SPECIFICS
# ==========================================================

MULTI_FEATURE_MODEL_NAME = "multi_feature_classifier.pth"
BEST_MULTI_FEATURE_MODEL_NAME = "best_multi_feature_classifier.pth"

MULTI_FEATURE_TRAINING_LOG = LOG_DIR / "multi_feature_training_log.csv"
MULTI_FEATURE_TRAIN_STATS_FILE = "multi_feature_train_statistics.json"

MULTI_FEATURE_EVAL_REPORT = REPORT_DIR / "multi_feature_eval_report.json"

# Learning Rates (Overrides if necessary, else identical to baseline)
MULTI_FEATURE_CLASSIFIER_LR = CLASSIFIER_LR  # 1e-3
MULTI_FEATURE_BACKBONE_LR = BACKBONE_LR      # 1e-4

print("Multi-Feature Configuration Loaded")
