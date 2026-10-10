# REPORT WRITING NOTES

### MUST INCLUDE
- The final architecture is a Zero-Shot Geometric Retrieval pipeline using a frozen CultureMERT-95M foundation model.
- 5s window embeddings (24kHz Mono) are mean-pooled over 60s and L2-normalized.
- Retrieval uses Class-wise Maximum Cosine Similarity against the TRF anchor space (61 Raags).
- Supervised classical models (ExtraTrees, SVM) were rejected due to overfitting (ExtraTrees: 99.6% train vs 28.2% val) and lower generalization.
- Exactly 327 folk recordings across 10 genres were successfully processed out of 328 targets.

### SHOULD INCLUDE
- SARAGA dataset (53 instances) provided the independent test set for validating cross-domain generalization.
- Confidence tiers are heuristic bounds based on cosine score margin (Top-1 - Top-2).
- The Jaccard matrix reveals shared melodic domains across specific folk genres (e.g., Baul-Dhamail at 0.46).

### DO NOT CLAIM
- Do not claim the model provides ground-truth Raag labels for folk songs (they are geometrically inferred similarities).
- Do not claim the system was locally fine-tuned or trained via backpropagation on the audio (the model weights are strictly frozen).
- Do not claim the confidence tiers are calibrated softmax probabilities.
- Do not claim an 'ablation study' for the comparison between distinct model families (CultureMERT vs SVM vs Trees is a Model Comparison).
- Do not claim causal cultural/historical conclusions from the inferred similarities.
