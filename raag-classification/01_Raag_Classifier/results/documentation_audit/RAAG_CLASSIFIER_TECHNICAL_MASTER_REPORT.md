# RAAG CLASSIFIER — TECHNICAL MASTER REPORT

## 1. Executive Summary
This report documents the forensic read-only audit of the Raag Classifier, an ML pipeline designed to infer Hindustani Raag associations in Indian traditional folk music. The system utilizes a frozen CultureMERT-95M foundation model to construct a geometric zero-shot retrieval space.

## 2. Project Objective
To map unlabelled Indian traditional folk music to canonical Hindustani Raag origins by estimating acoustic similarity against a reference classical music anchor space.

## 3. System Scope
The system involves extracting sequences from reference (TRF), evaluation (SARAGA), and target (Folk) domains, evaluating traditional vs. geometric classifiers, and generating statistical cross-tabulations of the target inferences.

## 4. Dataset
- **Reference:** TRF (Thaat and Raga Forest)
- **Evaluation:** SARAGA subset
- **Target:** Indian Folk Music (10 Genres)

## 5. Dataset Statistics
- **TRF:** 61 Raag classes, ~13,931 sequence embeddings.
- **SARAGA:** 53 evaluation instances.
- **Folk:** 328 isolated targets, 327 successfully processed and verified.

## 6. Audio Preprocessing
- **Resampling:** 24,000 Hz.
- **Channels:** Mono downmix.
- **Segmentation:** 5-second contiguous windows (120,000 samples) without overlap. Remainder is truncated.

## 7. Feature Extraction
- **Foundation Model:** `ntua-slp/CultureMERT-95M`.
- **Status:** Frozen (`requires_grad = False`).

## 8. Feature Representation Details
- **Output Layer:** `.last_hidden_state.mean(dim=1)`
- **Dimensionality:** 768-D.
- **Temporal Aggregation:** 60-second temporal mean pooling (aggregating 12 contiguous 5s windows).
- **Normalization:** L2 Vector Normalization is applied post-pooling.

## 9. Model Architecture
- **Final Model:** Zero-Shot Geometric Retrieval.
- **Retrieval Mechanism:** Class-wise Maximum Cosine Similarity.
- **Classical Experiments:** SVM (RBF Kernel), ExtraTrees.
- **Hybrid Experiments:** Temporal BiLSTM with Attention Pooling.

## 10. Training Configuration
- **CultureMERT:** No local training / zero-shot.
- **Classical Models:** Scikit-learn implementations fit on the TRF 60s normalized sequence embeddings.

## 11. Data Loading and Engineering Optimizations
- **Bottleneck Resolution:** Pre-extracted embeddings saved as `.npy` cache files to bypass the heavy audio extraction bottleneck during rapid prototyping and evaluation.

## 12. End-to-End Pipeline
Audio Ingestion -> 24kHz Mono Resampling -> 5s Chunking -> Frozen CultureMERT Embedding -> 60s Mean Pooling -> L2 Normalization -> Cosine Similarity Matrix -> Class-wise Maximum Retrieval -> Heuristic Confidence Categorization.

## 13. Experiment History
- Evaluated supervised Classical ML vs Zero-Shot Retrieval.
- Supervised models demonstrated severe structural overfitting and poor cross-domain generalization compared to geometric anchor spacing.

## 14. Classical Models
- **ExtraTrees (depth=None):** Train=99.6%, Val_Top1=28.23%, SARAGA_Top1=37.74% (Rejected).
- **SVM (C=10, scale):** Val_Top1=36.36%, SARAGA_Top1=50.94% (Rejected).

## 15. Hybrid and Reranking Models
- Evaluated BiLSTM and Hybrid approaches, but zero-shot retrieval remained structurally superior for pure generalization.

## 16. Ablation Study
- Technically classified as a Model Baseline Comparison (comparing discrete algorithmic paradigms), rather than an architectural ablation. 

## 17. Final Model Selection
- **CultureMERT Zero-Shot Retrieval:** Val_Top1=39.23%, SARAGA_Top1=56.60%, SARAGA_Top3=84.91%. Accepted as final architecture.

## 18. Folk Music Inference Pipeline
Folk songs traverse the exact 60s L2-normalized embedding pipeline, producing distance vectors against the 61 TRF Raags. Top-1 and Top-3 candidates are logged alongside margin confidences.

## 19. Final 327-Song Genre Mapping
Metadata mismatching between filenames and prediction IDs was resolved iteratively via string similarity/fuzzy logic, culminating in exactly 327 manually verified genre mappings.

## 20. Folk Genre × Raag Results
- **Dominant Predictions:** Malkauns (30), Pahadi (22), Megh Malhar (21), Durga (20), Megh (16).
- **Genre Patterns:** Jhumur dominant in Durga (10); Bihu dominant in Megh (7).

## 21. Statistical Analysis
- **Jaccard Index:** Calculated over shared predicted Raags (e.g., Baul-Dhamail = 0.46) indicating structural melodic overlap.

## 22. Figures and Visualizations
- `model_comparison.png`, `top_raags_distribution.png`, `architecture.mmd`, `genre_raag_heatmap_top25.png`, `genre_dominant_raag_bar.png`, `genre_jaccard_heatmap.png`.

## 23. Technical Challenges and Solutions
- Overfitting: Resolved by abandoning supervised learning for geometric retrieval.
- Sub-domain Label Shift: Resolved by manually mapping Carnatic SARAGA labels to Hindustani TRF labels.

## 24. Limitations
- Inference labels are not human ground-truth.
- Implicit singer-identity overlap is a theoretical leakage vector.
- Confidence margins are heuristic, not calibrated probabilities.

## 25. Hardware and Software Environment
- Frameworks: PyTorch, Transformers, Scikit-Learn, Librosa, Pandas.

## 26. Final Verified Results
- Refer to `FINAL_RESEARCH_RESULTS.xlsx` and component CSVs.

## 27. Reproducibility Information
- Random state seeds were fixed across classical baselines; the core CultureMERT inference is deterministic.

## 28. Source File Inventory
- `01_Raag_Classifier/results/` (Primary output matrices)
- `scripts/folk_inference_pipeline.py` (Core pipeline)
- `scripts/culturemert_cross_dataset_saraga_eval/` (Evaluation metrics)

## 29. Evidence / Verification Table
- All reported metrics are verified directly against raw logs and metadata files.
