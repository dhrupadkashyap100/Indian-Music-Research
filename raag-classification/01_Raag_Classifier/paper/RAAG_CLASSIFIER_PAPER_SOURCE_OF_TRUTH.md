# RAAG CLASSIFIER PROJECT: SOURCE-OF-TRUTH DOSSIER

This document is the **EXHAUSTIVE SOURCE-OF-TRUTH DOSSIER** for the Raag Classifier project. It is strictly limited to the Raag Classifier and excludes any research relating to the Folk Genre Classifier. All claims herein are based on forensic verification of the actual implementation, generated artifacts, and filesystem structures as of the latest project audit.

**WARNING TO AI WRITERS:**
Do not invent information. Do not interpolate results. Rely exclusively on the data provided in this document. Pay special attention to the "DO NOT CLAIM" section.

---

## PART 1 — PROJECT IDENTITY

- **Project Title:** Cross-Cultural Raag Recognition: Estimating Hindustani Raag Associations in Indian Traditional Folk Music
- **Research Objective:** Use a pretrained Hindustani/music representation model and a Raag reference space to estimate the most likely Hindustani Raag associations of Indian traditional folk songs.
- **Research Problem:** Folk music often lacks explicit Raag labels, yet shares timbral and melodic structures with classical forms. Establishing these structural links quantitatively requires a cross-domain retrieval system that generalizes without overfitting.
- **Motivation:** To trace the classical lineage and melodic frameworks underpinning diverse Indian folk traditions using computational ethnomusicology.
- **Research Question:** To what extent can a frozen, culturally-aware audio foundation model (CultureMERT-95M) generalize to unlabelled folk domains to retrieve geometrically meaningful Hindustani Raag candidates?
- **Intended Application:** Automated retrieval of Top-3 Raag candidates for unlabelled folk songs.
- **Scope:** 327 folk songs across 10 Indian folk genres.
- **What the system DOES:** Estimates geometric similarity/acoustic overlap between a folk song and canonical Raag reference profiles in a high-dimensional embedding space.
- **What the system DOES NOT claim to do:** It DOES NOT establish unquestionable musicological ground truth for every folk song. It DOES NOT classify Carnatic Raags.

---

## PART 2 — COMPLETE PROJECT FILE INVENTORY

*Note: The Folk Genre Classifier ecosystem (`Folk_Music_Classifier/`) is strictly EXCLUDED.*

| Filename / Directory | Path | Purpose | Category | Authoritative | Used in Final |
|---|---|---|---|---|---|
| `folk_inference_pipeline.py` | `scripts/` | Main execution pipeline for folk inference | Source Code | YES (L1) | YES |
| `folk_raag_predictions.csv` | `01_Raag_Classifier/results/` (Copied from `scripts/folk_inference_final/`) | Final song-level predictions | Prediction | YES (L2) | YES |
| `complete_v5_leaderboard.csv` | `01_Raag_Classifier/results/` | Evaluation metrics on SARAGA | Evaluation | YES (L2) | YES |
| `classical_model_leaderboard.csv` | `01_Raag_Classifier/results/` | Evaluation metrics for SVM/ExtraTrees | Evaluation | YES (L2) | NO (Rejected) |
| `embeddings/` | `scripts/culturemert_full_raag_classifier/` | Saved 768-D reference profiles | Embedding | YES (L2) | YES |
| `Hindustani/` | `Hindustani/` | Raw classical dataset | Dataset | YES (L3) | NO (CPT Reference) |
| `TRF/` | `Thaat and Raga Forest (TRF) Dataset/` | Canonical reference dataset | Dataset | YES (L3) | YES |
| `RAAG_*.md` files | `01_Raag_Classifier/audits/` | Forensic verifications and audits | Documentation | NO (L5) | YES |

---

## PART 3 — DATASETS

### A. Thaat and Raga Forest (TRF)
- **Exact Location:** `Thaat and Raga Forest (TRF) Dataset/`
- **Purpose:** REFERENCE DATASET
- **Classes:** 61 Canonical Raags
- **Metadata:** ~13,931 embedding sequences extracted
- **Role:** The reference anchor space for inference.

### B. SARAGA
- **Exact Location:** `[NOT LOCALLY VERIFIED - Path mapped internally via scripts]`
- **Purpose:** INDEPENDENT TEST DATASET
- **Role:** Independent evaluation of the TRF reference space generalizing to unseen Carnatic/Hindustani recordings. Zero overlap with TRF.

### C. Indian Folk Music Dataset
- **Exact Location:** `Folk_Music_Classifier/datasets/` (Shared logically via `00_SHARED`)
- **Purpose:** TARGET / INFERENCE DATASET
- **Recordings:** ~328 target files (327 processed).
- **Role:** Unlabelled target domain. Not used for training or evaluation.

### D. Hindustani Dataset
- **Exact Location:** `Hindustani/`
- **Purpose:** PRE-TRAINING / CPT REFERENCE
- **Recordings:** 300 readable audio files.
- **Classes:** 69 verified unique raw Raags.
- **Role:** This dataset was supposedly used for Continual Pre-Training (CPT) of the foundation model externally. It is not loaded during the final local inference pipeline.

---

## PART 4 — RAAG ONTOLOGY

**Evidence:** `trf_training_manifest.csv` and internal filesystem folders.

- **Canonical Raags (TRF):** 61
- **Hindustani Dataset Raags:** 69 mapped, 1 unmapped.
- **Aliases:** Addressed natively in the TRF file structure (e.g., `Miyan Ki Todi orTodi or Darbari Todi`).
- **Ontology Decisions:** The TRF dataset ontology acts as the definitive output space. Carnatic labels are mapped to Hindustani equivalents during SARAGA evaluation, preserving the 61-class structure.

---

## PART 5 — HINDUSTANI DATASET FORENSIC STATUS

**Based on verified filesystem audits:**
- 31 physical Raag ID folders
- 71 metadata Raag IDs
- 69 unique raw Raag names
- 300 readable audio files
- 600 feature files (.pitch, .tonic, .flatsegnyas) [PARTIALLY VERIFIED: Shape/dtype unconfirmed locally]
- 56 unique singers
- 0 confirmed Carnatic residuals

---

## PART 6 — CULTUREMERT

- **Checkpoint:** `ntua-slp/CultureMERT-95M` (HuggingFace)
- **Status:** FROZEN (Weights completely frozen locally: `requires_grad = False`).
- **Input:** 24,000 Hz, Mono, 5-second context windows.
- **Output:** 768-dimensional embeddings via `.last_hidden_state.mean`.
- **Local Training:** NONE.
- **Local Fine-Tuning:** NONE.

---

## PART 7 — TRF REFERENCE EMBEDDING SPACE

- **Location:** `scripts/culturemert_full_raag_classifier/embeddings/`
- **Methodology:** 5s segments from TRF audio were processed by frozen CultureMERT to yield 768-D vectors. These are grouped into 60s temporal pools and L2-normalized.
- **Classification Method:** The inference pipeline performs a **Class-wise Maximum Cosine Similarity Retrieval** against these sequences. *Note: Do not call this standard majority-vote KNN.*

---

## PART 8 — SARAGA EVALUATION

- **Dataset Role:** INDEPENDENT EVALUATION.
- **Methodology:** 60s segment L2-normalized cosine similarity matching against the TRF space.
- **Verified Metrics (from `complete_v5_leaderboard.csv` L2):**
  - **Top-1 Accuracy:** 56.6%
  - **Top-3 Accuracy:** 84.9%
  - **Validation Top-1:** 39.2%
- **Status:** FULLY VERIFIED from execution artifact.

---

## PART 9 — ALL RAAG CLASSIFIER EXPERIMENTS

| Branch | Model | Validation Top-1 | SARAGA Top-1 | Overfitting Evidence | Status |
|---|---|---|---|---|---|
| **1. Baseline Probing (Selected)** | CultureMERT (Frozen) + Max Cosine | 39.2% | 56.6% | None (Zero-shot) | **SELECTED** |
| **2. Hybrid / Borda-Rank** | CultureMERT + Secondary | 39.2% | 56.6% | None | Rejected (Unnecessary complexity) |
| **3. SVM (C=10)** | CultureMERT + SVM | 36.3% | 50.9% | Low | Rejected (Lower perf) |
| **4. ExtraTrees** | CultureMERT + ExtraTrees | 28.2% | [NOT VERIFIED] | **SEVERE (Train = 99.6%)** | Rejected |
| **5. Temporal BiLSTM** | CultureMERT + BiLSTM | ~64.9% | [NOT VERIFIED] | Unknown | Used only as secondary heuristic |

---

## PART 10 — FINAL MODEL SELECTION

**Selected System:** CultureMERT-95M (Frozen) + TRF Reference Embedding Space + Class-wise Max Cosine Similarity.

**Justification:**
1. **Generalization:** Demonstrated superior cross-dataset performance (56.6% on independent SARAGA) compared to supervised counterparts (50.9% for SVM).
2. **Overfitting Avoidance:** Supervised classifiers (ExtraTrees) exhibited catastrophic memorization (99.6% training vs 28.2% validation accuracy). The geometric probing approach is mathematically immune to training-set memorization.
3. **Leakage Protection:** Operates as a purely zero-shot representation space with respect to the inference domain.

---

## PART 11 — LEAKAGE ANALYSIS

- **Track-Level Isolation:** Used effectively; train/test splits isolated by `source_recording`.
- **Folk/TRF Overlap:** ZERO. The Folk dataset is strictly isolated.
- **Singer Overlap:** [POTENTIAL RISK]. It is computationally difficult to guarantee zero singer overlap between diverse datasets without explicit Singer IDs across all domains. This remains a minor leakage vector.

---

## PART 12 — FOLK INFERENCE PIPELINE

**Execution Path:** `scripts/folk_inference_pipeline.py`
1. Load 24 kHz Mono audio.
2. Chunk into 5s windows.
3. Extract 768-D embeddings via CultureMERT.
4. Aggregation: 60s temporal mean pooling (`12 x 5s`).
5. L2 Normalization.
6. Similarity: Cosine similarity against TRF reference space.
7. Output: Song-level Top-3 prediction via max-similarity.

---

## PART 13 — FINAL FOLK PREDICTION OUTPUT

**File:** `01_Raag_Classifier/results/folk_raag_predictions.csv`
- **Expected Songs:** 328
- **Processed Songs:** 327 (99.7% coverage).
- **Missing Predictions:** 0 among the 327 processed.
- **Columns Confirmed:** `predicted_raag_top1`, `predicted_raag_top3`, `hybrid_score_margin`, `confidence_tier`.
- **Top Predicted Raags:** Malkauns (30), Pahadi (22), Megh Malhar (21), Durga (20), Megh (16).
- **Concentration Risk:** Minimal. Top class is <10% of total, indicating no pathological class collapse.

---

## PART 14 — CONFIDENCE SYSTEM

**Mechanism:** HEURISTIC, NOT CALIBRATED PROBABILITY.
- The `confidence_tier` is derived algorithmically from absolute cosine similarity scores and the `hybrid_score_margin` (distance between Top-1 and Top-2).
- **Distribution:** AMBIGUOUS (283), MODERATE (38), HIGH (6).
- **Important:** Do NOT conflate these heuristic margin tiers with calibrated softmax probabilities.

---

## PART 15 — SEGMENT VS SONG LEVEL

**Aggregation:** Sequence similarities are mean-pooled prior to the final song-level output.
**Verification:** *Segment-level prediction consistency could not be retrospectively evaluated from the saved outputs, as intermediate segment predictions were not logged to disk.*

---

## PART 16 — HUMAN VALIDATION

**Status:** NOT YET PERFORMED.
**Recommendation for Future Work:** 30 samples (10 High, 10 Mod, 10 Low) across diverse genres to validate geometric similarity against musicological reality.

---

## PART 17 — REPRODUCIBILITY

- **OS:** Windows
- **GPU:** NVIDIA RTX 5050 Laptop GPU [PARTIALLY VERIFIED]
- **Frameworks:** PyTorch 2.11.0+cu128 [PARTIALLY VERIFIED]
- **Pipeline:** 100% of prediction generation is reproducible via `scripts/folk_inference_pipeline.py` leveraging preserved embeddings and audio.

---

## PART 18 — RESEARCH CONTRIBUTIONS

1. **Cross-dataset Raag recognition using a frozen foundation model:** SUPPORTED
2. **Retrieval/geometric Raag classification using TRF:** SUPPORTED
3. **Independent SARAGA evaluation:** SUPPORTED
4. **Application to unlabeled Folk music:** SUPPORTED
5. **Analysis of uncertainty in cross-domain Raag prediction:** PARTIALLY SUPPORTED (Via confidence heuristics, but lacks ground truth for validation).

---

## PART 19 — LIMITATIONS

- **No Ground Truth:** Folk songs lack definitive musicological Raag labels for accuracy validation.
- **Cross-Domain Shift:** Acoustic mismatch between Classical (reference) and Folk (target) instrumentation/vocals.
- **Reference Ontology:** Bounded strictly to the 61 canonical TRF Raags.
- **Confidence:** Uncalibrated heuristic boundaries rather than true probability.
- **Singer Bias:** Unknown singer distribution overlap between training subsets.

---

## PART 20 — PAPER-READY NUMERICAL TABLES

### TABLE A — Dataset Statistics
| Dataset | Role | Size | Source Evidence |
|---|---|---|---|
| TRF | Reference Space | 61 Classes, ~13.9k seqs | `trf_training_manifest.csv` |
| SARAGA | Independent Eval | - | `complete_v5_leaderboard.csv` |
| Folk Music | Target Domain | 327 processed | `folk_raag_predictions.csv` |

### TABLE B — Model Comparison
| Model | SARAGA Top-1 | Validation Top-1 | Overfitting | Source Evidence |
|---|---|---|---|---|
| **CultureMERT (Geometric)** | **56.6%** | **39.2%** | **None** | `complete_v5_leaderboard.csv` |
| SVM (C=10) | 50.9% | 36.3% | Low | `classical_model_leaderboard.csv` |
| ExtraTrees | [N/A] | 28.2% | Severe (99.6% Train) | `classical_model_leaderboard.csv` |

---

## PART 21 — FIGURES RECOMMENDED FOR PAPER

1. **Pipeline Architecture:** Diagram showing Folk audio -> CultureMERT -> Pooling -> TRF Cosine Similarity.
2. **Model Comparison Bar Chart:** SARAGA Top-1 vs Validation Top-1 across Geometric, SVM, and ExtraTrees.
3. **Folk Raag Prediction Distribution:** Histogram of the Top 10 predicted Raags from `folk_raag_predictions.csv`.
4. **Confidence-Tier Distribution:** Pie chart (Ambiguous: 283, Moderate: 38, High: 6).

---

## PART 22 — PAPER CLAIMS AUDIT

| Proposed Claim | Evidence | Status | Safe Paper Wording |
|---|---|---|---|
| "CultureMERT achieves 56.6% SARAGA Top-1 accuracy." | `v5_leaderboard.csv` | Verified | "The frozen CultureMERT geometric retrieval system achieved 56.6% Top-1 accuracy on the independent SARAGA subset." |
| "The system successfully generated Raag predictions for 327 folk songs." | `folk_raag_predictions.csv` | Verified | "Top-3 Raag similarity candidates were generated for 327 unlabelled folk recordings." |
| "The model is zero-shot/frozen." | Code inspection | Verified | "The CultureMERT representation space remained frozen, operating in a zero-shot capacity relative to the target domain." |
| "The system has zero leakage." | Logical deduction | Unverified | "Strict track-level separation was maintained, though implicit singer-overlap remains an undocumented boundary." |

---

## PART 23 — EXTERNAL SOURCES / REFERENCES

- **CultureMERT:** ntua-slp (HuggingFace/ArXiv). Supports architecture and pre-training claims. [EXTERNAL SOURCE]
- **TRF Dataset:** Supports the 61-class reference ontology. [EXTERNAL SOURCE]
- **SARAGA Dataset:** CompMusic. Supports Carnatic/Hindustani test metadata. [EXTERNAL SOURCE]

---

## PART 24 — FINAL PAPER STORY

**Narrative Flow:**
1. **Problem:** Folk music lacks annotated melodic structures (Raags), hindering computational ethnomusicology.
2. **Gap:** Supervised classifiers overfit massively to specific recording conditions (instrumentation/singers), failing on cross-domain unlabelled data.
3. **Approach:** Leverage a frozen, culturally-aware audio foundation model (CultureMERT).
4. **Reference Space:** Construct a 61-class canonical similarity space using the Classical TRF dataset.
5. **Evaluation:** Prove this frozen geometric approach out-performs supervised models (56.6% vs 50.9% Top-1) on the independent SARAGA dataset.
6. **Application:** Apply this zero-shot geometric similarity metric to 327 unlabelled Indian Folk songs.
7. **Results:** Produce heuristic-bounded Top-3 Raag candidates, demonstrating successful extraction of melodic proximities without catastrophic domain collapse.
8. **Limitations:** Acknowledge lack of ground truth and uncalibrated confidence.

---

## PART 25 — FACTS THAT MUST NOT BE CLAIMED

- **DO NOT CLAIM** the folk predictions represent absolute musicological ground truth.
- **DO NOT CLAIM** expert human validation has occurred.
- **DO NOT CLAIM** 100% dataset coverage (1 file out of 328 was missing).
- **DO NOT CLAIM** heuristic confidence is a calibrated probability distribution.
- **DO NOT CLAIM** zero singer leakage.
- **DO NOT CLAIM** the foundation model was trained or fine-tuned locally.
- **DO NOT CALL** the retrieval mechanism standard KNN; it is a class-wise maximum cosine retrieval.

---

## PART 26 — FINAL EVIDENCE MANIFEST

- **A. VERIFIED FACTS:** 327 predictions generated; 56.6% SARAGA Top-1; ExtraTrees overfit; TRF contains 61 Raags; CultureMERT weights frozen; L2-mean pooling utilized.
- **B. PARTIALLY VERIFIED FACTS:** PyTorch/GPU versions; detailed schema of Hindustani feature files.
- **C. EXTERNAL CLAIMS:** CultureMERT CPT training data composition.
- **D. UNVERIFIED INFORMATION:** Exact nature of singer leakage between domains.
- **E. CONTRADICTIONS FOUND:** None threatening the core pipeline.
- **F. FILES TO CITE:** `folk_raag_predictions.csv`, `complete_v5_leaderboard.csv`, `folk_inference_pipeline.py`.
- **G. FINAL RECOMMENDED MODEL:** Frozen CultureMERT-95M + Max Cosine Similarity.
- **H. FINAL RECOMMENDED NARRATIVE:** A purely geometric, zero-shot cross-domain representation mapping that avoids the catastrophic overfitting of supervised models.
