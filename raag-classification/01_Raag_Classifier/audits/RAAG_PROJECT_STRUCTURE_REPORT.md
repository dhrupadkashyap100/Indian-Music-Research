# Phase 1: Raag Classifier Project Structure Inventory

## Root Project Directory
`d:\Raaga_Recognition_Project`

## Major Folders

### Folder: `Folk_Music_Classifier`
- **Purpose:** Contains the datasets and scripts specifically for evaluating the target folk music recordings.
- **Important files:** `folk_tree.txt`, `scripts/06_split_dataset.py`, `datasets` (inferred)
- **Current status:** PRESENT
- **Used by active pipeline?** YES
- **Evidence:** `folk_inference_pipeline.py` directly references `D:\Raaga_Recognition_Project\Folk_Music_Classifier\datasets` as the source of folk audio.

### Folder: `Hindustani`
- **Purpose:** Storage for the Hindustani audio dataset used in the project, likely for CultureMERT training or validation.
- **Important files:** Audio files.
- **Current status:** PRESENT
- **Used by active pipeline?** YES
- **Evidence:** Project includes `hindustani_audio_inventory.csv` and `hindustani_dataset_manifest.csv` which catalog this folder. CultureMERT explicitly cites 200h of Hindustani data.

### Folder: `Reference_Model`
- **Purpose:** Contains the original reference implementation and dataset (SARAGA).
- **Important files:** `Raaga_Info\reference_dataset\metadata\saraga_metadata.csv`
- **Current status:** PRESENT
- **Used by active pipeline?** YES (for evaluation/benchmarking).
- **Evidence:** `run_phase3.py` explicitly references the SARAGA metadata in this directory for tonic validation.

### Folder: `Thaat and Raga Forest (TRF) Dataset`
- **Purpose:** Contains the original Thaat and Raga Forest dataset created by Majumder & Bhattacharya (2024).
- **Important files:** Audio samples as referenced by `TRF_README.md`.
- **Current status:** PRESENT
- **Used by active pipeline?** YES (used as the reference embedding space).
- **Evidence:** `folk_inference_pipeline.py` uses TRF reference sequences from `trf_training_manifest.csv` to perform KNN Raag assignment.

### Folder: `scripts`
- **Purpose:** Contains all the Python scripts for data auditing, pre-processing, training, and inference.
- **Important files:** `folk_inference_pipeline.py`, `culturemert_full_training.py`, `run_phase3.py`, `saraga_evaluation_pipeline.py`.
- **Current status:** PRESENT
- **Used by active pipeline?** YES
- **Evidence:** Verified active via Python script execution and recent edits.

### Folder: `docs`
- **Purpose:** Project documentation.
- **Important files:** `culturemert_readme.md`, `TRF_README.md`.
- **Current status:** PRESENT
- **Used by active pipeline?** NO (Documentation only)
- **Evidence:** Markdown files.

### Folder: `third_party`
- **Purpose:** External dependencies and submodules.
- **Important files:** `swaragram-master` directory.
- **Current status:** PRESENT
- **Used by active pipeline?** YES
- **Evidence:** `run_phase3.py` appends this to `sys.path` to import `swaragram.core`.

## Root Files
- **Configuration / Metadata:** `hindustani_dataset_manifest.csv`, `hindustani_ontology_mapping.csv`, `hindustani_audio_feature_pairing.csv`
- **Reports:** `verification_feature_content_analysis.md`, `hindustani_duplicate_report.csv`
- **Generated Feature Directories:** Inside `scripts/folk_culturemert_inference/embeddings/` and `scripts/culturemert_full_raag_classifier/embeddings/`.
