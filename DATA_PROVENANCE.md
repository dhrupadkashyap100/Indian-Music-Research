# Data Provenance

This document outlines the origins, rights, and distribution status of data used in the **Indian Music Research** repository.

## 1. Original Metadata & Source Code
All source code, Python scripts, configuration files, and evaluation logic found in this repository are original contributions by the research authors. The metadata files (such as `dataset_duration.csv`, `raag_dataset_summary.csv`) containing structural info about the datasets are provided to ensure complete reproducibility.

## 2. Audio Data (EXCLUDED)
**Status:** NOT DISTRIBUTED
- **Folk Music Dataset (10 Genres):** Proprietary or restricted-distribution dataset. Contains audio for genres: Baul, Bhangra, Bihu, Dhamail, Garba, Ghoomar, Jhumur, Lavani, Maand, Sambalpuri.
- **Thaat and Raga Forest (TRF):** Used for reference Hindustani Raag analysis.
- **SARAGA:** Used for evaluation.
No `.wav`, `.mp3`, or other audio formats are present in this repository to prevent unauthorized redistribution.

## 3. Derived Features and Checkpoints
**Status:** DISTRIBUTED (where permitted and size-appropriate)
- **Extracted Features (Log-Mel Spectrograms, CQT, Swaragrams):** Provided as numpy/pickle structures when applicable.
- **CultureMERT Embeddings:** Generated representations from the CultureMERT model used for metric learning and Raag classification.
- **Model Checkpoints:** Foundational and finetuned `.pt`/`.pth` checkpoints for research reproduction. Sizeable checkpoints are managed via Git LFS.

## Licensing
Users must verify the licenses of the underlying third-party datasets (like SARAGA) before applying the derived features or pre-trained models to commercial use cases. 
