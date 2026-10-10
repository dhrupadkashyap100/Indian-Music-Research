# 01 Raag Classifier Research Project

**Objective:** Use a Hindustani Raag classification/reference system to assign probable Hindustani Raag associations to Indian traditional folk music.

## Architecture
- **Foundation Model:** CultureMERT-95M
- **Reference Dataset:** TRF (Thaat and Raga Forest)
- **Evaluation:** SARAGA
- **Methodology:** 60s Temporal Mean Pooling, KNN Cosine Similarity

## Safe Organization Protocol
To strictly preserve execution reproducibility, **source code, datasets, and embeddings were NOT physically moved into this folder**. 
They remain fully intact in their original structural locations:
- **Scripts:** `../scripts/` (e.g. `../scripts/folk_inference_pipeline.py`)
- **Embeddings:** `../scripts/culturemert_full_raag_classifier/embeddings/`
- **Datasets:** `../Hindustani/` and `../Thaat and Raga Forest (TRF) Dataset/`

Finalized results (CSVs) and Markdown audit reports have been safely **copied** into `results/` and `audits/` for easy reference when drafting the research paper.
