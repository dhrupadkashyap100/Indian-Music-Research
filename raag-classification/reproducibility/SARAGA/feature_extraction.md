# SARAGA Feature Extraction
The CultureMERT pipeline expects 5-second overlapping chunks at 24 kHz mono.
- **Script**: `scripts/saraga_evaluation_pipeline.py` automatically handles processing if raw audio is provided in the expected directory.
- **Model**: `ntua-slp/CultureMERT-95M` (frozen) extracts 768-dimensional embeddings.
- **Aggregation**: Embeddings are 60-second aggregated (12 × 5-second windows) and L2 normalized.
- **Storage**: Cached as `.npy` in `scripts/culturemert_cross_dataset_saraga_eval/saraga_embeddings/`
