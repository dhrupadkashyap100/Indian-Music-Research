# TRF Feature Extraction
The CultureMERT pipeline expects:
- **Preprocessing**: 24 kHz mono audio chunked into 5-second overlapping windows.
- **Extraction**: `ntua-slp/CultureMERT-95M` frozen model generating 768-D embeddings.
- **Aggregation**: 60-second mean pooling (12 windows) and L2 normalization.
- **Script**: `load_trf_sequences()` inside `scripts/folk_inference_pipeline.py` reads the cached embeddings.
- **Cache Location**: `scripts/culturemert_full_raag_classifier/embeddings/`
