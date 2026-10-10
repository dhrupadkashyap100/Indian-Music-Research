# CultureMERT Extraction Procedure
The exact configuration for the final model:
- **Foundation Model**: `ntua-slp/CultureMERT-95M`
- **Audio Preprocessing**: 24,000 Hz, Mono, 5-second overlapping chunks.
- **Embeddings**: 768-Dimensional output from the frozen model.
- **Temporal Aggregation**: 60-second aggregation using 12 × 5-second windows.
- **Normalization**: L2 Normalization on the 60-second axis.
- **KNN**: Cosine similarity against TRF reference embeddings.
- **Classifier Checkpoint**: `scripts/culturemert_full_raag_classifier/training/results/60s_BiLSTM/best_f1.pt`
- **Architecture**: `TemporalModel` with a 2-layer BiLSTM and `AttentionPooling`.
- **Classes**: 61 Raag classes.

**Relevant Scripts:**
- `scripts/culturemert_full_training.py`
- `scripts/folk_inference_pipeline.py`
- `scripts/saraga_evaluation_pipeline.py`
