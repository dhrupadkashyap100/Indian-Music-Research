# AUDIO PREPROCESSING PIPELINE

1. **Raw Audio**
   ↓
2. **Resampling** (24kHz Mono for CultureMERT, 22.05kHz Mono for Swaragram/TRF)
   ↓
3. **Segmentation** (5s windows for CultureMERT, 30-60s for Tonic estimation)
   ↓
4. **Feature Extraction** (CultureMERT 768-D last_hidden_state mean, Swaragram LogCompression, Mel/STFT)
   ↓
5. **Temporal Pooling** (60s L2-Normalized Mean Pooling for KNN)
   ↓
6. **Model Inference** (KNN Cosine Similarity or BiLSTM)