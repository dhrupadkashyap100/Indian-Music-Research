# SARAGA CULTUREMERT KNN EVALUATION REPORT

## 1. Objective
Determine whether the CultureMERT representation space retains sufficient cross-dataset similarity using parameter-free KNN retrieval, identifying if the bottleneck is the trained MLPs or the embedding space itself.

## 2. Methodology
- **Reference Bank:** 28361 (30s) and 13931 (60s) TRF sequences.
- **Queries:** 2974 (30s) and 1474 (60s) eligible SARAGA sequences.
- **Similarity:** Maximum Cosine Similarity per TRF Raag class.
- **Aggregation:** Average class similarity across all sequences in a SARAGA recording.

## 3. Results Comparison
                    Model  Accuracy  Macro_F1     Top5
30s MeanPool (Supervised)  0.092600  0.060000 0.166700
  30s BiLSTM (Supervised)  0.074100  0.051000 0.185200
60s MeanPool (Supervised)  0.111100  0.082000 0.166700
  60s BiLSTM (Supervised)  0.111100  0.087000 0.222200
       30s MeanPool (KNN)  0.092593  0.044309 0.222222
       60s MeanPool (KNN)  0.111111  0.059580 0.203704

## 4. Interpretation
- **Supervised vs Unsupervised:** If KNN outperforms the supervised MLP, it suggests the downstream MLP failed to generalize or overfitted to TRF recording acoustics. If KNN is lower, it suggests the embedding space requires complex non-linear separation.
- **Representation Stability:** Mean-pooling the 768-D representation over 30s/60s provides a stable acoustic fingerprint for cosine distance.
- **Limitations:** Unsupervised retrieval forces the embeddings to act as identity verifiers without domain adaptation. It assumes Raags form tight spherical clusters, which may not hold for highly variable Hindustani performances.

## 5. Next Steps for Folk Analysis
Depending on the KNN margins, utilizing the embedding bank as a direct similarity anchor for the unlabelled Folk dataset might prove more statistically resilient than forcing hard classifier predictions, given the extreme out-of-distribution nature of Folk music.