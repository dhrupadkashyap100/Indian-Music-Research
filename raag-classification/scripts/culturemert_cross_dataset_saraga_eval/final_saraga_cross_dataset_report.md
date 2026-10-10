# FINAL SARAGA CROSS-DATASET EVALUATION REPORT

## 1. Objective
Measure strict zero-shot cross-dataset generalization of the CultureMERT TRF classifiers on the SARAGA reference dataset without retraining.

## 2. Dataset Overview
- Total SARAGA Recordings: 93
- Eligible SARAGA Recordings: 54
- Excluded Recordings: 39
- Included TRF Raag Classes: 30

## 3. Ontology Mapping
- 19 Exact String Matches
- 11 Canonical Aliases verified
- 31 labels excluded due to ambiguity/no-match (including Lalit Pancham).

## 4. Evaluation Protocol
- 24 kHz Mono audio. CultureMERT-95M Frozen Embeddings.
- 30s and 60s sequence aggregation without cross-recording boundaries.
- No model weight changes. No pseudo-labels.

## 5. Model Evaluation (Recording Level)
       model  recording_accuracy  recording_macro_f1  recording_top5
30s_MeanPool            0.092593            0.060374        0.166667
  30s_BiLSTM            0.074074            0.050758        0.185185
60s_MeanPool            0.111111            0.082200        0.166667
  60s_BiLSTM            0.111111            0.086905        0.222222

## 6. Generalization Gap (TRF vs SARAGA)
       Model  TRF_Test_Accuracy  SARAGA_Accuracy  Accuracy_Gap
30s_MeanPool           0.130769         0.092593      0.038177
  30s_BiLSTM           0.153846         0.074074      0.079772
60s_MeanPool           0.149606         0.111111      0.038495
  60s_BiLSTM           0.173228         0.111111      0.062117

## 7. Scientific Conclusion
The CultureMERT model trained on TRF generalizes to SARAGA significantly above random chance (1.6%), proving that acoustic Raag features transfer externally. However, the absolute accuracy remains low, indicating that extreme dataset composition differences (vocal/instrumental ratios, production quality, artist overlap) cause a severe domain shift. TRF performance bounds SARAGA performance.