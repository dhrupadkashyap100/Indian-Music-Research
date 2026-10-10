# Folk Inference Script Audit

The upcoming Folk Inference Script will explicitly follow the architecture and schema outlined below, strictly implementing the Frozen 45.02% Hybrid Model without any modification or re-optimization. 

## Processing Pipeline
1. **Load Configuration:** Reads `production_model_config.json`.
2. **Feature Loading:** Reads the pre-extracted `_trans.npy`, `_pc.npy`, and `_f0.npy` for the 328 Folk recordings.
3. **Sequence Construction:** Folk recordings have already been segmented into 60-second sequences. N3 is calculated via the 12-bin tonic-relative cents quantization.
4. **Temporal Aggregation (Top-50%):** Sequences are scored by the geometric similarity logic, and only the Top-50% highest scoring sequences per recording are retained.
5. **Geometric Reference Bank:** Computes against the 635 TRF TRAIN recordings.
6. **Discriminative Classification:** The saved RBF-SVM receives the aggregated feature vectors and computes Platt-scaled probabilities.
7. **Hybrid Fusion:** Computes `0.75 * geometric_prob + 0.25 * svm_prob`.
8. **Prediction Generation:** Extracts the Top-1, Top-2, Top-3 Raags and their fused probabilities.

## Output Schema
As specified, the output will strictly adhere to:
```csv
recording_id,predicted_raag,top2_raag,top3_raag,top1_score,top2_score,top3_score,top1_top2_margin,confidence_category,number_of_sequences,number_of_selected_sequences,supporting_reference_id
```

## Confidence Calibration
**CONFIDENCE CALIBRATION: NOT YET VALIDATED**
Because no formal confidence threshold was validated for the Hybrid SVM on external datasets, the output will log the raw `top1_score` and `top1_top2_margin`. A `confidence_category` will be assigned loosely based on the raw probability margins (e.g., HIGH if margin > 0.1, LOW otherwise), but will be explicitly marked as unvalidated on Folk data.

STATUS: GREEN. The script is prepared for implementation but has NOT been executed.
