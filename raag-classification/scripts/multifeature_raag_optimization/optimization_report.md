# Multi-Feature Raag Optimization Final Report

## 1. Feature Analysis
1. **Did Mel improve over CultureMERT?** No. (10.69% alone vs 12.55% MERT).
2. **Did Chroma improve?** No. (12.55% alone, identical to MERT, with much lower margin).
3. **Did Swaragram improve?** No. (8.83% alone).

## 2. Fusion Performance
4. **Which feature combination was best?** Fusion_Pitch_Dominant (MERT 0.40, Chroma 0.25, Swara 0.20, Mel 0.15).
5. **Did multi-feature beat PCA128 CultureMERT?** Barely on raw validation accuracy, but failed significantly on confidence margin.
6. **By exactly how many percentage points?** +0.93% (13.48% vs 12.55%).
7. **Did Macro F1 improve?** Yes, slightly (0.089 vs 0.074).
8. **Did Top-1/Top-2 margins improve?** NO. The margin halved from 0.084 to 0.045, making predictions significantly less confident.
9. **Did the improvement survive the TRF Test?** Fusion Test Top-1: 0.1660 vs MERT Test Top-1: 0.1277.
10. **Did it survive SARAGA?** Fusion SARAGA Top-1: 0.0926 vs MERT SARAGA Top-1: 0.0926.

## 3. Final Method Selection
11. **What is the simplest scientifically defensible final method?** The existing **CultureMERT PCA128 + Whitening** baseline. According to CASE B of the decision rule, the marginal <1% improvement does not justify the massive loss in margin separation and the added complexity of computing STFT/CQT across the Folk dataset.
12. **What exact frozen configuration should be used for the 328 Folk recordings?**
   - Extract 60s CultureMERT embeddings.
   - Transform via TRF Train-fitted PCA-128 + Whitening.
   - L2 Normalize.
   - Cosine Similarity against all TRF references.
   - Group by Raag using MAX reference similarity.
   - Aggregate temporal sequences using MEAN.
   - Rank all 61 Raags and return Top-1, 2, 3 with margins.
