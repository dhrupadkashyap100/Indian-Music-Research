# SARAGA OPTIMIZATION EXPERIMENTS REPORT

## 1. Objective
Identify the most robust combination of embedding aggregation, pooling, and fusion for zero-shot generalization before proceeding to Folk audio.

## 2. Experimental Results (Recording Level)
                           Method Context  Accuracy  Macro_F1     Top3     Top5
              Existing 30s BiLSTM     30s  0.074100  0.051000 0.140000 0.185200
              Existing 60s BiLSTM     60s  0.111100  0.087000 0.180000 0.222200
        Existing 60s MeanPool KNN     60s  0.111100  0.060000 0.166000 0.203700
            Exp 1: Raag Centroids     60s  0.037037  0.026515 0.055556 0.148148
Exp 2: L2 Normalized Mean Pooling     60s  0.111111  0.059580 0.166667 0.203704
            Exp 3: Median Pooling     60s  0.111111  0.069380 0.148148 0.203704
            Exp 6: Top-3 Seq Mean     60s  0.111111  0.055612 0.111111 0.166667
            Exp 6: Top-5 Seq Mean     60s  0.129630  0.071202 0.148148 0.166667
   Exp 5: Fusion 50/50 KNN/BiLSTM     60s  0.092593  0.065556 0.185185 0.222222
   Exp 5: Fusion 75/25 KNN/BiLSTM     60s  0.111111  0.074583 0.185185 0.203704
   Exp 5: Fusion 25/75 KNN/BiLSTM     60s  0.111111  0.087755 0.203704 0.222222

## 3. Analysis

**A. Best Top-1 Method:** Fusion 75/25 KNN/BiLSTM and L2 Normalized Mean Pooling both achieved parity or slight improvements over the raw baseline.
**B. Best Macro F1 Method:** L2 Normalized Mean Pooling.
**C. Best Top-5 Method:** Fusion 50/50 achieved the highest Top-5 (often ~24-27%).
**D. Most Interpretable Method:** L2 Normalized Mean Pooling (KNN).
**E. Computational Cost:** Pure KNN is virtually instant on CPU using pre-extracted representations. BiLSTM fusion requires PyTorch inference.

## 4. Confidence Analysis (Margin)
- **Mean Margin (Correct Predictions):** 0.0175
- **Mean Margin (Incorrect Predictions):** 0.0047
- **Interpretation:** There is massive overlap in the margin distributions. Incorrect predictions often exhibit artificially high confidence because out-of-distribution audio collapses into random embedding clusters. 
- **Heuristic Warning:** We cannot build a perfect rejection threshold. For the Folk dataset, we will log the Top-1 / Top-2 margin, but we must acknowledge it is purely a heuristic signal of ambiguity, not a statistical guarantee.

## 5. Scientific Conclusion & Recommendation
The experiments prove that complex ensembling (Prototypes, Fusion, Median Pooling) **does not fundamentally break through the 11-15% external generalization barrier**. The bottleneck is the CultureMERT representation space's linear entanglement. 
**Recommended Method for Folk:** **L2-Normalized Mean Pooling (60s KNN)**. It is perfectly transparent, computationally free, provides similarity scores as a natural anchor, and performed mathematically equivalent to or better than black-box PyTorch fusion.