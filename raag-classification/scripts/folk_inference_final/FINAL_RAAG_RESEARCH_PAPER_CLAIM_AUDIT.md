# Final Research Paper Claim Audit

This document formally categorizes every major scientific claim embedded in the final research paper to guarantee absolute scientific integrity and publication readiness. 

**Categorization Schema:**
- **[VERIFIED]**: A hard, empirically measured mathematical fact explicitly validated against the project codebase and artifacts.
- **[INTERPRETATION]**: A reasonable, scientifically defensible explanation for a verified observation, carefully worded to avoid asserting unproven causality.
- **[HYPOTHESIS]**: A computational mapping or prediction requiring independent future validation (e.g., human expert review).
- **[REQUIRES REFERENCE]**: A statement reliant on external prior art that currently lacks an authoritative bibliographic citation.
- **[UNSUPPORTED]**: Explicitly prohibited claims that overstate evidence, assert absolute ground truth where none exists, or invent causality.

---

## 1. Classical Raag Classification Claims
- **"The TRF Train set contains 635 recordings."** $\rightarrow$ **[VERIFIED]**
- **"The Geometric Control baseline achieved an observed 34.93% Validation accuracy."** $\rightarrow$ **[VERIFIED]**
- **"Metric Learning achieved an observed 33.49% Validation accuracy."** $\rightarrow$ **[VERIFIED]**
- **"The Hybrid SVM B model achieved an observed 45.02% Top-1 accuracy on the sealed TRF Test set."** $\rightarrow$ **[VERIFIED]**
- **"The Hybrid SVM is statistically significantly better than the control."** $\rightarrow$ **[UNSUPPORTED]** *(Must be rephrased: "The observed Top-1 accuracy increased from 43.10% to 45.02%").*
- **"The model understands Raag structures."** $\rightarrow$ **[UNSUPPORTED]** *(Must be rephrased: "The model mathematically captures structural similarities").*

## 2. Methodology & Optimization Claims
- **"The 1884-dimensional representation consists of 144 Transition, 12 PC, and 1728 N=3 features."** $\rightarrow$ **[VERIFIED]**
- **"The Hybrid geometric + SVM fusion operates at a 0.75 / 0.25 arithmetic ratio."** $\rightarrow$ **[VERIFIED]**
- **"The SVM utilized Platt scaling calibrated via TRF Train 5-fold CV."** $\rightarrow$ **[VERIFIED]**
- **"Metric learning failed because it memorized domain-specific timbral characteristics rather than musical invariants."** $\rightarrow$ **[INTERPRETATION]** *(Supported by the catastrophic drop in cross-corpus SARAGA generalization).*
- **"Hybrid SVM generalizes better due to error complementarity between the geometric space and the non-linear boundary."** $\rightarrow$ **[INTERPRETATION]** *(Supported by ablation logs).*
- **"The Yin algorithm reliably tracks monophonic $F_0$."** $\rightarrow$ **[REQUIRES REFERENCE]**

## 3. Data Isolation Claims
- **"The TRF Test set was sealed and accessed exactly once."** $\rightarrow$ **[VERIFIED]**
- **"Folk data was completely isolated and did not influence classical model training."** $\rightarrow$ **[VERIFIED]**
- **"Confidence thresholds were derived strictly from Classical TRF Train CV."** $\rightarrow$ **[VERIFIED]**

## 4. Folk Inference Claims
- **"327 Folk recordings were successfully processed."** $\rightarrow$ **[VERIFIED]**
- **"Malkauns was the most frequently generated Top-1 hypothesis (30 recordings)."** $\rightarrow$ **[VERIFIED]**
- **"86.5% of the Folk recordings fell into the Ambiguous confidence tier."** $\rightarrow$ **[VERIFIED]**
- **"30 Folk songs are definitively Malkauns."** $\rightarrow$ **[UNSUPPORTED]** *(Must be rephrased: "Malkauns was the model's structural-affinity hypothesis for 30 recordings").*
- **"Folk music relies heavily on Pahadi and Malkauns."** $\rightarrow$ **[HYPOTHESIS]** *(Generated computationally, requires human musicologist validation).*

## 5. Domain Shift Claims
- **"The observed 86.5% ambiguity rate is consistent with substantial domain shift between classical training data and Folk recordings."** $\rightarrow$ **[INTERPRETATION]**
- **"This proves Folk music is definitively Out-of-Distribution (OOD)."** $\rightarrow$ **[UNSUPPORTED]** *(Without an explicit OOD benchmark dataset, it "suggests" OOD behavior rather than "proving" it absolutely).*
- **"This proves Folk music has no Raag structure."** $\rightarrow$ **[UNSUPPORTED]** *(The model uses a closed 61-class boundary; ambiguity could simply mean the true structural root is not one of those 61 classes, or the acoustic recording is too noisy for DSP).*
