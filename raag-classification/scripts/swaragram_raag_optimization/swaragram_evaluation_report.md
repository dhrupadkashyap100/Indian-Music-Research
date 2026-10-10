# Swaragram Raag Retrieval Evaluation Report

## 1. Does Swaragram outperform CultureMERT?
- **CultureMERT Val Accuracy:** 0.1302
- **Swaragram Mean Val Accuracy:** 0.0930
- **Swaragram Top20 Val Accuracy:** 0.0930

## 2. Does Top-20%-frame pooling outperform simple mean pooling?
- Difference in Accuracy: 0.0000

## 3. Does Swaragram improve Top-1 accuracy?
See section 1.

## 4. Does it improve Top-3?
- **CultureMERT Top-3:** 0.1860
- **Swaragram Top-3:** 0.1674

## 5. Does it improve Macro F1?
- **CultureMERT Macro F1:** 0.0872
- **Swaragram Macro F1:** 0.0722

## 6. Does it improve Margin_1_2?
- **CultureMERT Margin_1_2:** 0.0734
- **Swaragram Margin_1_2:** 0.0078

## 7. Does it improve correct-vs-incorrect margin separation?
- **CultureMERT Separation:** 0.1592
- **Swaragram Separation:** 0.0138

## 8. Does fusion outperform CultureMERT?
- **Best Fusion Accuracy:** 0.1395

## 9. Does the improvement survive the TRF Test?
- **Test Accuracy (using Fusion_CM50_SW50):** 0.1574

## 10. Is Swaragram actually useful enough to justify using it for Folk inference?
Based on the metrics above, the final selected model is **Fusion_CM50_SW50**.
