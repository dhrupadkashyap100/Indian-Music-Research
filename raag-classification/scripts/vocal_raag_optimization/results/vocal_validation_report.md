# Final Vocal Optimization Report

## 1. Did vocal isolation improve Top-1 accuracy?
Original: 0.1302 -> Vocal: 0.0465

## 2. Did it improve Top-3 accuracy?
Original: 0.1860 -> Vocal: 0.1023

## 3. Did it improve Macro F1?
Original: 0.0872 -> Vocal: 0.0467

## 4. Did it improve mean/median confidence margins?
Mean Top-1 Sim: 0.4620 -> 0.4597
Median Top-1 Sim: 0.4107 -> 0.4260
Mean Margin: 0.0734 -> 0.0331
Median Margin: 0.0404 -> 0.0198

## 5. Did correct-vs-incorrect margin separation improve?
Original Correct-Incorrect Diff: 0.1592
Vocal Correct-Incorrect Diff: 0.0848

## 6. How many predictions changed Wrong->Correct?
2

## 7. How many changed Correct->Wrong?
20

## 8. Which approach is scientifically stronger?
ORIGINAL PCA128 CultureMERT

## 9. Is vocal isolation worth the additional computational cost?
No

## 10. Should we proceed to the final Folk inference using:
Original PCA128 CultureMERT
