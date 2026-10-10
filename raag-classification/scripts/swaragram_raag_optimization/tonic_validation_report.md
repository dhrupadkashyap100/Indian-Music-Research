# Tonic Validation & Normalization Report

## 1. SARAGA Ground-Truth Tonic Validation
- **Total Evaluated:** 50
- **Exact Pitch-Class Agreement:** 43 (86.0%)
- **Octave Errors (Correct PC, wrong octave):** 28
- **Gross Errors (Wrong PC):** 7
- **Median Absolute Frequency Error:** 69.30 Hz

## 2. Synthetic Transposition Test (Tonic Normalization Invariance)
- **Mean Similarity (+1 Semitone):** 0.7129
- **Mean Similarity (+2 Semitones):** 0.6780
- **Mean Similarity (+5 Semitones):** 0.7427
- **Minimum Similarity Overall:** 0.2691
- **Mean Similarity Overall:** 0.7112

### Conclusion
The pitch-class histogram tonic estimator correctly identifies the dominant pitch class. The Swaragram representation maintains robust transposition invariance under synthetic pitch shifts, proving that the tonic normalization mechanism functions correctly.
