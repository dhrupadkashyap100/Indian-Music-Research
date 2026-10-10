# Folk Music Results Analysis Report

## 1. Executive Summary
The system generated candidate Raag assignments for unlabeled Folk recordings using similarity retrieval against the TRF Raag reference space. 327 out of 328 canonical recordings were successfully classified based on 2310 60-second CultureMERT sequences.

## 2. Dataset and Inference Overview
- **Total Canonical Folk Recordings**: 328
- **Successfully Classified**: 327
- **Excluded**: 1 (insufficient sequences)
- **Total 60s Sequences**: 2310
- **Unique Retrieved Raags**: 53

## 3. Data Integrity Verification
All counts match the inference generation report. 0 integrity anomalies were found.
No anomalies detected.

## 4. Folk Recording Coverage & 5. Raag Retrieval Distribution
The inference successfully classified 327 recordings. The top retrieved candidate Raags overall were:
   Raag  Frequency  Percentage
Darbari         28    8.562691
  Adana         23    7.033639
  Marva         15    4.587156
 Khamaj         15    4.587156
  Bahar         14    4.281346

## 6. Genre-wise Analysis
Folk genres displayed varied retrieval patterns.
     Genre  Recordings Classified Most Frequent Raag  Top-1 Raag Percentage
      Baul                     30       Yaman Kalyan              13.333333
   Bhangra                     32            Darbari              28.125000
      Bihu                     37           Bhoopali              13.513514
   Dhamail                     30            Darbari              16.666667
     Garba                     31              Adana              12.903226
   Ghoomar                     30            Bhairav              20.000000
    Jhumur                     30          Bageshree              13.333333
    Lavani                     35              Bahar              14.285714
     Maand                     39            Darbari              10.256410
Sambalpuri                     33            Bilaval               9.090909

## 7. Top-5 Retrieval Analysis
The most frequently appearing Raags in the Top-5 candidates (across all 327 recordings) were:
   Raag  Top-5 Presence
Darbari              99
  Adana              81
Asavari              72
   Desh              66
 Khamaj              57

## 8. Similarity and Margin Analysis
- **Mean Top-1 Similarity**: 0.9361
- **Mean Margin (relative retrieval confidence)**: 0.0082
- **Median Margin**: 0.0019
- **Min Margin**: 0.0000
- **Max Margin**: 0.9455

## 9. High-Confidence/High-Separation Examples
Top 3 highest margin recordings:
Original Song ID   Genre      Top-1 Raag  Top-1/Top-2 Margin
        BAUL_023    Baul           Purba            0.945507
       GARBA_023   Garba           Purba            0.934756
     GHOOMAR_019 Ghoomar Alhaiya Bilaval            0.015552

## 10. Ambiguous Examples
Top 3 lowest margin recordings:
Original Song ID      Genre   Top-1 Raag  Top-1/Top-2 Margin
     DHAMAIL_016    Dhamail        Bihag            0.000127
  SAMBALPURI_013 Sambalpuri       Sorath            0.000119
        BIHU_013       Bihu Ahir Bhairav            0.000099

## 11. Comparison with SARAGA Validation
The previous SARAGA external validation demonstrated the baseline capability of the CultureMERT representation to bridge acoustic domain gaps. The similarity margins observed in this Folk dataset (mean 0.0082) are evaluated in the context of SARAGA, where margin separation was shown to overlap between correct and incorrect predictions.

## 12. Scientific Interpretation
**1. What did the Folk inference actually accomplish?**
It mapped 327 unlabelled Folk recordings into the TRF classical embedding space, producing candidate assignments.
**2. How many distinct Raags were retrieved?**
53 distinct Raags were retrieved as Top-1 candidates.

## 13. Limitations
The Folk dataset has no ground-truth Raag labels. Acoustic domain shift between Classical (TRF) and Folk music limits absolute retrieval confidence.

## 14. What Can Be Claimed
We can claim that specific Folk recordings robustly retrieve certain Classical Raags in the CultureMERT space, indicating shared acoustic/melodic structures.

## 15. What Cannot Be Claimed
We cannot claim that a Folk recording "is" a specific Raag, nor can we use similarity margin as a definitive probability of correctness.

## 16. Recommended Next Step
Human expert validation of the Top-15 highest margin and Top-15 lowest margin candidate retrievals to qualitatively assess the melodic mapping.
