# Baseline EfficientNet-B0 Test Evaluation Report

## Phase 1 – Checkpoint Verification
- **Checkpoint Path**: `D:\Raaga_Recognition_Project\Folk_Music_Classifier\outputs\models\best_efficientnet_classifier.pth`
- **Best Epoch**: 30 
- **Validation Accuracy at Save Time**: 70.82%

## Phase 2 – Test Dataset Verification
- **Number of Test Samples**: 3607
- **Number of Classes**: 10
- **Samples per Class (Support)**:
  - Baul: 271
  - Bhangra: 185
  - Bihu: 340
  - Dhamail: 361
  - Garba: 528
  - Ghoomar: 292
  - Jhumur: 275
  - Lavani: 250
  - Maand: 714
  - Sambalpuri: 391
*(These match the Baseline CNN evaluation exactly, ensuring a fair testing ground.)*

## Phase 3 – Inference
- Inference successfully completed across the entire test loader without shuffling.
- Evaluated entirely in `model.eval()` mode with `torch.no_grad()`.

## Phase 4 – Performance Metrics
- **Test Accuracy**: 70.14%
- **Test Loss**: 1.0834
- **Precision (Macro)**: 0.6986
- **Recall (Macro)**: 0.7445
- **F1-score (Macro)**: 0.6938

**Per-class Metrics**:
| Genre | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| Baul | 0.5597 | 0.6052 | 0.5816 |
| Bhangra | 0.6437 | 0.9081 | 0.7534 |
| Bihu | 0.8218 | 0.8412 | 0.8314 |
| Dhamail | 0.6243 | 0.6260 | 0.6252 |
| Garba | 0.7143 | 0.1515 | 0.2500 |
| Ghoomar | 0.7553 | 0.9726 | 0.8503 |
| Jhumur | 0.5281 | 0.9236 | 0.6720 |
| Lavani | 0.7782 | 0.8000 | 0.7890 |
| Maand | 0.7662 | 0.7297 | 0.7475 |
| Sambalpuri | 0.7941 | 0.8875 | 0.8382 |

## Phase 5 – Confusion Matrix
- `![EfficientNet-B0 Confusion Matrix](efficientnet_confusion_matrix.png)`
- **Strongest genres**: Ghoomar (Recall: 97.26%), Jhumur (Recall: 92.36%), Bhangra (Recall: 90.81%).
- **Weakest genres**: Garba (Recall: 15.15%) remains the bottleneck.
- **Qualitative Comparison to CNN**: EfficientNet significantly tightened predictions around the correct classes for Dhamail, Baul, and Bhangra compared to the scattered CNN matrix. However, it still exhibits systematic failure when isolating Garba from other large classes like Maand.

## Phase 6 – Classification Report
- The full classification report was successfully generated and saved to `efficientnet_classification_report.csv`.

## Phase 7 – Error Analysis
- **Frequently confused genres**: Garba heavily misclassifies into Maand and other dominant rhythm-heavy classes.
- **Impact of acoustic similarity**: EfficientNet's spatial MBConv blocks extract much richer hierarchical features from the Mel-spectrogram than the CNN. This drastically improved performance on classes with unique textural characteristics (e.g., Dhamail F1 rose from 0.42 to 0.62). However, genres heavily overlapping in pure rhythm and timbre still confuse the network.
- **Impact of class imbalance**: The model remains overly biased against Garba, effectively sacrificing recall (0.15) to maintain precision (0.71). 
- **Improvement over CNN**: EfficientNet proves that a deeper, more sophisticated spatial feature extractor improves generalisation, but single-feature Mel-Spectrogram processing hits a hard ceiling when rhythmic structures overlap identically between genres.

## Phase 8 – Generate Visualizations
- High-resolution confusion matrix saved as `efficientnet_confusion_matrix.png`.

## Phase 9 – Comparative Preparation

| Metric | EfficientNet-B0 |
| :--- | :--- |
| **Train Accuracy** | 95.40% *(Peak accuracy from logs)* |
| **Validation Accuracy** | 70.82% *(From loaded checkpoint)* |
| **Test Accuracy** | 70.14% |
| **Test Loss** | 1.0834 |
| **Precision** | 0.6986 |
| **Recall** | 0.7445 |
| **F1-score** | 0.6938 |
| **Best Epoch** | 30 *(From loaded checkpoint)* |

## Phase 10 – Comparative Analysis with CNN
- **Improvement in Test Accuracy**: Increased by +5.04% (from 65.10% to 70.14%).
- **Improvement in Precision**: Increased by +0.0493 (from 0.6493 to 0.6986).
- **Improvement in Recall**: Increased by +0.0649 (from 0.6796 to 0.7445).
- **Improvement in F1-score**: Increased by +0.0541 (from 0.6397 to 0.6938).
- **Changes in confusion matrix**: The dispersion of false positives is vastly reduced. Dhamail classifications became much sharper.
- **Genres that benefited most**: Dhamail (F1 +0.203), Bhangra (F1 +0.305), Bihu (F1 +0.097).
- **Genres that remain difficult**: Garba actually degraded in recall (from 20.6% to 15.1%), cementing the hypothesis that a single Mel-Spectrogram is insufficient to disentangle it without explicit pitch/harmonic tracking (Chromagram) and attention (CBAM).

## Final Questions Answered Explicitly

1. **Was EfficientNet-B0 successfully evaluated?** 
   Yes.
2. **What is the final Test Accuracy?** 
   70.14%.
3. **What are the final Precision, Recall, and F1-score?** 
   Precision: 0.6986, Recall: 0.7445, F1-score: 0.6938.
4. **Which genres achieved the highest and lowest performance?** 
   Highest: Ghoomar (F1: 0.8503). Lowest: Garba (F1: 0.2500).
5. **Which genre pairs are most frequently confused?** 
   Garba continues to be heavily confused with Maand and other dominant rhythm classes.
6. **How much did EfficientNet-B0 improve over the Baseline CNN?** 
   It delivered a massive +5.04% absolute improvement in Test Accuracy and boosted the Macro F1 by +0.0541, demonstrating vastly superior generalisation.
7. **Did any runtime errors occur?** 
   No runtime errors occurred; the evaluation script executed flawlessly.
8. **Is EfficientNet-B0 evaluation now complete and ready for comparison with the proposed Multi-Feature EfficientNet-B0 + CBAM model?** 
   Yes, the single-feature baseline evaluation is completely finished and ready for final comparative tables.
