# Baseline CNN Test Evaluation Report

## Phase 1 – Checkpoint Verification
- **Checkpoint Path**: `D:\Raaga_Recognition_Project\Folk_Music_Classifier\outputs\models\best_genre_classifier.pth`
- **Best Epoch**: 2 
- **Validation Accuracy at Save Time**: 65.73%

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

## Phase 3 – Inference
- Inference successfully completed across the entire un-shuffled test loader using the saved weights in `model.eval()` mode. 
- Gradients were completely disabled (`torch.no_grad()`).

## Phase 4 – Compute Metrics
- **Test Accuracy**: 65.10%
- **Test Loss**: 1.2743
- **Precision (Macro)**: 0.6493
- **Recall (Macro)**: 0.6796
- **F1-score (Macro)**: 0.6397

**Per-class Metrics**:
| Genre | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| Baul | 0.4132 | 0.6937 | 0.5179 |
| Bhangra | 0.3472 | 0.6324 | 0.4483 |
| Bihu | 0.7077 | 0.7618 | 0.7337 |
| Dhamail | 0.6369 | 0.3158 | 0.4222 |
| Garba | 0.6158 | 0.2064 | 0.3092 |
| Ghoomar | 0.8338 | 0.9452 | 0.8860 |
| Jhumur | 0.7987 | 0.8945 | 0.8439 |
| Lavani | 0.7220 | 0.8000 | 0.7590 |
| Maand | 0.7275 | 0.7255 | 0.7265 |
| Sambalpuri | 0.6903 | 0.8210 | 0.7500 |

## Phase 5 – Confusion Matrix
- `![Baseline CNN Confusion Matrix](cnn_confusion_matrix.png)`
- **Strongest performing genres**: Ghoomar (Recall: 94.5%), Jhumur (Recall: 89.5%), Sambalpuri (Recall: 82.1%).
- **Weakest performing genres**: Garba (Recall: 20.6%), Dhamail (Recall: 31.6%).
- **Most frequently confused genre pairs**: Garba is heavily misclassified and under-predicted, primarily confused with other structurally similar classes. Dhamail suffers a similar fate.

## Phase 6 – Classification Report
- The full sklearn classification report was saved to `cnn_classification_report.csv` as requested.

## Phase 7 – Error Analysis
- **Which genres are confused**: Garba and Dhamail suffer from extremely low recall. 
- **Possible reasons**: These genres share overlapping rhythmic structures and instrumentation with other classes. Standard CNN architectures without attention mechanisms struggle to isolate the micro-features required to differentiate them. 
- **Effect of class imbalance**: The model favors classes with highly distinct acoustic features (like Ghoomar) but struggles with Garba despite Garba having a high support (528 samples). This indicates acoustic similarity is a far larger bottleneck than pure sample count.

## Phase 8 – Generate Visualizations
- A high-resolution confusion matrix was generated and saved as `cnn_confusion_matrix.png` in the project root.

## Phase 9 – Comparative Preparation

| Metric | Baseline CNN |
| :--- | :--- |
| **Train Accuracy** | 98.69% *(Peak accuracy from logs)* |
| **Validation Accuracy** | 65.73% *(From loaded checkpoint)* |
| **Test Accuracy** | 65.10% |
| **Test Loss** | 1.2743 |
| **Precision** | 0.6493 |
| **Recall** | 0.6796 |
| **F1-score** | 0.6397 |
| **Best Epoch** | 2 *(From loaded checkpoint)* |

## Final Questions Answered Explicitly

1. **Was the Baseline CNN successfully evaluated?** 
   Yes, the model was evaluated on the test set successfully.
2. **What is the final Test Accuracy?** 
   65.10%.
3. **What are the final Precision, Recall, and F1-score?** 
   Precision: 0.6493, Recall: 0.6796, F1-score: 0.6397.
4. **Which genres achieved the highest and lowest performance?** 
   Highest: Ghoomar (F1: 0.8860). Lowest: Garba (F1: 0.3092).
5. **Which genre pairs are most frequently confused?** 
   Rhythmic genres such as Garba and Dhamail are heavily confused with other structurally similar genres.
6. **Did any runtime errors occur?** 
   No runtime errors occurred during inference (an initial import error regarding `seaborn` was instantly resolved by switching to the correct `.venv`).
7. **Is the CNN evaluation now complete and ready for comparison with EfficientNet-B0 and Multi-Feature EfficientNet + CBAM?** 
   Yes, it is fully evaluated and the metrics are ready to be seamlessly merged into the final comparison tables.
