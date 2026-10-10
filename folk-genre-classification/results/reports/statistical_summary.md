# Statistical Summary

## Dataset
- **Domain:** Indian Traditional Folk Music
- **Number of Classes:** 10 (Baul, Bhangra, Bihu, Dhamail, Garba, Ghoomar, Jhumur, Lavani, Maand, Sambalpuri)

## Training Configuration
- **Model Architecture (Baseline):** MultiFeatureEfficientNet (EfficientNet-B0 + Mel-Spectrogram & Chromagram fusion)
- **Model Architecture (Proposed):** MultiFeatureEfficientNetCBAM (Baseline + Convolutional Block Attention Module)
- **Optimizer:** Adam
- **Learning Rate:** 0.001 (Scheduled with ReduceLROnPlateau)
- **Batch Size:** 32
- **Epochs:** 40
- **Best Epoch (Baseline):** 24
- **Best Epoch (CBAM):** 33

## Model Complexity
- **Baseline Parameters:** 4,025,351
- **CBAM Model Parameters:** 4,025,577
- **Extra Parameters Introduced:** +226
- **Relative Parameter Increase:** +0.0056%

## Performance Metrics (Baseline vs CBAM)

| Metric | Baseline (Epoch 24) | CBAM (Epoch 33) | Absolute Improvement |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | ~67.44% | **70.78%** | **+3.34%** |
| **Validation Loss** | 1.3224 | **1.2751** | **-0.0473** |
| **Test Accuracy** | 65.07% | **68.70%** | **+3.63%** |
| **Test Loss** | N/A | **1.0559** | N/A |
| **Macro Precision** | 0.6579 | **0.6783** | **+2.04%** |
| **Macro Recall** | 0.6847 | **0.7108** | **+2.61%** |
| **Macro F1 Score** | 0.6556 | **0.6788** | **+2.32%** |
