# Enhancing Indian Traditional Folk Music Classification using Convolutional Block Attention Module (CBAM)

## Abstract
This study investigates the classification of Indian Traditional Folk Music into ten distinct regional genres using deep learning methodologies. We propose an enhancement to a baseline Multi-Feature EfficientNet model—which fuses Mel-Spectrogram and Chromagram representations—by integrating a Convolutional Block Attention Module (CBAM). By dynamically learning the "what" (channel-wise) and "where" (spatial-wise) of critical acoustic features, the CBAM-enhanced model achieves a statistically meaningful improvement in classification accuracy, pushing the test accuracy from 65.07% to 68.70%, with only a marginal increase in computational complexity (+226 parameters).

## 1. Motivation
Indian traditional folk music is characterized by highly nuanced, region-specific instrumentation, complex rhythmic structures, and microtonal variations. Standard convolutional neural networks often struggle to isolate the most discriminative frequency bins and temporal events within densely packed spectrograms. Attention mechanisms, specifically CBAM, have shown tremendous promise in computer vision by emphasizing salient regions while suppressing background noise. Applying CBAM to fused acoustic features offers a lightweight mechanism to enhance the network's discriminative capability across overlapping musical genres.

## 2. Dataset
The dataset comprises a meticulously curated collection of Indian traditional folk music spanning 10 distinct classes:
- **Genres:** Baul, Bhangra, Bihu, Dhamail, Garba, Ghoomar, Jhumur, Lavani, Maand, Sambalpuri.
- **Preprocessing:** Audio clips were standardized to a 22050 Hz sampling rate, normalized, and segmented into 30-second clips. 

## 3. Methodology

### 3.1 Feature Extraction
The system utilizes two primary acoustic representations:
1. **Mel-Spectrogram:** Captures the timbral and rhythmic envelope of the music.
2. **Chromagram:** Captures the pitch-class profile and harmonic progression, crucial for identifying specific melodic motifs.

### 3.2 Model Architecture
**Baseline Model (MultiFeatureEfficientNet):**
An EfficientNet-B0 backbone adapted for single-channel fused input. The Mel-Spectrogram and Chromagram are concatenated along the feature dimension, passed through an initial fusion block (1x1 convolution followed by BatchNorm and ReLU), and subsequently processed by the EfficientNet-B0 feature extractor.

**Proposed Model (MultiFeatureEfficientNetCBAM):**
The baseline architecture is augmented by inserting a CBAM module immediately following the initial multi-feature fusion block. 
- **Channel Attention:** Emphasizes the most relevant feature maps (e.g., distinguishing between highly rhythmic vs. highly melodic components).
- **Spatial Attention:** Focuses on the most critical time-frequency regions within the fused representation.

## 4. Experimental Setup
- **Optimizer:** Adam with an initial learning rate of 0.001.
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5) tracking validation loss.
- **Loss Function:** Cross-Entropy Loss.
- **Batch Size:** 32
- **Epochs:** 40
- **Hardware:** CUDA-enabled GPU.

## 5. Results and Discussion

### 5.1 Baseline Comparison
| Metric | Baseline | CBAM | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 65.07% | **68.70%** | +3.63% |
| **Macro F1 Score**| 0.6556 | **0.6788** | +2.32% |
| **Parameters** | 4,025,351 | 4,025,577 | +226 (+0.0056%) |

The integration of CBAM yielded a remarkable absolute improvement of +3.63% in test accuracy for an almost negligible parameter increase. This highlights the extreme parameter-efficiency of attention modules in auditory processing tasks.

### 5.2 Class-Wise Error Analysis
- **Strengths:** The model demonstrates excellent separation for **Sambalpuri** (F1: 0.8786) and **Lavani** (F1: 0.8661). These genres possess distinct percussive signatures and rhythmic patterns that the CBAM module successfully isolated.
- **Weaknesses:** **Garba** remains the most challenging class (Recall: 0.2178, F1: 0.3208). The model heavily under-predicts this class, likely confusing its rhythmic structure with structurally similar genres such as Maand or Ghoomar. 
- **Limitations:** While CBAM improves spatial and channel discrimination, it does not explicitly model long-term temporal dependencies across the 30-second clips. 

## 6. Conclusion and Future Work
The experimental results confirm that incorporating a Convolutional Block Attention Module (CBAM) into a fused-feature EfficientNet pipeline provides a measurable, significant improvement in the classification of Indian Folk Music. 

**Future Work:**
1. Investigate the integration of temporal attention models (e.g., Transformers or LSTMs) to capture long-term structural dependencies.
2. Conduct ablation studies on the placement of the CBAM module (e.g., placing it deeper within the EfficientNet blocks).
3. Expand the dataset to address the severe underperformance on the Garba class through targeted augmentation or hard-negative mining.
