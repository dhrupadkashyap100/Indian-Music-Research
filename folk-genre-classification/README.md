# AI-Based Analysis of Indian Traditional Folk Music

This repository contains the codebase and experiments for classifying Indian Traditional Folk Music into 10 distinct regional genres using deep learning. The project explores multi-feature fusion (Mel-Spectrograms and Chromagrams) combined with state-of-the-art Convolutional Neural Networks (EfficientNet) and Attention Mechanisms (CBAM).

## Project Overview
Indian folk music is characterized by unique regional instruments, complex rhythms, and microtonal nuances. This project aims to accurately classify 10 genres: **Baul, Bhangra, Bihu, Dhamail, Garba, Ghoomar, Jhumur, Lavani, Maand, and Sambalpuri**. 

We present a baseline **Multi-Feature EfficientNet** model and a proposed **CBAM-Enhanced** variant. The integration of the Convolutional Block Attention Module (CBAM) significantly improves classification accuracy by dynamically attending to critical spatial (time-frequency) and channel-wise features.

## Architecture
1. **Input:** Raw audio standardized to 22050 Hz and segmented into 30-second clips.
2. **Feature Extraction:** Mel-Spectrogram (timbre/rhythm) and Chromagram (pitch-class) extraction.
3. **Fusion:** Early concatenation along the channel dimension followed by a learned 1x1 convolution block.
4. **Attention (Proposed):** CBAM module applied to the fused representation.
5. **Backbone:** EfficientNet-B0 feature extractor.
6. **Classification:** Fully connected head outputting probabilities over 10 classes.

## Experimental Results
The CBAM architecture provided a measurable, meaningful improvement over the baseline model with a nearly negligible increase in parameter count (+226 parameters).

| Metric | Baseline | CBAM Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 65.07% | **68.70%** | +3.63% |
| **Macro F1 Score**| 0.6556 | **0.6788** | +2.32% |
| **Parameters** | 4,025,351| **4,025,577**| +0.0056% |

## Directory Structure
```text
Folk_Music_Classifier/
│
├── archive/
│   └── development_tools/    # Preserved scratch files, verification scripts, and temporary tools
│
├── checkpoints/              # Baseline models
├── datasets/                 # Audio files, segmented data, and extracted features
├── docs/                     # Additional project documentation
├── models/                   # PyTorch model definitions (EfficientNet, CBAM)
├── notebooks/                # Jupyter notebooks for data exploration
├── outputs/                  # Raw training logs, intermediate models, and cbam experiments
├── results/                  # 🌟 Final packaged research deliverables
│   ├── baseline/             # Baseline model weights
│   ├── cbam/                 # Final CBAM model weights
│   ├── figures/              # Publication-quality plots (Accuracy, Loss, CM, Bar Charts)
│   ├── logs/                 # Final aggregated training logs
│   ├── metrics/              # JSON evaluation reports
│   └── reports/              # Final research, statistical, and engineering reports
│
├── scripts/                  # Data preparation, training, and evaluation pipeline
├── utils/                    # Configuration and dataloader utilities
└── README.md                 # Project landing page
```

## Installation & Requirements
The project requires Python 3.8+ and CUDA for GPU acceleration.
```bash
# Clone the repository
git clone https://github.com/your-username/folk_music_classifier.git
cd folk_music_classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*(Ensure PyTorch and Torchaudio are installed with the appropriate CUDA version)*

## Training Instructions
To reproduce the training process from scratch, execute the scripts sequentially:

1. **Prepare Data:**
```bash
python scripts/02_standardize_audio.py
python scripts/03_segment_audio.py
python scripts/06_split_dataset.py
python scripts/08_extract_chromagram.py
python scripts/09_extract_melspectrogram.py
```
2. **Train Baseline Model:**
```bash
python scripts/17_train_multi_feature_classifier.py
```
3. **Train Proposed CBAM Model:**
```bash
python scripts/21_train_cbam_classifier.py
```

## Evaluation Instructions
To reproduce the evaluation results on the test set:
```bash
# Evaluate baseline
python scripts/18_evaluate_multi_feature.py

# Evaluate CBAM
python scripts/22_evaluate_cbam.py
```

All final results, metrics, and figures will be saved in the `results/` directory.

## License
MIT License. See `LICENSE` for more information.

## Citation
If you use this codebase in your research, please cite:
```bibtex
@misc{folk_music_cbam_2026,
  title={Enhancing Indian Traditional Folk Music Classification using Convolutional Block Attention Module},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
}
```
