# Repository Structure

```text
Folk_Music_Classifier/
├── .venv/                      # Python virtual environment
├── archive/
│   └── development_tools/      # Preserved development scripts and debug files
├── checkpoints/                # Original baseline models
├── datasets/                   # Audio files, segmented data, and extracted features
├── docs/                       # Project documentation
├── models/                     # PyTorch model definitions (EfficientNet, CBAM)
├── notebooks/                  # Jupyter notebooks for data exploration
├── outputs/                    # Raw training logs, intermediate models, and experiments
├── results/                    # Final packaged research deliverables
│   ├── baseline/               # Baseline model weights
│   ├── cbam/                   # Final CBAM model weights
│   ├── figures/                # Publication-quality plots (Accuracy, Loss, CM, Bar Charts)
│   ├── logs/                   # Final aggregated training logs
│   ├── metrics/                # JSON evaluation reports
│   └── reports/                # Final research, statistical, and engineering reports
├── scripts/                    # Data preparation, training, and evaluation pipeline
├── utils/                      # Configuration and dataloader utilities
├── requirements.txt            # Python dependencies
└── README.md                   # Project landing page
```
