# Indian Music Research

This repository unifies two related research projects on Indian music classification:

1. **[Folk Genre Classification](./folk-genre-classification/README.md):** Classification of Indian traditional folk songs into 10 respective folk genres using multi-feature extraction (Log-Mel, CQT Chromagram, Swaragram).
2. **[Raag Classification (CultureMERT)](./raag-classification/README.md):** Using Hindustani Raag classification to assign probable Hindustani Raag associations to Indian traditional folk music via CultureMERT embeddings.

## Repository Structure

```
Indian-Music-Research/
├── folk-genre-classification/  # Project 1: Folk genre CNN/EfficientNet classifiers
├── raag-classification/        # Project 2: Raag classifier via CultureMERT
├── README.md                   # This file
├── DATA_PROVENANCE.md          # Description of datasets and generated features
└── LICENSE                     # Project License
```

## Note on Audio Files
To comply with distribution rights and licensing constraints, **no audio files** are included in this repository. Researchers must provide their own audio files or obtain authorized access to the datasets mentioned in the respective project directories. The feature extraction scripts provided can then be used to recreate the feature files needed for training and inference.
