# Chromagram Pipeline Documentation

## 1. Overview
This document details the newly added Constant-Q Transform (CQT) Chromagram pipeline for the "AI-Based Analysis of Indian Traditional Folk Music" project. The pipeline was explicitly designed as an additive extension—it operates completely independently of the frozen Mel Spectrogram baseline to guarantee strict backward compatibility, while adhering to identical preprocessing paradigms for future Multi-Feature Fusion.

## 2. Rationale: CQT vs STFT
Standard `librosa.feature.chroma_stft()` relies on a linear frequency scale (Short-Time Fourier Transform). For musical applications (and especially Raaga Identification), notes are logarithmically spaced. The Constant-Q Transform (`chroma_cqt`) employs a logarithmic frequency scale, which perfectly aligns with musical pitch classes. This ensures that the energy within each pitch class (C, C#, D, etc.) is preserved with substantially higher precision, providing a stronger harmonic representation of traditional folk music.

## 3. Preprocessing Pipeline

### A. Extraction
- **Algorithm**: `librosa.feature.chroma_cqt()`
- **Sample Rate**: Maintains the original target `TARGET_SAMPLE_RATE` (e.g., 22050 Hz).
- **Hop Length**: Retained identical `HOP_LENGTH` to perfectly align temporal frames across features.
- **n_chroma**: 12 (representing the 12 semi-tones of a chromatic scale).
- **fmin**: `C1` (~32.7 Hz), covering lower fundamentals.

### B. Normalization Strategy
`librosa.feature.chroma_cqt()` inherently outputs normalized values bounded between `[0, 1]`. However, to guarantee absolute mathematical uniformity in our preprocessing philosophy, the generated Chromagrams are subsequently passed through the project's centralized explicit Min-Max normalizer (`normalize_feature`).
*Design Reason*: When performing deep Multi-Feature Fusion in future stages, network activations benefit significantly from identically scaled distributions. Using the exact same scalar function rules out bounded inconsistencies.

### C. Dimensional Alignment
- **Padding/Trimming**: Identical algorithm applied.
- **Output Shape**: `(12, Target_Frames)` (e.g., `(12, 938)`).

## 4. Folder Structure & Storage
To avoid dataset contamination, Chromagrams are strictly separated from existing folders.
- **Root**: `datasets/features/chromagrams/`
- **Sub-hierarchy**: `train/`, `validation/`, `test/` -> `<Genre>/` -> `.npy` (float32).
- **Metadata**: Generated as an independent entity: `datasets/metadata/chroma_metadata.csv`, structured exactly like `mel_metadata.csv`.

## 5. Validation Procedure
A custom automated suite (`scripts/14_validate_and_visualize.py`) enforces structural integrity. It maps `chroma_metadata.csv` over `mel_metadata.csv`, verifying:
1. 1-to-1 filename mirroring.
2. Tensor dimensional correctness (`(12, 938)`).
3. Float32 precision.
4. Total absence of `NaN` or `Inf` artifacts.

## 6. Forward Compatibility (The Roadmap)
The decoupled design of this pipeline lays a highly scalable foundation for:
1. **Multi-Feature Dataset**: PyTorch `Dataset` classes can now be easily modified to read parallel `.npy` paths from the aligned metadata files.
2. **Feature Fusion**: Temporal dimensional symmetry (`938` frames) ensures features can be stacked (`np.concatenate`) or passed through dual-branch feature extractors.
3. **Advanced Architectures**: Safely supports dual-headed Convolutional Recurrent Neural Networks (CRNNs) and CBAM attention models.
