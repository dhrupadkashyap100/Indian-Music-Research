# Dataset Architecture & Preprocessing Pipeline

This document provides an exhaustive, hyperparameter-level breakdown of the entire data lifecycle in the **AI-Based Analysis of Indian Traditional Folk Music** project. It details the exact mathematical rules, transformations, and splits applied to the dataset from raw audio down to the final tensor fed into the Neural Network.

---

## 1. Raw Dataset & Initialization

### The Folk Genres
The project classifies traditional Indian folk music across **10 distinct genres**:
1. Baul
2. Bhangra
3. Bihu
4. Dhamail
5. Garba
6. Ghoomar
7. Jhumur
8. Lavani
9. Maand
10. Sambalpuri

Each genre's source audio is tracked against detailed Excel metadata (e.g., `Baul_metadata.xlsx`), which logs original song IDs, artists, and YouTube links.

---

## 2. Audio Standardization (`02_standardize_audio.py`)

Before any machine learning happens, all disparate raw audio files undergo strict standardization to prevent format-related bugs.

**Standardization Parameters:**
- **Sample Rate:** Forced to `22050 Hz` (Standard for speech/music processing).
- **Channels:** Mixed down to Mono (`1 Channel`).
- **Bit Depth / Format:** 16-bit PCM (`PCM_16`).
- **File Extension:** `.wav`.
- **Amplitude Normalization:** Peak normalized to `0.99` to ensure uniform loudness across all tracks.

---

## 3. Audio Segmentation (`03_segment_audio.py`)

Feeding full 5-minute songs into a CNN is computationally impossible. The songs are sliced into fixed-length windows.

**Segmentation Rules & Math:**
- **Segment Duration:** `10 seconds`.
- **Overlap (Hop Size):** `5 seconds`. (This 50% overlap effectively doubles the dataset size while preserving temporal context across boundaries).
- **Minimum Duration:** `8 seconds`. (Any leftover chunk at the end of a song shorter than 8 seconds is discarded to prevent severe padding artifacts).
- **Intelligent Intro Skip:** 
  - *Rule:* Skips `10%` of the total song duration, capped at a maximum of `30 seconds`.
  - *Purpose:* Traditional Indian folk music often features long instrumental intros before the defining rhythmic/vocal characteristics emerge. Skipping this improves the classifier's accuracy.

---

## 4. Dataset Grouping & Splitting (`05_group_by_song.py` & `06_split_dataset.py`)

If a model trains on segment `A` of a song and tests on segment `B` of the *same* song, it has essentially "memorized" the studio acoustics rather than learned the genre (Data Leakage). 

**The Splitting Architecture:**
- **Song-Level Grouping:** All segments belonging to the same `Original Song ID` are grouped together.
- **Split Ratios:** 
  - `70%` Training
  - `15%` Validation
  - `15%` Testing
- **Execution:** The split happens strictly at the *song* level. If a Baul song is assigned to the Test set, *all* its 10-second segments go to the Test set.
- **Randomization:** Fixed `RANDOM_SEED = 42` guarantees the exact same split occurs every time the code runs.

---

## 5. Data Augmentation (`07_augment_training_data.py`)

To force the CNN to learn robust features rather than memorizing the training data, the training set is artificially expanded.

**Augmentation Rules:**
- **Target:** Applied *only* to the `70%` Training set. Validation and Test sets remain pure.
- **Multiplier:** `2` augmentations per original sample.
- **Transformations Used (Randomly selected per sample):**
  1. **Pitch Shifting:** Shifts pitch randomly between `-2` and `+2` semitones.
  2. **Time Stretching:** Speeds up or slows down the audio by a factor between `0.90x` and `1.10x` (without altering pitch).
  3. **Noise Injection:** Adds white noise at a factor of `0.003`.
  4. **Gain Modification:** Scales volume randomly between `80%` and `120%`.

---

## 6. Feature Extraction (Log-Mel Spectrograms)

The neural network does not "listen" to audio waveforms; it "looks" at images (spectrograms). 

**Spectrogram Hyperparameters (`09_extract_melspectrogram.py`):**
- **FFT Window Size (`N_FFT`):** `2048` samples.
- **Hop Length:** `512` samples (controls the time resolution).
- **Window Length:** `2048` samples.
- **Mel Bands (`N_MELS`):** `128` (controls the frequency resolution/height of the image).
- **Frequency Range:** Minimum `20 Hz` to Maximum `11025 Hz` (Nyquist frequency of 22050 Hz).
- **Decibel Conversion:** The power spectrogram is converted to Log-scale (Decibels) to mimic human auditory perception.

**Tensor Normalization & Dimensionality:**
- **Padding/Trimming:** Every single spectrogram is strictly padded or trimmed to exactly `TARGET_FRAMES = 938`.
- **Value Normalization:** Pixel values are Min-Max scaled to exist between `[0, 1]`.
- **Data Type:** Cast to `float32` to optimize GPU memory.
- **Output:** Saved as `.npy` matrices to `datasets/features/melspectrograms/`.

---

## 7. Model Input specifications

By the time the data reaches the `GenreClassifierCNN` (`13_train_genre_classifier.py`), the physical audio has been entirely abstracted away. 

**The Final Tensor:**
- **Shape:** `(Batch_Size, Channels, Height, Width)`
- **Actual Dimensions:** `(32, 1, 128, 938)`
  - `32` = Batch Size
  - `1` = Single Channel (Grayscale image)
  - `128` = Mel Frequency Bins (Y-axis)
  - `938` = Time Frames (X-axis)

**Training Hyperparameters:**
- **Epochs:** 100 maximum (with Early Stopping).
- **Learning Rate:** 0.001 (with `ReduceLROnPlateau` scheduling).
- **Classes:** 10 Output Logits.
