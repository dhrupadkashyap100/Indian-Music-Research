# Comprehensive Project Documentation

This document is a combined compilation of the Project History, Dataset Architecture, and Codebase Details for the Folk Music Classifier.

---

# Project History Summary: AI-Based Analysis of Indian Traditional Folk Music

This document provides a highly detailed, comprehensive breakdown of every single step you have accomplished in the `Folk_Music_Classifier` project from day one until today. It covers data ingestion, processing pipelines, utility architectures, and model training in precise detail.

## 1. Project Initialization & Architecture Setup
You set up a robust, scalable project architecture designed specifically for machine learning pipelines.
* **Directory Structure:** Created specialized directories for datasets (`raw_audio`, `segmented`, `augmented`, `processed`, `metadata`), outputs (`logs`, `models`, `reports`), scripts, and modular utilities.
* **Core Configuration:** Built `utils/config.py` to centralize paths, constants, and hyperparameters, ensuring you don't hardcode values across multiple scripts.
* **Custom Logger:** Developed `utils/logger.py` (`ProjectLogger`) to implement dual-stream logging (console output + file storage), ensuring every script execution leaves a traceable footprint.

## 2. Dataset Verification (`01_verify_dataset.py`)
* **Metadata Intake:** You ingested 10 distinct regional folk music metadata `.xlsx` files: `Baul`, `Bhangra`, `Bihu`, `Dhamail`, `Garba`, `Ghoomar`, `Jhumur`, `Lavani`, `Maand`, and `Sambalpuri`.
* **Integrity Checks:** Wrote logic to validate the presence, integrity, and basic structure of the raw audio dataset against the metadata records, using `utils/metadata_utils.py` and `utils/file_utils.py`.

## 3. Audio Standardization (`02_standardize_audio.py`)
You ensured all audio files follow a strict uniform format, which is critical for reliable feature extraction.
* **Signal Processing Utilities:** Built `utils/audio_utils.py` with reusable functions like `resample_audio`, `convert_to_mono`, `normalize_audio`, `get_duration`, and clipping/NaN checks.
* **Standardization Pipeline:** You loaded raw audio, resampled it to a standard sampling rate, converted all tracks to mono channels, and normalized the amplitudes.
* **Logging:** Tracked the entire operation step-by-step in `outputs/logs/standardization_log.csv`.

## 4. Audio Segmentation (`03_segment_audio.py`)
To increase your dataset size and provide fixed-length inputs to the Convolutional Neural Network (CNN), you sliced the standardized audio into smaller chunks.
* **Slicing Logic:** Extracted sequential segments (e.g., 3-5 seconds long) from the full-length songs.
* **Output Storage:** Saved these short chunks systematically in `datasets/segmented/` under their respective genre folders.
* **Metadata Tracking:** Built `datasets/metadata/segment_metadata.csv` mapping every segment back to its parent song, and logged operations to `outputs/logs/segmentation_log.csv`.

## 5. Master Dataset & Song Grouping (`04_build_master_dataset.py` & `05_group_by_song.py`)
You prevented a critical machine learning flaw (Data Leakage) where segments from the same song appear in both training and testing sets.
* **Master Compilation:** Merged all regional data and segment info into one massive `master_dataset.csv` (~7.3 MB) and generated a textual report (`master_dataset_report.txt`).
* **Group Management:** Wrote logic to explicitly group segments by their original parent song ID, saving this grouping logic to `song_groups.csv` and validating it via `song_group_report.txt`.

## 6. Dataset Splitting (`06_split_dataset.py`)
* **Strict Splitting:** Using the song groups, you split the dataset safely into three dedicated sets: Training, Validation, and Testing.
* **Metadata Outputs:** Generated independent CSV files (`train.csv`, `validation.csv`, `test.csv`) to manage the split seamlessly.
* **Split Report:** Generated `dataset_split_report.txt` to keep track of the distribution of genres across the sets.

## 7. Data Augmentation (`07_augment_training_data.py`)
To make your final model highly robust and prevent overfitting, you synthetically multiplied your training dataset.
* **Augmentation Techniques:** Using your `utils` library, you implemented functions to inject `noise`, shift `pitch`, `time_stretch` the audio, and alter the `gain`.
* **Targeted Application:** You applied this *only* to the training set (leaving validation/test data pure) and saved the thousands of new files into `datasets/augmented/`.
* **Metadata Extension:** Recorded these synthetic files in an expansive `augmented_train.csv` file (~15 MB).

## 8. Feature Extraction: Mel Spectrograms (`09_extract_melspectrogram.py`)
You transitioned the project from raw audio waveforms into visual representations (spectrograms) suitable for CNNs.
* **Feature Utils:** Expanded `utils/features_utils.py` to handle the extraction math.
* **Mel Spectrogram Pipeline:** Extracted log-mel spectrograms from both original and augmented segments.
* **Normalization & Padding:** Ensured every spectrogram had uniform dimensions by applying rigorous padding/trimming algorithms and normalizing the numerical feature values.
* **NumPy Storage:** Saved the massive numerical matrices as `.npy` arrays inside `datasets/features/` so they wouldn't have to be recomputed during training.
* **Tracking:** Generated `mel_metadata.csv` (~16.5 MB) mapping arrays to labels, and logged extraction times and paths in `outputs/logs/melspectrogram_log.csv`.
* *(Note: You also set up placeholders like `08_extract_chromagram.py`, `10_extract_mfcc.py`, and `11_extract_swaragram.py` indicating your future roadmap).*

## 9. PyTorch Data Preparation (`12_prepare_genre_data.py` & `utils/dataset.py`)
You bridged the gap between your hard drive files and PyTorch's Deep Learning engine.
* **Custom PyTorch Dataset:** Designed a robust `GenreDataset` class inside `utils/dataset.py` that efficiently loads the `.npy` mel spectrogram arrays into memory only when needed.
* **Deterministic Mappings:** Created a consistent, unchanging mapping between the 10 folk genre text labels and integer class IDs (0-9).
* **DataLoaders:** Instantiated PyTorch `DataLoaders` with batching logic, ready to feed tensors directly into the neural network.

## 10. Training the Genre Classifier CNN (`13_train_genre_classifier.py`)
You successfully engineered and trained the primary Deep Learning architecture.
* **CNN Architecture:** Implemented a custom 4-block 2D Convolutional Neural Network (`GenreClassifierCNN`). You included Batch Normalization for stability and `AdaptiveAvgPool2d` to handle dimensionality safely.
* **Hardware Acceleration:** Programmed the script to automatically detect and utilize CUDA/GPU if available, falling back to CPU otherwise.
* **Deterministic Training:** Forced PyTorch, NumPy, and Python's `random` to use fixed seeds, ensuring identical, reproducible results every time the model trains.
* **Advanced Training Loops:**
  - Designed full `train_one_epoch` and `validate_epoch` loops with Loss and Accuracy metrics.
  - Implemented `ReduceLROnPlateau` to automatically lower the learning rate when validation accuracy stalled.
  - Added **Early Stopping** to halt training and save computing resources once the model reached peak performance.
* **Checkpoint & Logging Systems:**
  - Dynamically saved the best model state-dictionaries to `outputs/models/` and `checkpoints/models/`.
  - Logged epoch-by-epoch loss metrics into `outputs/logs/training_log.csv`.
  - Saved final training metrics and hyperparameter metadata to `train_statistics.json`.

## Summary of the Current State
As of today, you have successfully built an end-to-end Machine Learning pipeline. You've taken gigabytes of raw, messy folk music audio, standardized it, sliced it, augmented it, extracted neural network features from it, safely split it, and built a fully functional PyTorch Convolutional Neural Network that actively trains on it. The roadmap ahead (`14` through `20`) is primed for model evaluation, Raaga recognition extensions, and music generation.


---

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


---

# Codebase Details: Folk Music Classifier

This document provides a highly detailed, file-by-file technical breakdown of every script and utility module in the `Folk_Music_Classifier` project. It describes what each file does at a high level, followed by the specific responsibilities of every function inside it.

---

## Part 1: Pipeline Scripts (`scripts/`)

These numbered scripts form the execution pipeline, running sequentially to transform raw data into a trained model.

### `01_verify_dataset.py`
**Purpose:** Validates the presence and integrity of raw audio files against their respective `.xlsx` metadata records.
- Ensures that every row in the raw metadata actually corresponds to an existing audio file on disk before any processing begins.

### `02_standardize_audio.py`
**Purpose:** Converts all raw audio files to a uniform format: WAV, Mono channel, 22050 Hz sample rate, and 16-bit PCM.
- `process_genre(genre)`: Iterates over a specific genre's raw audio directory, running the standardization pipeline on each file and logging the result.
- `main()`: Orchestrates the process across all 10 folk genres, writing outputs to `datasets/processed/`.

### `03_segment_audio.py`
**Purpose:** Slices the standardized audio files into overlapping fixed-length segments (e.g., 10 seconds with a 5-second hop) while skipping initial instrumental intros.
- `load_audio(audio_path)`: Reads the WAV file into a NumPy array.
- `validate_audio(...)`: Runs 7 strict checks (mono, length, NaN, silence) to guarantee audio integrity.
- `calculate_intro_skip(duration)`: Computes how many seconds to skip at the beginning of the song to avoid long musical intros.
- `segment_audio(...)`: Slices the audio array into segments.
- `save_segment(...)`: Writes the segment array back to disk as a WAV file.
- `create_metadata(...)`: Builds a dictionary mapping the segment to its original parent song.
- `process_song(...)`: Handles the full loading, segmenting, and metadata creation lifecycle for one song.
- `process_genre(...)`: Delegates song processing for all files in a genre.
- `process_dataset()` & `main()`: Executes the pipeline globally and saves `segment_metadata.csv` and `segmentation_log.csv`.

### `04_build_master_dataset.py`
**Purpose:** Merges the segment metadata with the original Excel metadata to create a unified `master_dataset.csv`.
- `load_segment_metadata()`: Reads segment data, normalizing the merge keys to handle extensions cleanly.
- `_normalise_genre_df(df)`: Standardizes column names (e.g., handling variants of "YouTube Link" or "Song Name") per genre.
- `load_genre_metadata(genre)` / `load_all_metadata()`: Loads and concatenates the cleaned Excel sheets.
- `merge_metadata()`: Left-joins the segment data with the cleaned Excel metadata on the normalized song identifier.
- `validate_master_dataset(master)`: Checks for nulls or duplicate IDs in the final master dataset.

### `05_group_by_song.py`
**Purpose:** Groups audio segments by their parent song to prevent data leakage between training and testing sets.
- `group_songs(data)`: Aggregates segment records based on `Original Song ID`.
- `save_song_groups(grouped)`: Outputs a clean `song_groups.csv` used directly by the dataset splitter.

### `06_split_dataset.py`
**Purpose:** Splits the dataset into Training, Validation, and Test sets based on the song groups.
- `split_genre(song_df)`: Applies a Train/Val/Test ratio cleanly within a single genre to maintain class balance.
- `create_song_splits(song_groups_df)`: Maps the splits across all songs.
- `map_segments(master_df, song_split_df)`: Assigns every individual segment to the split assigned to its parent song.
- `save_datasets(...)`: Outputs `train.csv`, `validation.csv`, and `test.csv`.

### `07_augment_training_data.py`
**Purpose:** Synthetically expands the training dataset to prevent model overfitting.
- `add_noise(audio)` / `pitch_shift(audio)` / `time_stretch(audio)` / `gain(audio)`: Applying distinct audio transformations using the `librosa` backend.
- `augment_dataset(df)`: Iterates through training rows, creating two unique augmented variations for each file.
- `save_metadata(...)`: Generates `augmented_train.csv`, mapping all original and synthetic segments.

### `09_extract_melspectrogram.py`
**Purpose:** Converts audio segments into Log-Mel Spectrogram arrays (`.npy`), serving as visual features for the CNN.
- `select_training_metadata()`: Chooses `augmented_train.csv` automatically if it exists.
- `generate_mel_spectrogram(audio, sample_rate)`: Computes the power mel spectrogram and converts it to Decibel scale.
- `normalize_feature_values(feature)`: Applies Min-Max normalization to bind feature values between `[0, 1]`.
- `pad_or_trim_feature(feature, target_frames)`: Right-pads zeros or trims the time axis to guarantee uniform tensor dimensions.
- `build_output_path(...)`: Dynamically builds the `.npy` filename using the exact audio stem (crucial for linking augmented files).
- `process_audio(...)`: The core wrapper that loads audio, extracts the spectrogram, normalizes, pads, validates, and saves it.
- `process_dataset(...)`: Loops `process_audio` over all rows in a split.

### `12_prepare_genre_data.py`
**Purpose:** Bridges the saved `.npy` features with PyTorch DataLoaders.
- `build_and_save_genre_mapping(...)`: Creates a deterministic dictionary mapping genre text strings to integer class IDs (0-9) and saves it as JSON.
- `create_dataloaders(...)`: Instantiates the custom PyTorch `GenreDataset` class and returns PyTorch `DataLoader` objects for the train, validation, and test splits.

### `13_train_genre_classifier.py`
**Purpose:** The core Deep Learning script. Builds, trains, validates, and saves a Convolutional Neural Network.
- `GenreClassifierCNN` (Class): Defines a 4-block 2D CNN architecture. Accepts a `(Batch, 1, 128, 938)` input tensor and outputs class logits. Implements Batch Normalization, ReLU, MaxPool2d, and a final `AdaptiveAvgPool2d`.
- `set_seed(seed)`: Forces PyTorch, NumPy, and Python to run deterministically.
- `train_one_epoch(...)`: Runs the forward pass, calculates CrossEntropyLoss, applies backpropagation, and steps the optimizer.
- `validate_epoch(...)`: Runs inference on the validation set without gradient tracking to evaluate true generalization accuracy.
- `save_checkpoint(...)`: Saves the model state dictionary (`.pth`) to `outputs/models/` and `checkpoints/models/` when a new best validation loss is achieved.
- `append_training_log(...)` / `save_training_statistics(...)`: Logs epoch metrics to CSV and final results to JSON.
- `main()`: Controls the training lifecycle, including the `ReduceLROnPlateau` scheduler and Early Stopping.

*(Note: Scripts `08`, `10`, `11`, and `14-20` exist as empty placeholders marking the future project roadmap for evaluating models and generating music).*

---

## Part 2: Utility Library (`utils/`)

These modules provide reusable logic across all the numbered scripts.

### `utils/config.py`
Contains all static project configurations:
- Directory paths (`RAW_AUDIO_DIR`, `PROCESSED_AUDIO_DIR`, etc.)
- Audio hyper-parameters (`TARGET_SAMPLE_RATE`, `N_MELS`, `HOP_LENGTH`, `WIN_LENGTH`)
- Training hyper-parameters.

### `utils/logger.py`
- `ProjectLogger` (Class): A custom logging handler that writes structured text cleanly to both `stdout` (the terminal) and a physical `.txt`/`.csv` file. Exposes functions like `.info()`, `.warning()`, and `.error()`.

### `utils/audio_utils.py`
Contains raw signal processing functions using `librosa` and `soundfile`.
- `load_audio`, `save_audio`, `convert_to_mono`, `resample_audio`, `normalize_audio`.
- Verification functions: `is_silent`, `is_clipped`, `has_nan`, `has_inf`.
- `standardize_audio`: A wrapper that applies all preprocessing steps in sequence.

### `utils/features_utils.py`
Contains extraction and transformation math for ML features.
- `extract_melspectrogram`, `extract_mfcc`, `extract_chromagram`.
- `normalize_feature`: Mathematical Min-Max array normalization.
- `pad_feature`: Tensor padding along the time axis.
- `validate_feature`: Verifies array shapes and ensures no NaNs are passed to the neural network.

### `utils/dataset.py`
- `GenreDataset` (Class): A PyTorch `Dataset` implementation. Its `__getitem__` method loads a `.npy` matrix from the hard drive, converts it to a `torch.Tensor`, maps the genre string to an integer label, and returns `(feature_tensor, label_tensor, feature_name)`.

### `utils/metadata_utils.py` & `utils/file_utils.py`
Handles system I/O and pandas DataFrame logic.
- `load_metadata`, `validate_metadata`, `normalize_filename`.
- `is_audio_file`, `ensure_directory`, `change_extension`.


---



---

## Project Directory Structure

`
Folk_Music_Classifier/
├── checkpoints
│   └── models
│       ├── best_genre_classifier.pth
│       └── genre_classifier_cnn.pth
├── datasets
│   ├── augmented
│   │   ├── Baul
│   │   ├── Bhangra
│   │   ├── Bihu
│   │   ├── Dhamail
│   │   ├── Garba
│   │   ├── Ghoomar
│   │   ├── Jhumur
│   │   ├── Lavani
│   │   ├── Maand
│   │   └── Sambalpuri
│   ├── features
│   │   ├── chromagram
│   │   ├── chromagrams
│   │   ├── mel_spectrogram
│   │   ├── melspectrograms
│   │   │   ├── test
│   │   │   │   ├── Baul
│   │   │   │   ├── Bhangra
│   │   │   │   ├── Bihu
│   │   │   │   ├── Dhamail
│   │   │   │   ├── Garba
│   │   │   │   ├── Ghoomar
│   │   │   │   ├── Jhumur
│   │   │   │   ├── Lavani
│   │   │   │   ├── Maand
│   │   │   │   └── Sambalpuri
│   │   │   ├── train
│   │   │   │   ├── Baul
│   │   │   │   ├── Bhangra
│   │   │   │   ├── Bihu
│   │   │   │   ├── Dhamail
│   │   │   │   ├── Garba
│   │   │   │   ├── Ghoomar
│   │   │   │   ├── Jhumur
│   │   │   │   ├── Lavani
│   │   │   │   ├── Maand
│   │   │   │   └── Sambalpuri
│   │   │   └── validation
│   │   │       ├── Baul
│   │   │       ├── Bhangra
│   │   │       ├── Bihu
│   │   │       ├── Dhamail
│   │   │       ├── Garba
│   │   │       ├── Ghoomar
│   │   │       ├── Jhumur
│   │   │       ├── Lavani
│   │   │       ├── Maand
│   │   │       └── Sambalpuri
│   │   ├── mfcc
│   │   ├── swaragram
│   │   └── swaragrams
│   ├── metadata
│   │   ├── augmented_train.csv
│   │   ├── dataset_split_report.txt
│   │   ├── master_dataset.csv
│   │   ├── master_dataset_report.txt
│   │   ├── mel_metadata.csv
│   │   ├── segment_metadata.csv
│   │   ├── song_group_report.txt
│   │   ├── song_groups.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── processed
│   │   ├── Baul
│   │   ├── Bhangra
│   │   ├── Bihu
│   │   ├── Dhamail
│   │   ├── Garba
│   │   ├── Ghoomar
│   │   ├── Jhumur
│   │   ├── Lavani
│   │   ├── Maand
│   │   └── Sambalpuri
│   ├── raw_audio
│   │   ├── Baul
│   │   ├── Bhangra
│   │   │   ├── Ambran Tu Ayi Hui Hoor Sohniye  The Safri Boyz  Balwinder Safri  full video  OSA Official.mp3
│   │   │   ├── Apna Sangeet-Apna Sangeet Vaje Apna Sangeet (Full Song) HD.mp3
│   │   │   ├── Balwinder Safri - Pao Bhangra 1997 Music Video.mp3
│   │   │   ├── BHAINA NACHIYAAN - OFFICIAL VIDEO - APNA SANGEET (1995).mp3
│   │   │   ├── Bolo Ta Ra Ra (Lyrical) Daler Mehndi  Punjabi Pop Song  Superhit Punjabi Party Song.mp3
│   │   │   ├── Daler Mehndi - Ho Jayegi Ballle Balle  Lyrical Video  Jawahar Wattal  Superhit Punjabi Song.mp3
│   │   │   ├── Daler Mehndi - Saade Dil Te Chhuriyan Chaliyan.mp3
│   │   │   ├── Dardi Rab Rab Kardi ► Daler Mehndi _ Official Music Video _ DRecords.mp3
│   │   │   ├── Dupatta tera satrang da Surjit Bindrakhia  Dupatta.mp3
│   │   │   ├── Golden Star - Kurri Garam Jayee (Version-1) HD.mp3
│   │   │   ├── JATT DI PASAND (HD Video) _ SURJIT BINDRAKHIA _ PUNJABI WEDDING SONGS.mp3
│   │   │   ├── Kinni Sohni.mp3
│   │   │   ├── Kuldeep Manak  Jugni  Audio  Old Punjabi Tunes.mp3
│   │   │   ├── Kuldeep Manak - Kuriyan Ne Paya Bhangra.mp3
│   │   │   ├── Malkit Singh - Chal Hun (Official Music Video)  Revibe  Hindi Songs.mp3
│   │   │   ├── Mera Yaar Vajay Dhol.mp3
│   │   │   ├── Mitran Ne Mundri Te  The Safri Boyz  Balwinder Safri  full video  OSA Official.mp3
│   │   │   ├── MUNDA KI MANGDA.mp3
│   │   │   ├── Na Na Na Na Na Re  Rajkummar Rao, Triptii  Daler Mehndi, Gurdeep M  Vicky Vidya Ka Woh Wala Video.mp3
│   │   │   ├── Nawan Le Liya Truck Tere Yaar Ne.mp3
│   │   │   ├── Panjabi MC feat. Kuldeep Manak - Jodi - Big Day Party (Official Video).mp3
│   │   │   ├── Putt Jattan De - Surinder Shinda  Aman Hayer (2002).mp3
│   │   │   ├── Sanu Tedi Tedi Takdi Tu (Lyrical)  Surjit Bindrakhia  Atul Sharma  Punjabi Hits.mp3
│   │   │   ├── Sariyan Toon Sohni - The Safri Boyz - Balwinder Safri - Best Song  official HD  OSA Worldwide.mp3
│   │   │   ├── Soho Road Utey.mp3
│   │   │   ├── Sucha Soorma - (Balbiro Bhabhi) - Surinder Shinda.mp3
│   │   │   ├── Surinder Shinda  Jeona Morh  Audio Part 1  Old Punjabi Tunes.mp3
│   │   │   ├── Surinder Shinda _ Jeona Morh _ Audio Part 1 _ Old Punjabi Tunes.mp3
│   │   │   ├── Surjit Bindrakhia - Tera Yaar Bolda [Full Song] Surjit Bindrakhia _ Phulkari (feat. The Speacialist).mp3
│   │   │   ├── Tunak Tunak Tun Video _ Daler Mehndi _ Full Song _ Daler Mehndi Music.mp3
│   │   │   ├── Tutak Tutak Tutiya Title Song  Full Video   Malkit Singh, Kanika Kapoor, Sonu Sood.mp3
│   │   │   └── WANG WARGI KUDI (Full Video)  SURJIT BINDRAKHIA  Evergreen Punjabi Songs  T-Series.mp3
│   │   ├── Bihu
│   │   │   ├── Aag Fale Dekhilo Audio Song _ Assamese Song.mp3
│   │   │   ├── Aghunor Potharot.mp3
│   │   │   ├── Amona Dhanoke _ Lyrical Video _ Assamese Bihu Song _ Khagen Mahanta _ NK Production.mp3
│   │   │   ├── Bhupen Hazarika EIJONI KORENO NASONI এইজনী যোৰহাটৰ নাচনী BIHU.mp3
│   │   │   ├── BIHURENU BIRINA _ CHINAKI GEET _ ASSAMESE LYRICAL VIDEO SONG _ BHUPEN HAZARIKA.mp3
│   │   │   ├── BOHAGI 2020 BY RUPANKRITA ALANKRITA _ NEW ASSAMESE BIHU SONG _ ANNANYA __SUMI.mp3
│   │   │   ├── DEUTAR BOREGHAR _ CHINAKI GEET _ ASSAMESE LYRICAL VIDEO SONG _ BHUPEN HAZARIKA.mp3
│   │   │   ├── Dhanxiri Pare Hoi ( ধনশিৰি পাৰে হৈ.) - by Khagen Mahanta.mp3
│   │   │   ├── Dhengkoli _ ধেংকলি _ Navanita Sharma _ Nishita Phukan  _ New Assamese Bihu Song 2025.mp3
│   │   │   ├── Dikhow Noi Earibo Nuwaru - দিখৌ নৈ এৰিব নোৱাৰো - Zubeen Garg - বিহু গীত - New Assamese Song 2021.mp3
│   │   │   ├── Ebela Hale Bau.mp3
│   │   │   ├── GABHORU BIHU __ Deeplina Deka __ XUNARU 2019 __ EXCLUSIVE ASSAMESE bihu SINGLE __ 2019.mp3
│   │   │   ├── Janmoni Xopunote Aheba.. Singing by Zubeen Garg.mp3
│   │   │   ├── KAHILI PUWATE _ JAANMONI 2009 _ ASSAMESE MUSIC VIDEO _ ZUBEEN GARG _ BIHU SONG.mp3
│   │   │   ├── KI SAAI ASO _ CHINAKI GEET _ ASSAMESE LYRICAL VIDEO SONG _ BHUPEN HAZARIKA _ BIHU SONG.mp3
│   │   │   ├── KINU SAWONIRE _ JAANMONI 2008 _ VOL 1 _ ASSAMESE MUSIC VIDEO _ ZUBEEN GARG _ BIHU SONG.mp3
│   │   │   ├── KOKALOT TONGALI _ JAANMONI 2010 _ ASSAMESE VIDEO SONG _ ZUBEEN GARG _ SHYAMANTIKA SHARMA.mp3
│   │   │   ├── Kopou Phool Phulelee.mp3
│   │   │   ├── LAJE MORU MORU _ CHINAKI GEET _ ASSAMESE LYRICAL VIDEO SONG _ BHUPEN HAZARIKA _ BIHU SONG.mp3
│   │   │   ├── MONORE PHULONITE _ JAANMONI 2006 _ ASSAMESE MUSIC VIDEO _ ZUBEEN GARG _ BIHU SONG.mp3
│   │   │   ├── MOROM LOGAI LOGAI _ JAANMONI 2010 _ ASSAMESE VIDEO SONG _ ZUBEEN GARG _ SHYAMANTIKA SHARMA.mp3
│   │   │   ├── NILA NILA AKHORE _ JAANMONI 2007 _ ASSAMESE MUSIC VIDEO _ GOLDEN COLLECTION OF ZUBEEN GARG _ BIHU.mp3
│   │   │   ├── O Dehi Oi.mp3
│   │   │   ├── Oi Nahor Fular Botor Zubeen Garg Best Song #zubeengargbestsong #zubeen_garg #assamesesong.mp3
│   │   │   ├── PHOOL PHULISE _ JAANMONI 2005 _ ASSAMESE MUSIC VIDEO _ GOLDEN COLLECTION OF ZUBEEN GARG _ BIHU.mp3
│   │   │   ├── Podulit Bihu __ Rupankrita Alankrita __ Pritom Nath __ Babon Bornil __ New Assamese Lyrical Video .mp3
│   │   │   ├── Porbator Dhekiya _ Junbai 2 _ Zubeen Garg _ Manas Robin _ Bihu _ Parbator dhe dhekiya _ Evergreen.mp3
│   │   │   ├── ROHARE OI ROHEDOI _ Manas Robin _ Vitali Das _ Assamese Bihu _ Manas Robinor Bihu _ Bihu 2023.mp3
│   │   │   ├── SANGOR SIRA PITHA _ CHINAKI GEET _ ASSAMESE LYRICAL VIDEO SONG _ BHUPEN HAZARIKA _ BIHU SONG.mp3
│   │   │   ├── Senai O Tumake Bisari.mp3
│   │   │   ├── TOI NAGINI _ JAANMONI 2010 _ ASSAMESE VIDEO SONG _ ZUBEEN GARG _ SHYAMANTIKA SHARMA.mp3
│   │   │   ├── Tok Dekhi Mor Gaa _ Abhishruti Bezbaruah _ New Assamese Bihu Song _ Official HD Video.mp3
│   │   │   ├── TUMAKE LOGOT _ JAANMONI 2010 _ ASSAMESE VIDEO SONG _ ZUBEEN GARG _ SHYAMANTIKA SHARMA.mp3
│   │   │   ├── Various Artist - Petore Juruni Bhat.mp3
│   │   │   ├── XUNJONI NUXUDIBA _ JAANMONI 2005 _ ASSAMESE MUSIC VIDEO _ GOLDEN COLLECTION OF ZUBEEN GARG.mp3
│   │   │   ├── Zubeen Garg, Zubeen Garg, Zubeen Garg - Tumi Nasoni __ Assamese Old Bihu Song, Zubeen Garg And Shanta Uzir __.mp3
│   │   │   └── ব'হাগ আহিল বুলি __ Bohag Ahil Buli __ শান্তা উজীৰ __ Shanta Uzir.mp3
│   │   ├── Dhamail
│   │   │   ├── Aj Radhar Kunje_আজ রাধার কুঞ্জে.mp3
│   │   │   ├── Ami Ar Jabona Jole_আর যাবো না জলে.mp3
│   │   │   ├── Amra Kar Barit_আমরা কার বাড়িত আইলাম.mp3
│   │   │   ├── Bajaiaona more soi go_জাগাইওনা মোরে সই গো.mp3
│   │   │   ├── Bashi Bajgo Roiya Roiya_বাঁশি বাজগো রইয়া রইয়া.mp3
│   │   │   ├── Bol Go Bol Sokhi_বল গো বল সখি.mp3
│   │   │   ├── Bondhu Bine Pran Bache_বন্ধু বিনে প্রাণ বাঁচে না.mp3
│   │   │   ├── Bondhu Outto Jaibay Giya_বন্ধু অতও যাইবায় গিয়া.mp3
│   │   │   ├── Channi Raite Aiyo Kala_চান্নি রাইতে আইও কালা.mp3
│   │   │   ├── Deikhe Aailam Taare_দেইখে আইলাম তারে সইগো.mp3
│   │   │   ├── Dhamail Dew Go Chorchori Ma_ধমইল দও গ চড়চড়র ম.mp3
│   │   │   ├── Dhamail ses hoia jay_ধামাইল শেষ হইয়া যায়.mp3
│   │   │   ├── Ita Kita Kare Go Mone_ইতা কিতা করে গো মনে.mp3
│   │   │   ├── Jaiyo Na Re Prano_যাইও না রে প্রানো বন্ধু আমারে বুলিয়া.mp3
│   │   │   ├── Jano Ni Go Sai_জানো নি গো সই.mp3
│   │   │   ├── Jole Giyechilam Shai_জলে গিয়াছিলাম সই.mp3
│   │   │   ├── Joler Ghate_জলের ঘাটে.mp3
│   │   │   ├── Joler Joler Ghate_জলার জলার ঘাটে.mp3
│   │   │   ├── Jugol Milon Hoilo Go_যুগল মিলন হইলো গো.mp3
│   │   │   ├── Koto Adore Adore_কত আদরে আদরে.mp3
│   │   │   ├── Kunjero Majhe_কুঞ্জের মাঝে.mp3
│   │   │   ├── Maadal - Bhalo Kore Bajan Re.mp3
│   │   │   ├── Maadal - Jhiko Jhiko Krodi Re.mp3
│   │   │   ├── Maadal - Tora Lowo Go Jol.mp3
│   │   │   ├── Moner Jwalai Jwole_মনের জ্বালায় জ্বলে মরি.mp3
│   │   │   ├── Nrittangon Sunamganj.mp3
│   │   │   ├── Ochena Ak Rupoboti_অচেনা এক রূপবতী.mp3
│   │   │   ├── Pousali Banerjee - Bolo Go Bolo Go Shokhi.mp3
│   │   │   ├── Shumogo Shumogo Sakhi_শুনোগো শুনোগো সখি.mp3
│   │   │   └── Shyamare Ekhon Aar_শ্যামরে এখন আর কাদিও না.mp3
│   │   ├── Garba
│   │   ├── Ghoomar
│   │   │   ├── Assembled Brass Band - Hatheliyan Re Beech.mp3
│   │   │   ├── Assembled Brass Band - Talariya Magariya Re (Instrumental Version).mp3
│   │   │   ├── Assembled Brass Band - नैणां रा लोभी _  _ Hit Rajasthani Ghoomar Song _ World Record Folk Dance _ Seema Mishra _.mp3
│   │   │   ├── Cham Cham Chamke Chunari _ Rajasthani Folk Song _ Ghoomar 2 _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Ital Peetal Ro _ Hit Rajasthani Folk Song _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Mharo Gorband Nakhralo _ Rajasthani Ghoomar Song _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Rumal Mharo Leta Jaijo Ji Dilda _ Latest Hit Rajasthani Song _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Seema Mishra & Gaurav Jain - Chaumaso.mp3
│   │   │   ├── Seema Mishra & Rakesh Kala - Moriya Achho Bolyo Re.mp3
│   │   │   ├── Seema Mishra - Anndata Leta Ajyoji Ghumer Dar Lanjo.mp3
│   │   │   ├── Seema Mishra - Banna Re Bagan Main Jhula Ghalya.mp3
│   │   │   ├── Seema Mishra - Chirmi.mp3
│   │   │   ├── Seema Mishra - Dal Badal Beech Chamke Ji Tara.mp3
│   │   │   ├── Seema Mishra - Dheere Chhal Re Panihari.mp3
│   │   │   ├── Seema Mishra - Gangaur.mp3
│   │   │   ├── Seema Mishra - Ghoomar.mp3
│   │   │   ├── Seema Mishra - Hichki.mp3
│   │   │   ├── Seema Mishra - Hivde Ro Haar.mp3
│   │   │   ├── Seema Mishra - Jeero.mp3
│   │   │   ├── Seema Mishra - Kangasiyo.mp3
│   │   │   ├── Seema Mishra - Kesariya Ro Pag.mp3
│   │   │   ├── Seema Mishra - Lehariyo.mp3
│   │   │   ├── Seema Mishra - Loongan Ro Batwo.mp3
│   │   │   ├── Seema Mishra - More Bole Re.mp3
│   │   │   ├── Seema Mishra - Panihari (Original Song) _  Rajasthani Folk Song _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Seema Mishra - Podina.mp3
│   │   │   ├── Seema Mishra - Run Jhun Baje Ghooghara.mp3
│   │   │   ├── Seema Mishra - Samdariyo Lahra Leve Sa.mp3
│   │   │   ├── Tute Bajudan Ri Loom _ Rajasthani Ghoomar Song _ Ghoomar Dance Song _ Seema Mishra _ Veena Music.mp3
│   │   │   └── तारां री चूनड़ी _ Taran Ri Chundari _ Rajasthani Ghoomar Song _ Baisa Ra Beera Jaipur Jaijo Ji.mp3
│   │   ├── Jhumur
│   │   │   ├── Aaji Ei Suprabhate.mp3
│   │   │   ├── Aho Rani Koromoti __ Official Full Hd Video __ Amrit Tanti & Jyotika Baruah __ 2023.mp3
│   │   │   ├── BARIRE SIMARE _ ZUBEEN GARG _ MANAS ROBIN _ AJIT CHETIA _ CHAMPA _ TEA TRIBE SONG _  ADIVASI SONG.mp3
│   │   │   ├── Chaliya Chhodi __ New jhumur Song __ Amrit Tanti __Jyotika.mp3
│   │   │   ├── DEEWANA  __ OFFICIAL VIDEO  __ ROMANTIC JHUMUR SONG 2025 __ SHIBANI THAKUR __ JAYANTA TANTI __.mp3
│   │   │   ├── Ekadoshi Ratiya.mp3
│   │   │   ├── Genda Phool _Neel Akash , Micheal Pathor & Jyotika Baruha _ Bikrom Chawrok _ New Jhumoir song.mp3
│   │   │   ├── GOHONA __ A Modern Song 2022-23 __ Amrit Tanti __ Jyotika Baruah __ Amin Tanti __ Bikram Chawrok.mp3
│   │   │   ├── GOYA By NeeL Akash __ New Adivasi Song 2019.mp3
│   │   │   ├── HARIYA __ AMRIT TANTI.mp3
│   │   │   ├── JA JA RE MINI __ JHUMUR SONG __ NEEL AKASH __ CHANDINI 2017 __.mp3
│   │   │   ├── Jhuli jhule _ Assamese Jhumur Song _ Zubeen Garg Adivasi Song _@SpicyAssamMultimediaPvt.Ltd..mp3
│   │   │   ├── Kale Gori Muke.mp3
│   │   │   ├── Kansiya।।New jhumor Song।।Amrit Tanti  ।।Jyotika Boruwah।।Bikram Chawrok।।.mp3
│   │   │   ├── Koli Juge Alai Doyal.mp3
│   │   │   ├── Koromake Akhora Me Priya Ke Dekhi.mp3
│   │   │   ├── Koti khone bole __ Dulal Manki _ Super Hit Old Jhumur Songs __.mp3
│   │   │   ├── Kuilu Kuilu _ Jhumur Songs _ Assam.mp3
│   │   │   ├── LAAL PARI __ BEJNATH TANTI __ JOON JUNUKA  __ JHUMUR SONG 2024.mp3
│   │   │   ├── Nisitei.mp3
│   │   │   ├── O Mor Dhoni Go Karam Ker Ratiya _ New Jhumur Cover Song 2026 _ AK Nagpuri Vibes Official.mp3
│   │   │   ├── Prem Nadi Tire Tire _ Shaan Tanti _ Jyotika Baruah _ New Jhumair 2021 #trending.mp3
│   │   │   ├── Sajanigo.mp3
│   │   │   ├── Sol Sokhi.mp3
│   │   │   ├── SONAMONI 2 __ AMRIT TANTI __ JYOTIKA BARUAH __ BIKROM CHAWROK __ JHUMUR SONG 2024.mp3
│   │   │   ├── The Jhumur Song.mp3
│   │   │   ├── Tohe Gori Nawa Nari _ Jhumur Songs _ Assam.mp3
│   │   │   ├── Tore Bina New  Modern Jhumar  Song 2022 Assam __ Bejnath Tanti & Sujata Kartik.mp3
│   │   │   ├── ঝাটকা _Jhatka _ New Jhumur Folk Song 2025 _ Amrit Tanti & Jyotti Karmakar _ JHUMOR Video Song 2025.mp3
│   │   │   └── মই তো যাবো কাম জাৰি _ Moi to Jabo Kam Jari _ Zubeen Garg Jhumur Song.mp3
│   │   ├── Lavani
│   │   │   ├── Aai Majhya Lagnachi with lyrics  आई मझय लगनच  Usha Mangeshkar  Dada Kondke  Usha.mp3
│   │   │   ├── Apsara Aali Full Song  Natarang  Sonalee Kulkarni, Ajay Atul  Marathi Songs.mp3
│   │   │   ├── Asha Bhosle - Bai Mi Patang Udvit Hote.mp3
│   │   │   ├── Bela Shende - Asa Wajwa Ki.mp3
│   │   │   ├── Chala Jejurila jau  Lavni  Uttara Kelkar  Live performance Navara majha navsacha movei swarankit.mp3
│   │   │   ├── Disla Ga Bai Disla  दसल ग बई दसल  Usha Mangeshkar  Marathi Song  मरठ गण.mp3
│   │   │   ├── Dolyavar Gogal Lava  Bhirkit  Akshada P, Girish K, Tanaji G  Urmila D  Mangesh K  Shail-Pritesh.mp3
│   │   │   ├── Ghotala song  Vaishali Samant   Sagarika Music Marathi.mp3
│   │   │   ├── Hi Natrangi Naar.mp3
│   │   │   ├── Kalidar Kapuri Paan Lyrical  कळदर कपर पन  Sulochana Chavan  Old Marathi Hit Song.mp3
│   │   │   ├── Kashi Mi Jau Mathurechya Bajari  Natarang  Atul Kulkarni  Ajay-Atul.mp3
│   │   │   ├── Kashyala Lavato (Lavani Song) - Poshter Girl  Superhit Marathi Song  Rasika  Amitraj, Bela Shende.mp3
│   │   │   ├── Lyrical  Kasa Kay Patil Bara Hay Ka_  Old Marathi Song with Lyrics  Sulochana Chavan.mp3
│   │   │   ├── Majha Taat Kar Aadhi.mp3
│   │   │   ├── Mala London Foreign Dakhwa.mp3
│   │   │   ├── Naav Gaav Kashala Pusta- Sulochana Chavan, Film Rangalya Ratri Ashaa.mp3
│   │   │   ├── Navi Navi Navari Mee  Pushpa Pagdhare, Sulochana Chavan, Jaywant Kulkarni  Gulchhadi  Ultra Music.mp3
│   │   │   ├── Padaravarti Jartaricha Mor  पदरवरत जरतरच  Sulochana Chavan  Old Marathi Song  मरठ गण.mp3
│   │   │   ├── Raya Chala Ghodyavarti Basu  रय चल घडयवरत बस  Usha Mangeshkar  Songadya  मरठ गण.mp3
│   │   │   ├── Raya Mala Pavsat Neu Naka with lyrics  रय मल पवसत नऊ नक  Pushpa Pagdhare  Songadya.mp3
│   │   │   ├── Reshamachya Renghani [Full Marathi Song] Lavani Maharashtrachi- Vol.1.mp3
│   │   │   ├── Saaman Avjad Majha.mp3
│   │   │   ├── Sola Hazarat.mp3
│   │   │   ├── THEMBA THEMBAN KELAY MALA GAAR  BAYA CHALALI ZHOKAAT - MARATHI DJ MIX  LYRICAL VIDEO.mp3
│   │   │   ├── Tujha Usala Lagal Kolha , Sulochana Chavan (1).mp3
│   │   │   ├── Ugavali Shukrachi Chandani  De Dhakka  Full Song  Aarati Ankalikar  Gauri Vaidya.mp3
│   │   │   ├── Usha Mangeshkar - Pikala Jambhul Todu Naka.mp3
│   │   │   ├── Wajle Ki Bara  Natarang  Amruta Khanvilkar  Ajay-Atul  Lavani Songs.mp3
│   │   │   ├── छबदर छबदर  Pinjra  Sandhya  Super Hit Marathi Movie Song  SuperHit Marathi Movie Song.mp3
│   │   │   ├── नक डळ छन Naki Doli Chhaan Song  रम रम गगरम Ram Ram Gangaram  दद कडक Dada Kondke.mp3
│   │   │   ├── बई मझ करगळ मडल  Bai Mazi Karangali Modali  Asha Bhosle  Marathi Lavani  मरठ गण.mp3
│   │   │   ├── रनमध एकटच डलतय नग (लवण)  Raanamandi Ekatach Dultoy  Lavani Song  Sharyat  Bhargavi.mp3
│   │   │   ├── रय मल छतश कवळन धर (लवण)- सरख पणकर RAYA MALA CHHATISHI KAVLUN DHARA -SUREKHA PUNEKAR.mp3
│   │   │   ├── लवण - झपर मझ ढल  LAAVNI - JHAMPAR MAJHA DHILA  LOK LAAVNI (Marathi) BY SUREKHA PUNEKAR.mp3
│   │   │   └── सळव वरस धकयच  Solava Varis Dhokyacha  Lavani  Sulochana C Jayshree Gadkar, Arun Sarnaik.mp3
│   │   ├── Maand
│   │   │   ├── Allah Jilai Bai - Jawai Balha Lagoji.mp3
│   │   │   ├── Allah Jilai Bai Songs From Rajasthan _ Kesariya Balam Aavo Ni.mp3
│   │   │   ├── Allah Jilai Bai- Moomal.mp3
│   │   │   ├── Apne Desh Ri Aajadi Ra Geet _ ANWAR KHAN _ अपणे देश री आजादी रा गीत _ अनवर खान.mp3
│   │   │   ├── Badila Yahi Rahiyo Saa _ Rajasthani Mand _ Rathodi Geet _ Traditional Folk Song.mp3
│   │   │   ├── Bai Sa Ra Beera.mp3
│   │   │   ├── Kad Janulee Sasre _ Best Song of Rehana Mirza _ Chandra Shekhar Kalla - Rajasthani Song.mp3
│   │   │   ├── Keerthi Sagathia - Chhap Tilak.mp3
│   │   │   ├── Kesariya Balam _ Anwar Khan Manganiyar & Troupe _ Dhaka International FolkFest 2015.mp3
│   │   │   ├── Kesariya Balam _ Vocal_ Anwar Khan Manganiyar,_ Mohan Veena_ Vishwamohan Bhatt - Desert Slide Group.mp3
│   │   │   ├── Kesariya Balam Aao Ni Padharo Mhare Desh _ Padma Shri Anwar Khan Manganiyar & Pt. Vishwa Mohan Bhatt.mp3
│   │   │   ├── Kesariya Balam Aavo_Gavri Devi.mp3
│   │   │   ├── Kesariya Balam Awo Ni Padharo _ Rajasthani Folk Song _ Seema Mishra _ Veena Music.mp3
│   │   │   ├── Kesariya Ro Pag - Ghoomar _ Seema Mishra _ Superhit Rajasthani Dance _ Rajputi Dance.mp3
│   │   │   ├── Le Dekh Nangi Melo _ Rehana Mirza _ Ikram Rajasthani _ Old Rajasthani Song.mp3
│   │   │   ├── Mame Khan - Chaudhary.mp3
│   │   │   ├── Moray Aaj Aaye Saiyan _Allah Jilai Bai, Traditional Dadra _ Indian classical music.mp3
│   │   │   ├── Munshi Khan - Daru Dakha Ro.mp3
│   │   │   ├── Munshi Khan - Mahara Sajan Base Pardesh.mp3
│   │   │   ├── Munshi Khan - Padharo Mhare Mehal.mp3
│   │   │   ├── Munshi Khan - Thane Samjhaye Sakha Sa.mp3
│   │   │   ├── Nand Lal Bhat - Rajwadi Mehfil.mp3
│   │   │   ├── Raitila Rajasthan - Kesariya Balam _ Rajasthani Folk Song  _ Raitila _.mp3
│   │   │   ├── Rehana Mirza - The To Piyo Sadarudi.mp3
│   │   │   ├── Swaroop Khan - BALAM.mp3
│   │   │   ├── Swaroop Khan Sura - Kuraja Mahra Bhanvr Manae Lav.mp3
│   │   │   ├── Taniya Resham Ki  _ तनिया  रेशम की _ Banarasi Babu _ JMD Rajasthani __ Marwadi Geet__Rajwadi Geet.mp3
│   │   │   ├── Ustad Mame Khan _ Raag Maand _ Soulful Rajasthani Folk Performance.mp3
│   │   │   ├── Ustad Mame Khan _ Rajasthani Folk Songs & Raag Maand Bandish.mp3
│   │   │   ├── आलीजा ढोला एकलड़ी  _मांड_ Jamila & Kulsum Bai.mp3
│   │   │   ├── कुर्जा म्हारी भंवर मिलादे __ राजस्थानी सोंग __ Kurja Mhari Bhanwar Milade __ गायक सिकंदर खान बारू.mp3
│   │   │   ├── केसरिया बालम पधारो म्हारे देश __ KESARIYA BALAM __ भूंगर खान.mp3
│   │   │   ├── घूमर __ पद्म श्री अल्लाह जिलाई बाई __ ALLAH JILAI BAI __ GHOOMAR.mp3
│   │   │   ├── दारूड़े रो रोग लागियो __ New Mehfil Song __ Darude Ro Rog Lagiyo __ गायक सिकंदर खान बारू.mp3
│   │   │   ├── मोहे आई ना जग की लाज _ खेताखान फकीरा खान जुगल्बन्धी _ Kheta Fakira _ Fakira Khan Bhadresh _ newsong.mp3
│   │   │   ├── म्हारी सजनी ऐ आज तो बादिला घर आसी !! Rajasthani रज़वाड़ी Mand !! Singer- Nandlal Bhat !! लाईव ननावद.mp3
│   │   │   ├── राजपुती सोंग _ कलाली _ Kalali New Song _ Rajasthani New Song _ लतीफ खांन _ Marwadi Geet.mp3
│   │   │   ├── राजस्थानी ओल्ड मांड लोकगीत। जूरी -थारे  ना मारे राड  ।स्वर -जमीला बाई &कुलसुम बाई। Audio.mp3
│   │   │   └── राजस्थानी मांड ओल्ड लोकगीत _ Mumal _ Jameela Bai, Kulsum Bai _ मूमल _ Rajasthani Mp3 Song _ Marwadi.mp3
│   │   └── Sambalpuri
│   │       ├── A Labanga Lata __ New Sambalpuri Song __ Ruku Suna & Archana Padhi.mp3
│   │       ├── A Mo Dil__ New Sambalpuri Song __ Smith & Sasmita __Mantu Chhuria & Anamika __ Official Full Video.mp3
│   │       ├── Aha Mor Galar Malire Sambalpuri Song!! Jitendra Haripal!! Sambalpuri Song!!.mp3
│   │       ├── Anta Halei Halei __ New Sambalpuri Song __ Ruku Suna __ Baisali __ Archana Padhi __ Full Video __4K.mp3
│   │       ├── Aseema Panda & Mantu Chhuria - Rani Guri.mp3
│   │       ├── Badnam _ Sambalpuri Song _ Prakash Jal _ Sambalpuri Video _ Sad Song _ Prakash Jal New Song.mp3
│   │       ├── CHAHANEWALI _ New Sambalpuri song _ Song by Antara Chakrabarti and Ashish Kumbhar.mp3
│   │       ├── Chandir Paenjhal _ New Sambalpuri Song _ Ruku Suna & Aseema Panda _ Official Full Song 2025.mp3
│   │       ├── Deewana Banalu Rani _ New Sambalpuri Song _ Ruku Suna _ Aseema Panda _ Official Studio Version.mp3
│   │       ├── Dil Bhitre Lekhideli _ Sambalpuri Romantic Song _ Prakash Jal , Ira Mohanty _ Malaya Mishra.mp3
│   │       ├── Gidja Gija Re _ Sambalpuri Song _ Official Studio Version _ Human Sagar,Asima Panda _ Roul Music.mp3
│   │       ├── Hae Go _ Official Full Video _ Mantu Chhuria _ New Sambalpuri Song.mp3
│   │       ├── JHUPA JHUPA BAUL JHUPA _ SAMBALPURI BHAJAN_IRA MOHANTY_SONI CREATION.mp3
│   │       ├── Jitendriya Haripal - Din Gani Gani.mp3
│   │       ├── Jitendriya Haripal - Namo Namo Ago.mp3
│   │       ├── Kulfirani Chocobar _ New Sambalpuri Song _ Full Video _ Harry, Simran _ Ira Mohanty, Ruku Suna.mp3
│   │       ├── Le Le Baula _ Full Video _ Sambalpuri Song _ Nikita Mishra & Akan _ Ruku Suna & Antara Chakrabarty.mp3
│   │       ├── Mantu Chhuria - Aukaat.mp3
│   │       ├── Mantu Chhuria - Rasia _ New Sambalpuri song _ Full Music Video _ Mantu Chhuria _ Aseema Panda _ Priyambada Swain.mp3
│   │       ├── Nabu Ki Sambalpur _ Official Full Video _ Romyanjali ,Manmay _ Human Sagar , Ira Mohanty _ Odia Song.mp3
│   │       ├── New Sambalpuri Song__Tor Lagi Marijimi__Human Sagar__Top Sambalpuri.mp3
│   │       ├── Pardesia Raja Sambalpuri Song __ Joydev Roul & Romyanjali __ Humane Sagar __ Asima Panda _ #pardesia.mp3
│   │       ├── Prakash Jal - Bewafa Mita.mp3
│   │       ├── Prakash Jal - Bheti Aibu Sajani Kalekale.mp3
│   │       ├── Prem Paga _ Ruku Suna & Sipra _ Official Full Video _ Ruku Suna & Arpita Choudhury _Sambalpuri Video.mp3
│   │       ├── Ramagaru __ Asima panda & Prakash jal __ Sambalpuri 4k video song.mp3
│   │       ├── Rangabati Re Rangabati sambalpuri alltime hit song Jitendra Harpal & Krishna Patel.mp3
│   │       ├── Rasa Mandara Mali Kahanra _ New Sambalpuri Song _ Ruku Suna _ Arpita _ Jayshree _ Kamlesh.mp3
│   │       ├── Sacha Lover _ Sambalpuri Hit Song _ Humane Sagar & Antara Chakrabarty _ Raaz Rock & Jayshree.mp3
│   │       ├── SambalPuri Saiyaan _ Official Studio Version _ Odia Dance Song _ Human Sagar _ Asima Panda.mp3
│   │       ├── Sundri Janha _ New Sambalpuri Song _ Humane Sagar & Swagatika _ Jayshree & Snehashis _ Kamalesh _.mp3
│   │       ├── Tui Bewafa Asima Panda & Amar Dash __ Sambalpuri Song.mp3
│   │       ├── Umakant Barik - Feel My Love.mp3
│   │       └── ଲାଲ୍ ଶାଢ଼ୀ _ Lal Sadhi _ Full Video Song _ Humane Sagar _ Pradeep _ Richa Sharma _ Sambalpuri Song.mp3
│   ├── segmented
│   │   ├── Baul
│   │   ├── Bhangra
│   │   ├── Bihu
│   │   ├── Dhamail
│   │   ├── Garba
│   │   ├── Ghoomar
│   │   ├── Jhumur
│   │   ├── Lavani
│   │   ├── Maand
│   │   └── Sambalpuri
│   ├── test
│   ├── train
│   └── validation
├── docs
│   ├── comprehensive_project_documentation.md
│   ├── dataset_architecture_and_pipeline.md
│   ├── project_codebase_details.md
│   └── project_history_summary.md
├── models
├── notebooks
├── outputs
│   ├── logs
│   │   ├── melspectrogram_log.csv
│   │   ├── segmentation_log.csv
│   │   ├── standardization_log.csv
│   │   └── training_log.csv
│   ├── models
│   │   ├── best_genre_classifier.pth
│   │   ├── genre_classifier_cnn.pth
│   │   ├── genre_mapping.json
│   │   └── train_statistics.json
│   └── reports
├── scripts
│   ├── 01_verify_dataset.py
│   ├── 02_standardize_audio.py
│   ├── 03_segment_audio.py
│   ├── 04_build_master_dataset.py
│   ├── 05_group_by_song.py
│   ├── 06_split_dataset.py
│   ├── 07_augment_training_data.py
│   ├── 08_extract_chromagram.py
│   ├── 09_extract_melspectrogram.py
│   ├── 10_extract_mfcc.py
│   ├── 11_extract_swaragram.py
│   ├── 12_prepare_genre_data.py
│   ├── 13_train_genre_classifier.py
│   ├── 14_evaluate_genre_model.py
│   ├── 15_predict_genre.py
│   ├── 16_prepare_raaga_data.py
│   ├── 17_train_raaga_classifier.py
│   ├── 18_evaluate_raaga_model.py
│   ├── 19_predict_raaga.py
│   └── 20_generate_music.py
├── utils
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── config.py
│   ├── dataset.py
│   ├── features_utils.py
│   ├── file_utils.py
│   ├── logger.py
│   └── metadata_utils.py
├── .gitignore
├── codebase_summary.txt
├── scratch_edit.py
└── tree_structure.txt

`
