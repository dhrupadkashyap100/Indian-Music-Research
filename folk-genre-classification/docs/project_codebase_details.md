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
