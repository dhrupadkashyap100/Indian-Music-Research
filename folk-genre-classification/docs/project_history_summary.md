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
