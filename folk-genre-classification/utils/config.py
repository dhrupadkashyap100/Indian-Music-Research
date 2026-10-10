"""
=========================================================
Project Configuration

Project:
AI-Based Analysis of Indian Traditional Folk Music

Author:
Dhrupad Kashyap
=========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATASET DIRECTORIES
# ==========================================================

DATASET_DIR = PROJECT_ROOT / "datasets"

RAW_AUDIO_DIR = DATASET_DIR / "raw_audio"

PROCESSED_AUDIO_DIR = DATASET_DIR / "processed"

SEGMENTED_AUDIO_DIR = DATASET_DIR / "segmented"

AUGMENTED_DIR = DATASET_DIR / "augmented"

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"

METADATA_DIR = DATASET_DIR / "metadata"

FEATURES_DIR = DATASET_DIR / "features"

# ==========================================================
# FEATURE DIRECTORIES
# ==========================================================

CHROMAGRAM_DIR = FEATURES_DIR / "chromagrams"

MELSPECTROGRAM_DIR = FEATURES_DIR / "melspectrograms"

MFCC_DIR = FEATURES_DIR / "mfcc"

SWARAGRAM_DIR = FEATURES_DIR / "swaragrams"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = OUTPUT_DIR / "logs"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"

# ==========================================================
# LOG FILES
# ==========================================================

STANDARDIZATION_LOG = LOG_DIR / "standardization_log.csv"

SEGMENTATION_LOG = LOG_DIR / "segmentation_log.csv"

FEATURE_EXTRACTION_LOG = LOG_DIR / "feature_extraction_log.csv"

MELSPECTROGRAM_LOG = LOG_DIR / "melspectrogram_log.csv"

CHROMA_LOG = LOG_DIR / "chromagram_log.csv"

TRAINING_LOG = LOG_DIR / "training_log.csv"

# ==========================================================
# AUDIO SETTINGS
# ==========================================================

TARGET_SAMPLE_RATE = 22050

SAMPLE_RATE = TARGET_SAMPLE_RATE

TARGET_CHANNELS = 1

TARGET_SUBTYPE = "PCM_16"

TARGET_EXTENSION = ".wav"

SUPPORTED_AUDIO_FORMATS = [

    ".wav",
    ".WAV",

    ".mp3",
    ".MP3",

    ".flac",
    ".FLAC",

    ".ogg",
    ".OGG",

    ".m4a",
    ".M4A"

]

# ==========================================================
# AUDIO VALIDATION
# ==========================================================

MIN_AUDIO_DURATION = 1.0

MAX_AUDIO_DURATION = 1800.0

NORMALIZATION_PEAK = 0.99

SILENCE_THRESHOLD = 1e-4

CLIPPING_THRESHOLD = 0.999

# ==========================================================
# AUDIO SEGMENTATION
# ==========================================================

SEGMENT_DURATION = 10

SEGMENT_OVERLAP = 5

MIN_SEGMENT_DURATION = 8

INTRO_SKIP_PERCENT = 0.10

MAX_INTRO_SKIP = 30

# ==========================================================
# FEATURE EXTRACTION SETTINGS
# ==========================================================

N_FFT = 2048

HOP_LENGTH = 512

WIN_LENGTH = 2048

N_MELS = 128

N_MFCC = 40

FMIN = 20

FMAX = TARGET_SAMPLE_RATE // 2

TARGET_FRAMES = 938

FEATURE_DTYPE = "float32"

# ==========================================================
# MODEL INPUT SHAPE
# ==========================================================

NUM_INPUT_CHANNELS = 1

INPUT_HEIGHT = N_MELS

INPUT_WIDTH = TARGET_FRAMES

# ==========================================================
# DATASET SPLIT
# ==========================================================

TRAIN_SPLIT = 0.70

VALIDATION_SPLIT = 0.15

TEST_SPLIT = 0.15

RANDOM_SEED = 42

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

ENABLE_AUGMENTATION = True

AUGMENTATIONS_PER_SAMPLE = 2

PITCH_SHIFT_RANGE = (-2, 2)

TIME_STRETCH_RANGE = (0.90, 1.10)

NOISE_FACTOR = 0.003

GAIN_RANGE = (0.80, 1.20)

# ==========================================================
# DATASET GENRES
# ==========================================================

GENRES = [

    "Baul",

    "Bhangra",

    "Bihu",

    "Dhamail",

    "Garba",

    "Ghoomar",

    "Jhumur",

    "Lavani",

    "Maand",

    "Sambalpuri"

]

# ==========================================================
# MODEL SETTINGS
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

NUM_CLASSES = len(GENRES)

MODEL_NAME = "genre_classifier_cnn.pth"

BEST_MODEL_NAME = "best_genre_classifier.pth"

# ==========================================================
# DATALOADER SETTINGS
# ==========================================================

NUM_WORKERS = 4

PIN_MEMORY = True

# ==========================================================
# DISPLAY SETTINGS
# ==========================================================

LINE_SEPARATOR = "=" * 60

SMALL_SEPARATOR = "-" * 60

# ==========================================================
# FILE NAMES
# ==========================================================

MASTER_DATASET_FILE = "master_dataset.csv"

SONG_GROUP_FILE = "song_groups.csv"

TRAIN_FILE = "train.csv"

VALIDATION_FILE = "validation.csv"

TEST_FILE = "test.csv"

AUGMENTED_TRAIN_FILE = "augmented_train.csv"

MEL_METADATA_FILE = "mel_metadata.csv"

CHROMA_METADATA_FILE = "chroma_metadata.csv"

# ==========================================================
# GENRE CLASSIFIER FILES
# ==========================================================

GENRE_MAPPING_FILE = "genre_mapping.json"

GENRE_LABELS_FILE = "genre_labels.csv"

TRAIN_STATS_FILE = "train_statistics.json"

GENRE_DATASET_SUMMARY = "genre_dataset_summary.txt"

# ==========================================================
# CREATE IMPORTANT DIRECTORIES
# ==========================================================

IMPORTANT_DIRECTORIES = [

    RAW_AUDIO_DIR,

    PROCESSED_AUDIO_DIR,

    SEGMENTED_AUDIO_DIR,

    AUGMENTED_DIR,

    TRAIN_DIR,

    VALIDATION_DIR,

    TEST_DIR,

    METADATA_DIR,

    FEATURES_DIR,

    CHROMAGRAM_DIR,

    MELSPECTROGRAM_DIR,

    MFCC_DIR,

    SWARAGRAM_DIR,

    OUTPUT_DIR,

    LOG_DIR,

    MODEL_DIR,

    REPORT_DIR

]

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

for directory in IMPORTANT_DIRECTORIES:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )

# ==========================================================
# EFFICIENTNET CONFIGURATION
# ==========================================================

EFFICIENTNET_MODEL_NAME = "efficientnet_classifier.pth"

BEST_EFFICIENTNET_MODEL_NAME = "best_efficientnet_classifier.pth"

EFFICIENTNET_TRAINING_LOG = LOG_DIR / "efficientnet_training_log.csv"

EFFICIENTNET_TRAIN_STATS_FILE = "efficientnet_train_statistics.json"

FREEZE_EPOCHS = 10

BACKBONE_LR = 1e-4

CLASSIFIER_LR = 1e-3

UNFREEZE_BLOCKS = 2

# ==========================================================
# EFFICIENTNET GENERALIZATION SETTINGS
# ==========================================================

LABEL_SMOOTHING = 0.1

ENABLE_SPECAUGMENT = True

TIME_MASK_PROBABILITY = 0.5

TIME_MASK_WIDTH = 20

NUM_TIME_MASKS = 2

FREQUENCY_MASK_PROBABILITY = 0.5

FREQUENCY_MASK_WIDTH = 10

NUM_FREQUENCY_MASKS = 2

ENABLE_TTA = True

NUM_TTA_PASSES = 3

TTA_TIME_SHIFT = 10

# ==========================================================
# PROJECT INFORMATION
# ==========================================================


PROJECT_NAME = "AI-Based Analysis of Indian Traditional Folk Music"

AUTHOR = "Dhrupad Kashyap"

VERSION = "1.0"

print(f"{PROJECT_NAME} Configuration Loaded")