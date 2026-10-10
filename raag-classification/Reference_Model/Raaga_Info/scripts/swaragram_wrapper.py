import sys
from pathlib import Path

# Add third_party to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SWARAGRAM_DIR = BASE_DIR / "third_party" / "swaragram-master"
if str(SWARAGRAM_DIR) not in sys.path:
    sys.path.append(str(SWARAGRAM_DIR))

# Mock essentia to avoid ImportErrors for getTonic
# The pipeline relies on ground-truth tonic from CSV, so we don't need essentia's getTonic.
from unittest.mock import MagicMock
if 'essentia' not in sys.modules:
    sys.modules['essentia'] = MagicMock()
    sys.modules['essentia.standard'] = MagicMock()

from swaragram import core, utils

def load_audio(filepath, sr=22050):
    """ Loads audio sample. """
    return utils.loadAudio(filepath, sampleRate=sr)

def compute_stft(x):
    """ Calculates STFT of an audio signal. """
    return utils.computeSTFT(x)

def compute_ref_pitch(tonic):
    """ Computes tonic pitch and its frequency. """
    return utils.computeRefPitch(tonic)

def compute_swaragram(stft, ref_pitch, ref_freq):
    """ Computes a swaragram. """
    return core.computeSwaragram(stft, refPitch=ref_pitch, refFreq=ref_freq)

def apply_log_compression(swara):
    """ Applies log compression to the input matrix. """
    return utils.logCompression(swara)
