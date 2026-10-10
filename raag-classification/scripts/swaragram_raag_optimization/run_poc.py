
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import sys
import time
import numpy as np
import pandas as pd
import librosa
from collections import Counter
from numba import njit

# Add swaragram to path
sys.path.append(os.path.join(PROJECT_ROOT, r"third_party\swaragram-master"))
from swaragram import core

OPT_DIR = os.path.join(PROJECT_ROOT, r"scripts\swaragram_raag_optimization")
SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\raag_validation_optimization\split_manifest.csv")
TRF_MANIFEST = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\trf_training_manifest.csv")

# Re-implement utils.py functions to avoid essentia dependency
def computeSTFT(x, win_length=2048, hop_length=512, window='hann', pad_mode='constant'):
    return librosa.stft(x, n_fft=win_length, hop_length=hop_length, win_length=win_length, window=window, pad_mode=pad_mode)

def computeRefPitch(tonic):
    freqGen = np.vectorize(lambda x: 440 * 2 ** ((x - 69) / 12))
    midiPitchFreq = freqGen(np.arange(0,128))
    tonicMidiPitch = np.abs(tonic - midiPitchFreq).argmin()
    return tonicMidiPitch, midiPitchFreq[tonicMidiPitch]

@njit
def logCompression(x):
    eps = 2.220446049250313e-16
    return (23 * np.log10(eps + x))

def estimate_tonic_fast(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0:
        return 261.63 # Default C4
    midi_rounded = np.round(midi * 10) / 10
    most_common = Counter(midi_rounded).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def run_poc():
    print("Loading manifests...")
    df_split = pd.read_csv(SPLIT_MANIFEST)
    trf = pd.read_csv(TRF_MANIFEST).drop_duplicates('source_recording')
    
    splits = df_split.groupby('source_recording').first()['split'].to_dict()
    
    train_recs = [r for r, s in splits.items() if s == 'TRAIN'][:5]
    val_recs = [r for r, s in splits.items() if s == 'VAL'][:3]
    test_recs = [r for r, s in splits.items() if s == 'TEST'][:2]
    
    selected_recs = train_recs + val_recs + test_recs
    
    audio_paths = {}
    for _, row in trf.iterrows():
        if row['source_recording'] in selected_recs:
            audio_paths[row['source_recording']] = row['audio_path']
            
    print(f"Selected {len(audio_paths)} recordings for POC.")
    
    results = []
    
    for rec in selected_recs:
        path = audio_paths.get(rec)
        if not path or not os.path.exists(path):
            print(f"Missing audio for {rec}")
            continue
            
        print(f"\nProcessing {rec}...")
        t0 = time.time()
        
        try:
            # 1-3. Load audio robustly, mono, resample to 22050
            x, sr = librosa.load(path, sr=22050, mono=True)
            t_load = time.time() - t0
            
            # 4. Extract F0 and Tonic (fast)
            tonic = estimate_tonic_fast(x, sr)
            t_tonic = time.time() - t0 - t_load
            
            # 6-7. Convert to tonic-relative pitch and Generate Swaragram
            refPitch, refFreq = computeRefPitch(tonic)
            stft = computeSTFT(x, win_length=2048, hop_length=512)
            swara = core.computeSwaragram(stft, sampleRate=sr, winLength=2048, refPitch=refPitch, refFreq=refFreq)
            swara_log = logCompression(swara)
            
            t_swara = time.time() - t0 - t_load - t_tonic
            
            # 8-10. Verify shape, NaN/Inf, octave invariance
            shape = swara.shape
            has_nan = np.isnan(swara).any()
            has_inf = np.isinf(swara).any()
            
            results.append({
                "recording": rec,
                "split": splits[rec],
                "duration_sec": len(x)/sr,
                "tonic_hz": tonic,
                "refPitch_midi": refPitch,
                "shape": str(shape),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "t_load": t_load,
                "t_tonic": t_tonic,
                "t_swara": t_swara,
                "t_total": time.time() - t0
            })
            
            # Save POC artifact
            np.save(os.path.join(OPT_DIR, f"{rec}_swara.npy"), swara_log)
            
            print(f"  Tonic: {tonic:.1f}Hz (MIDI {refPitch}) | Shape: {shape}")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OPT_DIR, "poc_results.csv"), index=False)
    
    print("\n--- POC SUMMARY ---")
    print(f"Mean Tonic Time: {df_res['t_tonic'].mean():.2f}s")
    print(f"Mean Swaragram Time: {df_res['t_swara'].mean():.2f}s")
    print(f"NaNs found: {df_res['has_nan'].sum()}")
    print(f"Infs found: {df_res['has_inf'].sum()}")

if __name__ == '__main__':
    run_poc()
