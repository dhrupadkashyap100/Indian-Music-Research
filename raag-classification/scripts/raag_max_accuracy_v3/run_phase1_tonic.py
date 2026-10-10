
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\raag_max_accuracy_v3")
SARAGA_META = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
SARAGA_MAPPING = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval\saraga_evaluation_manifest.csv")

def get_cents_error(est, gt):
    cents = 1200 * np.log2(est / gt)
    pc_err = np.abs(np.mod(cents + 600, 1200) - 600)
    octave = np.abs(cents) >= 1150 and pc_err < 50
    exact = np.abs(cents) < 50
    gross = pc_err >= 50
    return cents, pc_err, exact, octave, gross

def est_existing(f0):
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0: return 261.63
    most_common = Counter(np.round(midi * 10) / 10).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def est_strong(f0, rms):
    voiced_mask = (f0 > 0) & (rms > 0.15 * np.max(rms))
    midi = librosa.hz_to_midi(f0[voiced_mask])
    if len(midi) == 0: return est_existing(f0)
    most_common = Counter(np.round(midi * 10) / 10).most_common(1)[0][0]
    return librosa.midi_to_hz(most_common)

def est_median(f0):
    midi = librosa.hz_to_midi(f0[f0 > 0])
    if len(midi) == 0: return 261.63
    return librosa.midi_to_hz(np.median(midi))

def est_ensemble(f0):
    chunks = np.array_split(f0, 5)
    ests = []
    for c in chunks:
        midi = librosa.hz_to_midi(c[c > 0])
        if len(midi) > 0:
            ests.append(Counter(np.round(midi * 10) / 10).most_common(1)[0][0])
    if not ests: return 261.63
    final = Counter(ests).most_common(1)[0][0]
    return librosa.midi_to_hz(final)

def process_recording(task):
    rec_id = task['id']
    audio_path = task['audio_path']
    tonic_gt = task['tonic_gt']
    
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        f0 = librosa.yin(y, fmin=50, fmax=1000, sr=sr)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_interp = np.interp(np.arange(len(f0)), np.arange(len(rms)) * 512 / (sr/22050), rms)
        
        e_ex = est_existing(f0)
        e_st = est_strong(f0, rms_interp)
        e_med = est_median(f0)
        e_ens = est_ensemble(f0)
        
        res = {'id': rec_id, 'GT': tonic_gt}
        for name, est in [('Existing', e_ex), ('StrongVoiced', e_st), ('Median', e_med), ('Ensemble', e_ens)]:
            c, pc, ex, oc, gr = get_cents_error(est, tonic_gt)
            res[f'{name}_cents'] = c
            res[f'{name}_pc_err'] = pc
            res[f'{name}_exact'] = ex
            res[f'{name}_octave'] = oc
            res[f'{name}_gross'] = gr
            
        return res
    except Exception as e:
        print(f"Error {rec_id}: {e}")
        return None

def main():
    sar_meta = pd.read_csv(SARAGA_META)
    sar_map = pd.read_csv(SARAGA_MAPPING)
    sar_mapped = sar_map[sar_map['evaluation_eligibility'] == True] if 'evaluation_eligibility' in sar_map.columns else sar_map[sar_map['mapped_trf_raag'].notna() & (sar_map['mapped_trf_raag'] != 'NONE')]
    sar_joined = pd.merge(sar_mapped, sar_meta, left_on='saraga_recording_id', right_on='unique_id')
    
    tasks = []
    for _, row in sar_joined.iterrows():
        tasks.append({
            'id': row['saraga_recording_id'],
            'audio_path': os.path.join(row['folder'], row['audio_file']),
            'tonic_gt': row['tonic']
        })
        
    print(f"Running tonic evaluation on {len(tasks)} SARAGA recordings...")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-2)) as executor:
        futures = [executor.submit(process_recording, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    
    summary = []
    for m in ['Existing', 'StrongVoiced', 'Median', 'Ensemble']:
        exact = df[f'{m}_exact'].mean()
        octave = df[f'{m}_octave'].mean()
        gross = df[f'{m}_gross'].mean()
        med_err = df[f'{m}_pc_err'].median()
        summary.append({
            'Method': m,
            'Exact_Match': exact,
            'Octave_Error': octave,
            'Gross_Error': gross,
            'Median_PC_Error_Cents': med_err
        })
        
    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string())
    sum_df.to_csv(os.path.join(WORKSPACE, "tonic_optimization_results.csv"), index=False)

if __name__ == '__main__':
    main()
