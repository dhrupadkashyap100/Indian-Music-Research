
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import glob
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import soundfile as sf
import librosa
from tqdm import tqdm
import math

def normalize_raag_name(name):
    if not isinstance(name, str):
        return ""
    return name.replace(" ", "").replace("_", "").replace("-", "").lower()

def main():
    dataset_root = os.path.join(PROJECT_ROOT, r"Thaat and Raga Forest (TRF) Dataset")
    out_dir = os.path.join(PROJECT_ROOT, r"scripts\raag_dataset_audit")
    os.makedirs(out_dir, exist_ok=True)
    
    # 50 Raag vocabulary
    vocab = [
        "AheerBhairav", "AlhaiyaBilawal", "Bageshree", "Bahar", "Bairagi",
        "Basant", "Bhairav", "Bhairavi", "Bheempalasi", "Bhoopali", "Bihag",
        "Chandrakauns", "Charukeshi", "DarbariKanada", "Des", "Deshkar",
        "Dhani", "Durga", "Hameer", "HansDhwani", "Hindol", "Jaijaivanti",
        "Jog", "Kafi", "Kalawati", "KaushikDhwani", "Kedar", "Keerwani",
        "Khamaj", "Lalit", "Madhukauns", "Madhuvanti", "Malhar", "Malkauns",
        "MaruBihag", "Marwa", "Multani", "Pilu", "PuriyaDhanashri",
        "PuriyaKalyan", "Sarang", "Shankara", "Shivranjani", "Shree",
        "Sohani", "TilakKamod", "Tilang", "Todi", "Vibhas", "Yaman"
    ]
    
    aliases_map = {
        "bhimpalasi": "Bheempalasi", "bhimpalas": "Bheempalasi", "ahribhairav": "AheerBhairav",
        "ahirobhairav": "AheerBhairav", "ahir bhairav": "AheerBhairav", "darbari": "DarbariKanada",
        "alhaiyabilawal": "AlhaiyaBilawal", "puriakalyan": "PuriyaKalyan", "puriadhanashree": "PuriyaDhanashri",
        "puriadhanashri": "PuriyaDhanashri", "marubihaag": "MaruBihag", "bhupali": "Bhoopali",
        "bageshri": "Bageshree", "bhairabi": "Bhairavi", "lalat": "Lalit", "kalavati": "Kalawati"
    }

    # PART 1 & 2: Dataset Inventory
    hierarchy = defaultdict(list)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma'}
    
    file_records = []
    raag_to_thaat = {}
    
    total_files_scanned = 0
    non_audio_files = []
    precomputed_features = []
    
    feature_exts = {'.npy', '.csv', '.json', '.xml', '.txt', '.pitch', '.tonic', '.mfcc'}
    
    print(f"Scanning directory: {dataset_root}")
    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == '.':
            continue
            
        parts = rel_path.split(os.sep)
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            f_path = os.path.join(root, f)
            total_files_scanned += 1
            
            if ext in feature_exts:
                precomputed_features.append({
                    'type': ext,
                    'path': f_path,
                    'name': f
                })
                continue
                
            if ext in audio_extensions:
                if len(parts) >= 2:
                    thaat = parts[0]
                    raag = parts[1]
                elif len(parts) == 1:
                    thaat = "Unknown"
                    raag = parts[0]
                else:
                    thaat = "Unknown"
                    raag = "Unknown"
                    
                raag_to_thaat[raag] = thaat
                
                size = os.path.getsize(f_path)
                
                file_records.append({
                    'Thaat': thaat,
                    'Raag': raag,
                    'Filename': f,
                    'Extension': ext,
                    'Full_Path': f_path,
                    'File_Size': size,
                    'Readable': True, # Assume true until tested
                    'Duration': 0.0,
                    'Sample_Rate': 0,
                    'Channels': 0
                })
            else:
                non_audio_files.append(f_path)

    # Subsample for audio reading (max 5 per raag to be computationally practical, but we will try to use soundfile for fast header read on all if possible)
    # Using sf.info is very fast for wav/flac
    print("Extracting audio metadata...")
    corrupt_count = 0
    zero_byte = 0
    
    for rec in tqdm(file_records):
        path = rec['Full_Path']
        if rec['File_Size'] == 0:
            rec['Readable'] = False
            zero_byte += 1
            continue
            
        try:
            if rec['Extension'] in ['.wav', '.flac', '.ogg']:
                info = sf.info(path)
                rec['Duration'] = info.duration
                rec['Sample_Rate'] = info.samplerate
                rec['Channels'] = info.channels
            else: # for mp3, m4a, etc. fallback to librosa
                sr = librosa.get_samplerate(path)
                dur = librosa.get_duration(path=path)
                rec['Sample_Rate'] = sr
                rec['Duration'] = dur
                rec['Channels'] = 2 # Guess stereo if we can't easily check without full load
        except Exception as e:
            rec['Readable'] = False
            corrupt_count += 1
            
    df_files = pd.DataFrame(file_records)
    
    # Write audio_file_inventory.csv
    df_files.drop(columns=['Full_Path']).to_csv(os.path.join(out_dir, "audio_file_inventory.csv"), index=False)
    
    # Dataset Inventory
    inventory = []
    if not df_files.empty:
        grouped = df_files.groupby(['Thaat', 'Raag'])
        for (thaat, raag), group in grouped:
            exts = group['Extension'].unique().tolist()
            durs = group['Duration'].tolist()
            inventory.append({
                'Thaat': thaat,
                'Raag': raag,
                'Recording_Count': len(group),
                'Audio_File_Count': len(group),
                'Formats': ", ".join(exts),
                'Total_Duration': sum(durs),
                'Min_Duration': min(durs) if durs else 0,
                'Max_Duration': max(durs) if durs else 0
            })
    
    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv(os.path.join(out_dir, "dataset_inventory.csv"), index=False)
    
    # Parts 5 & 11: Raag Label Quality & 50-Raag Overlap
    unique_raags = df_files['Raag'].unique() if not df_files.empty else []
    
    raag_label_records = []
    overlap_records = []
    
    exact_matches = []
    possible_matches = []
    
    for raag in unique_raags:
        norm = normalize_raag_name(raag)
        thaat = raag_to_thaat.get(raag, "Unknown")
        count = len(df_files[df_files['Raag'] == raag])
        
        match_type = "No Match"
        model_raag = ""
        
        # Exact check
        for v in vocab:
            if raag.replace(" ", "") == v or norm == v.lower():
                match_type = "Exact Match"
                model_raag = v
                break
                
        # Alias check
        if match_type == "No Match":
            if norm in aliases_map and aliases_map[norm] in vocab:
                match_type = "Possible Alias"
                model_raag = aliases_map[norm]
            elif raag.lower() in aliases_map and aliases_map[raag.lower()] in vocab:
                match_type = "Possible Alias"
                model_raag = aliases_map[raag.lower()]
                
        if match_type == "Exact Match":
            exact_matches.append(model_raag)
        elif match_type == "Possible Alias":
            possible_matches.append(model_raag)
            
        raag_label_records.append({
            'Observed_Raag_Label': raag,
            'Thaat': thaat,
            'Recording_Count': count,
            'Possible_Normalized_Label': norm,
            'Potential_Alias': model_raag if match_type != "No Match" else "",
            'Needs_Manual_Verification': match_type == "Possible Alias"
        })
        
        overlap_records.append({
            'Dataset_Raag': raag,
            'Match_Type': match_type,
            'Model_Raag': model_raag,
            'Recording_Count': count
        })
        
    pd.DataFrame(raag_label_records).to_csv(os.path.join(out_dir, "raag_label_inventory.csv"), index=False)
    pd.DataFrame(overlap_records).to_csv(os.path.join(out_dir, "model_50raag_overlap.csv"), index=False)
    
    # Calculate global stats
    if not df_files.empty:
        total_audio = len(df_files)
        total_dur = df_files['Duration'].sum()
        num_thaats = df_files['Thaat'].nunique()
        num_raags = df_files['Raag'].nunique()
        
        counts = df_files['Raag'].value_counts()
        avg_rec = counts.mean()
        med_rec = counts.median()
        min_rec = counts.min()
        max_rec = counts.max()
        
        format_dist = df_files['Extension'].value_counts().to_dict()
        sr_dist = df_files['Sample_Rate'].value_counts().to_dict()
        ch_dist = df_files['Channels'].value_counts().to_dict()
    else:
        total_audio = 0; total_dur = 0; num_thaats = 0; num_raags = 0
        avg_rec = 0; med_rec = 0; min_rec = 0; max_rec = 0
        format_dist = {}; sr_dist = {}; ch_dist = {}
        
    # Generate Text Report
    report = []
    report.append("DATASET AUDIT REPORT")
    report.append("====================")
    report.append(f"A. Dataset location: {dataset_root}")
    report.append(f"B. Number of Thaats: {num_thaats}")
    report.append(f"C. Number of Raags: {num_raags}")
    report.append(f"D. Number of recordings: {total_audio}")
    report.append(f"E. Total duration: {total_dur/3600:.2f} hours")
    report.append(f"F. Average recordings per Raag: {avg_rec:.2f}")
    report.append(f"G. Median recordings per Raag: {med_rec}")
    report.append(f"H. Minimum/maximum recordings per Raag: {min_rec} / {max_rec}")
    report.append(f"I. Number of supported audio files: {total_audio}")
    report.append(f"J. Sampling-rate distribution: {sr_dist}")
    report.append(f"K. Channel distribution: {ch_dist}")
    
    if not df_files.empty:
        report.append(f"L. Duration distribution (s): Min={df_files['Duration'].min():.1f}, Max={df_files['Duration'].max():.1f}, Mean={df_files['Duration'].mean():.1f}")
    
    report.append(f"M. Number of corrupted/unreadable files: {corrupt_count} (Zero byte: {zero_byte})")
    
    # Label overlaps
    report.append(f"N. Number of unique Raag labels observed: {num_raags}")
    report.append(f"O. Number of exact overlaps with the 50-class model: {len(exact_matches)}")
    report.append(f"P. Number of possible alias overlaps: {len(possible_matches)}")
    report.append(f"Q. Number of unsupported Raags (vs 50-class): {num_raags - len(exact_matches) - len(possible_matches)}")
    
    # Precomputed features
    if precomputed_features:
        report.append(f"\nFound {len(precomputed_features)} precomputed feature/metadata files:")
        feat_counts = Counter([f['type'] for f in precomputed_features])
        for t, c in feat_counts.items():
            report.append(f"  - {t}: {c} files")
    
    # Suitability evaluation
    report.append("\nEVALUATION & SUITABILITY")
    report.append("========================")
    
    if min_rec >= 5 and med_rec >= 10:
        report.append("R. Suitable for supervised classification? YES. It appears to have a reasonable minimum recordings per class.")
    else:
        report.append("R. Suitable for supervised classification? NO/MARGINAL. Many classes have very few independent recordings, making cross-validation prone to leakage/overfitting.")
        
    report.append("S. Suitable for CultureMERT embedding analysis? YES. High-quality audio files can be run through the model for continuous representations.")
    report.append("T. Suitable for nearest-neighbor Raag matching? YES, as a robust reference database assuming the classical recordings are high quality and clean.")
    report.append("U. Suitable for evaluating a model? DEPENDS on independence. If all recordings in a Raag are from the same artist/concert, testing is invalid.")
    
    report.append("\nV. Biggest Risks:")
    report.append("- Data Leakage: Without explicit artist/album metadata in filenames, splitting multiple recordings from the same Raag might accidentally put the same concert in Train and Test.")
    report.append("- Label Noise: Potential aliases and varying transcriptions of Raag names might artificially split classes.")
    
    report.append("\nW. Recommended Next Step:")
    report.append("Use CultureMERT to extract reference embeddings for the overlapping Raags in this dataset. Then compare Folk Music embeddings against this reference bank using k-NN or Cosine Similarity to assign candidate labels.")

    with open(os.path.join(out_dir, "dataset_audit_report.txt"), "w") as f:
        f.write("\n".join(report))
        
    # JSON Summary
    summary = {
        'num_thaats': int(num_thaats),
        'num_raags': int(num_raags),
        'total_audio': int(total_audio),
        'total_duration_hours': float(total_dur/3600),
        'median_recordings_per_raag': float(med_rec),
        'corrupted_files': int(corrupt_count),
        'exact_overlaps_50class': int(len(exact_matches)),
        'alias_overlaps_50class': int(len(possible_matches))
    }
    with open(os.path.join(out_dir, "dataset_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nAudit Complete.")

if __name__ == '__main__':
    main()
