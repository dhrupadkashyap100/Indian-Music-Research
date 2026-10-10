
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import pandas as pd
import librosa
import numpy as np
from math import floor

def normalize_raag_name(name):
    if not isinstance(name, str):
        return ""
    return str(name).replace(" ", "").replace("_", "").lower()

def main():
    metadata_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    out_dir = os.path.join(PROJECT_ROOT, r"scripts\saraga_reference_audit")
    os.makedirs(out_dir, exist_ok=True)
    
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
    vocab_lower = {v.lower(): v for v in vocab}
    
    aliases = {
        "bhimpalasi": "Bheempalasi",
        "bhimpalas": "Bheempalasi",
        "ahribhairav": "AheerBhairav",
        "ahirobhairav": "AheerBhairav",
        "ahir bhairav": "AheerBhairav",
        "darbari": "DarbariKanada",
        "alhaiyabilawal": "AlhaiyaBilawal",
        "puriakalyan": "PuriyaKalyan",
        "puriadhanashree": "PuriyaDhanashri",
        "puriadhanashri": "PuriyaDhanashri",
        "marubihaag": "MaruBihag",
        "bhupali": "Bhoopali",
        "bageshri": "Bageshree",
        "bhairabi": "Bhairavi",
        "lalat": "Lalit",
        "kalavati": "Kalawati"
    }

    df = pd.read_csv(metadata_path)
    
    exact_matches = []
    alias_candidates = []
    unsupported = []
    
    inventory_records = []
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        raag_label = str(row['canonical_raga'])
        norm_label = normalize_raag_name(raag_label)
        
        match_status = "Unsupported"
        mapped_model_label = ""
        
        # Check Exact
        for v in vocab:
            if raag_label.replace(" ", "") == v or norm_label == v.lower():
                match_status = "Exact"
                mapped_model_label = v
                break
                
        # Check Alias
        if match_status == "Unsupported":
            if norm_label in aliases and aliases[norm_label] in vocab:
                match_status = "Possible Alias"
                mapped_model_label = aliases[norm_label]
            elif raag_label.lower() in aliases and aliases[raag_label.lower()] in vocab:
                match_status = "Possible Alias"
                mapped_model_label = aliases[raag_label.lower()]
                
        # Process audio if Exact
        duration = 0.0
        sr = 0
        channels = 0
        file_format = os.path.splitext(audio_path)[1].replace('.', '')
        
        if os.path.exists(audio_path):
            try:
                # Load metadata quickly without full decode if possible, or just use librosa
                sr = librosa.get_samplerate(audio_path)
                duration = librosa.get_duration(path=audio_path)
                channels = 2 # Assuming standard stereo for SARAGA, librosa doesn't expose channels easily without loading
            except Exception as e:
                print(f"Error reading {audio_path}: {e}")
        else:
            print(f"File missing: {audio_path}")
            
        artist = row.get('artist', 'Not available')
        
        rec = {
            'original_label': raag_label,
            'model_label': mapped_model_label,
            'match_status': match_status,
            'filename': row['audio_file'],
            'full_audio_path': audio_path,
            'duration_sec': duration,
            'sampling_rate': sr,
            'channels': channels,
            'file_format': file_format,
            'artist': artist,
            'usable_5sec_segments': floor(duration / 5.0)
        }
        
        if match_status == "Exact":
            exact_matches.append(rec)
            inventory_records.append(rec)
        elif match_status == "Possible Alias":
            alias_candidates.append(rec)
        else:
            unsupported.append(rec)
            
    # Create Inventory CSV
    inv_df = pd.DataFrame(inventory_records)
    inv_df.to_csv(os.path.join(out_dir, "saraga_exact_overlap_inventory.csv"), index=False)
    
    # Create Alias Candidates CSV
    alias_records = []
    alias_grouped = {}
    for a in alias_candidates:
        key = (a['original_label'], a['model_label'])
        if key not in alias_grouped:
            alias_grouped[key] = 0
        alias_grouped[key] += 1
        
    for (orig, model_l), count in alias_grouped.items():
        alias_records.append({
            'SARAGA_label': orig,
            'Potential_model_label': model_l,
            'Reason': "Known alias/spelling variation",
            'Number_of_SARAGA_recordings': count
        })
    pd.DataFrame(alias_records).to_csv(os.path.join(out_dir, "saraga_alias_candidates.csv"), index=False)
    
    # Reference Quality
    ref_records = []
    grouped = inv_df.groupby('model_label')
    
    for raag, group in grouped:
        num_rec = len(group)
        tot_dur = group['duration_sec'].sum()
        avg_dur = group['duration_sec'].mean()
        min_dur = group['duration_sec'].min()
        max_dur = group['duration_sec'].max()
        tot_seg = group['usable_5sec_segments'].sum()
        
        if num_rec == 1:
            quality = "VERY LIMITED REFERENCE"
            rec_cmd = "EXCLUDE FOR NOW"
        elif num_rec >= 2 and num_rec <= 3:
            quality = "LIMITED REFERENCE"
            rec_cmd = "USE WITH CAUTION"
        else:
            quality = "GOOD REFERENCE"
            rec_cmd = "USE IN FIRST EXPERIMENT"
            
        ref_records.append({
            'Raag': raag,
            'Number of recordings': num_rec,
            'Total duration': tot_dur,
            'Average duration': avg_dur,
            'Minimum duration': min_dur,
            'Maximum duration': max_dur,
            '5-sec segments': tot_seg,
            'Reference quality': quality,
            'Recommendation': rec_cmd
        })
        
    # Sort by number of recordings descending, then total duration
    ref_df = pd.DataFrame(ref_records).sort_values(by=['Number of recordings', 'Total duration'], ascending=[False, False])
    ref_df.to_csv(os.path.join(out_dir, "saraga_reference_quality.csv"), index=False)
    
    # Summary JSON
    summary_data = {
        'total_saraga_recordings': len(df),
        'total_saraga_raags': df['canonical_raga'].nunique(),
        'exact_overlap_raags': len(ref_df),
        'possible_alias_raags': len(alias_records),
        'unsupported_raags': len(set([x['original_label'] for x in unsupported]))
    }
    with open(os.path.join(out_dir, "saraga_reference_audit_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)
        
    # Report TXT
    report = []
    report.append("SARAGA REFERENCE AUDIT REPORT")
    report.append("=============================")
    report.append(f"A. Total SARAGA recordings: {len(df)}")
    report.append(f"B. Total SARAGA Raags: {df['canonical_raga'].nunique()}")
    report.append(f"C. Number of exact overlaps: {len(ref_df)}")
    report.append(f"D. Number of possible alias overlaps: {len(alias_records)}")
    report.append(f"E. Number of unsupported Raags: {len(set([x['original_label'] for x in unsupported]))}")
    
    report.append("\nF. Reference Quality of Exact-Overlap Raags:")
    for _, row in ref_df.iterrows():
        report.append(f"   - {row['Raag']}: {row['Number of recordings']} recordings, {row['Total duration']:.2f}s total -> {row['Reference quality']}")
        
    report.append("\nG. Recommended classes for first similarity experiment:")
    rec_classes = ref_df[ref_df['Recommendation'] == 'USE IN FIRST EXPERIMENT']['Raag'].tolist()
    if not rec_classes:
        report.append("   - NONE (No Raag has >= 4 recordings. Best available are 'LIMITED REFERENCE')")
    else:
        for r in rec_classes:
            report.append(f"   - {r}")
            
    report.append("\nH. Classes that should be excluded (Very Limited Reference):")
    ex_classes = ref_df[ref_df['Recommendation'] == 'EXCLUDE FOR NOW']['Raag'].tolist()
    for r in ex_classes:
        report.append(f"   - {r}")
        
    report.append("\nI. Why conventional supervised training is currently inappropriate:")
    report.append("   Most overlapping Raags (like Bairagi, Multani, Marwa) only have a single recording in SARAGA.")
    report.append("   A supervised classifier trained on one recording will perfectly memorize the singer's voice, tempo, and instruments, but completely fail to generalize to a different artist or genre (like the Folk dataset). There is zero intra-class variance.")
    
    report.append("\nJ. Why segment-level splitting would cause leakage:")
    report.append("   Taking a single 15-minute recording, cutting it into 5-second segments, and randomly dividing them into Train/Validation/Test sets causes massive data leakage. The model will see segments from the exact same singer/recording session in both training and testing, artificially inflating accuracy without learning true musical invariants.")
    
    report.append("\nK. Important limitations of using SARAGA as a reference bank:")
    report.append("   SARAGA contains high-quality long-form classical performances, but lacks the dataset diversity needed for robust Raag reference. A single recording may focus heavily on a specific movement (e.g., Alap) and not capture the full melodic progression of the Raag. Comparing lively, rhythmic 2-minute Folk songs to a single 20-minute slow classical Alap using basic acoustic features could yield poor similarity scores due to tempo and stylistic mismatch rather than Raag mismatch.")
    
    with open(os.path.join(out_dir, "saraga_reference_audit_report.txt"), "w") as f:
        f.write("\n".join(report))
        
    print("Audit Complete.")
    print("\n".join(report))

if __name__ == '__main__':
    main()
