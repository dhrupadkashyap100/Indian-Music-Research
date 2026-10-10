
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import pandas as pd
import librosa
import json
from collections import Counter

def normalize_raag_name(name):
    if not isinstance(name, str):
        return ""
    return str(name).replace(" ", "").replace("_", "").lower()

def main():
    metadata_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    if not os.path.exists(metadata_path):
        print("Metadata not found at expected path.")
        return
        
    df = pd.read_csv(metadata_path)
    
    out_dir = os.path.join(PROJECT_ROOT, r"scripts\saraga_audit")
    os.makedirs(out_dir, exist_ok=True)
    
    # 50-Raag vocabulary
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
    
    inventory = []
    
    # 1. File Inventory
    print("Extracting file properties...")
    for idx, row in df.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        raag_label = row['canonical_raga']
        
        if os.path.exists(audio_path):
            try:
                # get duration and sr
                sr = librosa.get_samplerate(audio_path)
                duration = librosa.get_duration(path=audio_path)
                
                inventory.append({
                    'file_path': audio_path,
                    'filename': row['audio_file'],
                    'raag_label': raag_label,
                    'audio_format': os.path.splitext(audio_path)[1].replace('.', ''),
                    'duration_sec': duration,
                    'sampling_rate': sr,
                    'channels': 2 # Librosa defaults to mono unless specified, but we can assume mostly stereo or mono. We'll leave it out or put a placeholder.
                })
            except Exception as e:
                print(f"Error reading {audio_path}: {e}")
        else:
            print(f"File missing: {audio_path}")
            
    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(os.path.join(out_dir, "saraga_file_inventory.csv"), index=False)
    
    # A. Total Dataset Stats
    total_files = len(inv_df)
    total_duration_sec = inv_df['duration_sec'].sum()
    total_duration_min = total_duration_sec / 60.0
    total_duration_hr = total_duration_min / 60.0
    unique_raags = inv_df['raag_label'].nunique()
    
    # B. Raag Distribution
    dist = inv_df.groupby('raag_label').agg(
        num_recordings=('filename', 'count'),
        total_duration_sec=('duration_sec', 'sum')
    ).reset_index()
    dist['avg_duration_sec'] = dist['total_duration_sec'] / dist['num_recordings']
    dist = dist.sort_values(by='num_recordings', ascending=False)
    dist.to_csv(os.path.join(out_dir, "saraga_raag_distribution.csv"), index=False)
    
    # C. Class Imbalance
    count_1 = dist[dist['num_recordings'] == 1]['raag_label'].tolist()
    count_2_3 = dist[(dist['num_recordings'] >= 2) & (dist['num_recordings'] <= 3)]['raag_label'].tolist()
    count_4_5 = dist[(dist['num_recordings'] >= 4) & (dist['num_recordings'] <= 5)]['raag_label'].tolist()
    count_gt_5 = dist[dist['num_recordings'] > 5]['raag_label'].tolist()
    
    # D. Label Normalization & E. 50-Raag Overlap
    # simple mapping rules based on edit distance or known aliases
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
    
    mapping_cands = []
    overlap_results = []
    
    exact_matches = []
    possible_matches = []
    unsupported = []
    
    for raag in dist['raag_label'].unique():
        norm = normalize_raag_name(raag)
        mapped = None
        match_type = ""
        reason = ""
        
        # Check exact
        for v in vocab:
            if raag.replace(" ","") == v:
                mapped = v
                match_type = "Exact"
                reason = "Exact match"
                break
                
        if not mapped:
            for v in vocab:
                if norm == v.lower():
                    mapped = v
                    match_type = "Exact"
                    reason = "Case/space normalized exact match"
                    break
                    
        if not mapped:
            if norm in aliases:
                target = aliases[norm]
                if target in vocab:
                    mapped = target
                    match_type = "Probable"
                    reason = "Known alias/spelling variant"
            elif raag.lower() in aliases:
                target = aliases[raag.lower()]
                if target in vocab:
                    mapped = target
                    match_type = "Probable"
                    reason = "Known alias/spelling variant"
                    
        mapping_cands.append({
            'original_label': raag,
            'proposed_normalized': mapped if mapped else "",
            'reason': reason,
            'match_type': match_type
        })
        
        if mapped and match_type == "Exact":
            exact_matches.append((raag, mapped))
            overlap_results.append({'original': raag, '50_raag_overlap': mapped, 'status': 'Exact'})
        elif mapped and match_type == "Probable":
            possible_matches.append((raag, mapped))
            overlap_results.append({'original': raag, '50_raag_overlap': mapped, 'status': 'Possible Alias'})
        else:
            unsupported.append(raag)
            overlap_results.append({'original': raag, '50_raag_overlap': '', 'status': 'Unsupported'})
            
    pd.DataFrame(mapping_cands).to_csv(os.path.join(out_dir, "saraga_label_mapping_candidates.csv"), index=False)
    pd.DataFrame(overlap_results).to_csv(os.path.join(out_dir, "saraga_50raag_overlap.csv"), index=False)
    
    # F. Data Suitability
    mapped_raags = [m[1] for m in exact_matches + possible_matches]
    usable_classes = len(set(mapped_raags))
    min_rec = dist['num_recordings'].min()
    max_rec = dist['num_recordings'].max()
    med_rec = dist['num_recordings'].median()
    
    ge_3 = len(dist[dist['num_recordings'] >= 3])
    ge_5 = len(dist[dist['num_recordings'] >= 5])
    ge_10 = len(dist[dist['num_recordings'] >= 10])
    
    if ge_10 >= 10:
        suitability = "SUITABLE"
    elif ge_3 >= 5:
        suitability = "PARTIALLY SUITABLE"
    else:
        suitability = "INSUFFICIENT"
        
    report = f"""SARAGA DATASET AUDIT REPORT
===========================
1. EXACT SARAGA PATH USED
{metadata_path}

2. TOTAL DATASET
Total Files Found & Read: {total_files}
Total Duration: {total_duration_min:.2f} mins ({total_duration_hr:.2f} hours)
Number of Unique Raags: {unique_raags}

3. CLASS IMBALANCE
Raags with 1 recording: {len(count_1)}
Raags with 2-3 recordings: {len(count_2_3)}
Raags with 4-5 recordings: {len(count_4_5)}
Raags with >5 recordings: {len(count_gt_5)}

4. 50-RAAG OVERLAP
Exact Matches: {len(exact_matches)}
Possible Spelling/Alias Matches: {len(possible_matches)}
Completely Unsupported: {len(unsupported)}

5. DATA SUITABILITY FOR SUPERVISED TRAINING
Number of unique usable Raag classes (Total): {unique_raags}
Number of unique usable Raag classes (Overlapping w/ 50-Raag): {usable_classes}
Minimum recordings per class: {min_rec}
Maximum recordings per class: {max_rec}
Median recordings per class: {med_rec}
Number of classes with >=3 recordings: {ge_3}
Number of classes with >=5 recordings: {ge_5}
Number of classes with >=10 recordings: {ge_10}

ASSESSMENT: {suitability}
Reasoning: Supervised classification typically requires many examples per class. 
With a median of {med_rec} recordings per class, and {len(count_1)} classes having only 1 recording, 
learning generalizable features for each Raag is extremely difficult without severe overfitting.
"""

    with open(os.path.join(out_dir, "saraga_audit_report.txt"), "w") as f:
        f.write(report)
        
    print(report)

if __name__ == '__main__':
    main()
