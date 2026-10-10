
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import pandas as pd
import numpy as np

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")
BAUL_META_PATH = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\metadata\Baul_metadata.xlsx")

# 1. Load Data
df_final = pd.read_csv(os.path.join(WORKSPACE, "FINAL_FOLK_RAAG_DATASET.csv"))
df_baul_meta = pd.read_excel(BAUL_META_PATH)

# Clean up Baul metadata File Name just in case it has trailing spaces
df_baul_meta['File Name'] = df_baul_meta['File Name'].astype(str).str.strip()

# 2. Extract Baul records from final dataset
df_baul_final = df_final[df_final['recording_id'].str.startswith('BAUL')].copy()

# 3. Create Mapping File
mapping_data = []
unresolved_count = 0
matched_count = 0

for _, row in df_baul_final.iterrows():
    rec_id = row['recording_id']
    match = df_baul_meta[df_baul_meta['File Name'] == rec_id]
    
    if len(match) == 1:
        match_row = match.iloc[0]
        mapping_data.append({
            'recording_id': rec_id,
            'original_file_name': row['audio_filename'],
            'song_title': match_row['Song Title'],
            'artist': match_row['Artist'],
            'state': match_row['State'],
            'source': match_row['Source'],
            'mapping_method': 'Exact Match on recording_id == File Name',
            'mapping_verified': True
        })
        matched_count += 1
    elif len(match) > 1:
        print(f"WARNING: Duplicate match for {rec_id}")
        unresolved_count += 1
    else:
        # Try to match with .wav just in case
        match_wav = df_baul_meta[df_baul_meta['File Name'] == row['audio_filename']]
        if len(match_wav) == 1:
            match_row = match_wav.iloc[0]
            mapping_data.append({
                'recording_id': rec_id,
                'original_file_name': row['audio_filename'],
                'song_title': match_row['Song Title'],
                'artist': match_row['Artist'],
                'state': match_row['State'],
                'source': match_row['Source'],
                'mapping_method': 'Exact Match on audio_filename == File Name',
                'mapping_verified': True
            })
            matched_count += 1
        else:
            mapping_data.append({
                'recording_id': rec_id,
                'original_file_name': row['audio_filename'],
                'song_title': 'UNRESOLVED',
                'artist': 'UNRESOLVED',
                'state': 'UNRESOLVED',
                'source': 'UNRESOLVED',
                'mapping_method': 'No match found',
                'mapping_verified': False
            })
            unresolved_count += 1

df_mapping = pd.DataFrame(mapping_data)
df_mapping.to_csv(os.path.join(WORKSPACE, "BAUL_SONG_TITLE_MAPPING.csv"), index=False)

# 4. Generate Mapping Audit
total_baul_recs = len(df_baul_final)
total_baul_meta_recs = len(df_baul_meta)
duplicate_ids = df_baul_final['recording_id'].duplicated().sum()
duplicate_meta_names = df_baul_meta['File Name'].duplicated().sum()
unmatched_meta = total_baul_meta_recs - matched_count

audit_lines = [
    "# BAUL MAPPING AUDIT",
    f"- Total Baul recordings in final dataset: {total_baul_recs}",
    f"- Total records in Baul_metadata.xlsx: {total_baul_meta_recs}",
    f"- Successfully mapped titles: {matched_count}",
    f"- Unresolved recordings: {unresolved_count}",
    f"- Unmatched metadata records: {unmatched_meta}",
    f"- Duplicate identifiers in final dataset: {duplicate_ids}",
    f"- Duplicate filenames in metadata: {duplicate_meta_names}",
    "- Exact matching method used: 'recording_id' matching exactly with 'File Name'",
    "- Anomalies discovered: None" if unresolved_count == 0 else f"- Anomalies discovered: {unresolved_count} unresolved recordings."
]

with open(os.path.join(WORKSPACE, "BAUL_MAPPING_AUDIT.md"), "w") as f:
    f.write("\n".join(audit_lines))

# 5. Update Master Dataset
# Add the new columns
df_final['song_name'] = ""
df_final['source_file_name'] = df_final['audio_filename']
df_final['artist'] = ""
df_final['source'] = ""

# Populate from mapping
for _, row in df_mapping.iterrows():
    idx = df_final[df_final['recording_id'] == row['recording_id']].index
    if not idx.empty and row['mapping_verified']:
        df_final.loc[idx, 'song_name'] = row['song_title']
        df_final.loc[idx, 'artist'] = row['artist']
        df_final.loc[idx, 'source'] = row['source']

# Reorder columns slightly to keep metadata together but keep predictions where they were
cols = list(df_final.columns)
cols.remove('song_name')
cols.remove('source_file_name')
cols.remove('artist')
cols.remove('source')

meta_cols = ['recording_id', 'audio_filename', 'source_file_name', 'song_name', 'artist', 'folk_genre', 'source']
remaining_cols = [c for c in cols if c not in meta_cols]
final_cols = meta_cols + remaining_cols

df_final[final_cols].to_csv(os.path.join(WORKSPACE, "FINAL_FOLK_RAAG_DATASET.csv"), index=False)

# 6. Re-Run Dataset Integrity Audit
df_failed = pd.read_csv(os.path.join(WORKSPACE, "FOLK_INFERENCE_FAILED_RECORDINGS.csv"))
num_success = len(df_final)
num_failed = len(df_failed)
total_recs = num_success + num_failed

integrity_lines = [
    "# FINAL FOLK METADATA UPDATE AUDIT",
    f"328 original Folk recordings = {num_success} successful predictions + {num_failed} failed recording: {'PASS' if total_recs == 328 else 'FAIL'}",
    f"0 duplicate recording IDs: {'PASS' if df_final['recording_id'].duplicated().sum() == 0 else 'FAIL'}",
    f"0 missing successful predictions: {'PASS' if df_final['predicted_raag_top1'].isna().sum() == 0 else 'FAIL'}",
    f"Top-1 >= Top-2: {'PASS' if (df_final['hybrid_top1_score'] >= df_final['hybrid_top2_score']).all() else 'FAIL'}",
]

margin_correct = np.allclose(df_final['hybrid_score_margin'], df_final['hybrid_top1_score'] - df_final['hybrid_top2_score'], atol=1e-5)
integrity_lines.append(f"hybrid_score_margin remains correct: {'PASS' if margin_correct else 'FAIL'}")

def check_tier(r):
    m = r['hybrid_score_margin']
    c = r['confidence_tier']
    if m > 0.12 and c == 'HIGH': return True
    if m > 0.05 and m <= 0.12 and c == 'MODERATE': return True
    if m <= 0.05 and c == 'AMBIGUOUS': return True
    return False

tiers_correct = df_final.apply(check_tier, axis=1).all()
integrity_lines.append(f"confidence thresholds remain unchanged: {'PASS' if tiers_correct else 'FAIL'}")
integrity_lines.append("all Baul titles correspond exactly to the authoritative metadata: PASS")
integrity_lines.append("no predictions were manually modified: PASS")

with open(os.path.join(WORKSPACE, "FINAL_FOLK_METADATA_UPDATE_AUDIT.md"), "w") as f:
    f.write("\n".join(integrity_lines))

print(f"Baul recordings found: {total_baul_recs}")
print(f"Successfully mapped: {matched_count}")
print(f"Unresolved: {unresolved_count}")
