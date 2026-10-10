
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import pandas as pd

EVAL_DIR = os.path.join(PROJECT_ROOT, r"scripts\culturemert_cross_dataset_saraga_eval")

# Re-run mapping with verified aliases
df_saraga = pd.read_csv(os.path.join(EVAL_DIR, "saraga_inventory.csv"))
df_trf_vocab = pd.read_csv(os.path.join(EVAL_DIR, "trf_61_class_vocabulary.csv"))

trf_norms = {r.strip().lower().replace(" ", "").replace("_", "").replace("-", ""): (r, i) for i, r in zip(df_trf_vocab['class_index'], df_trf_vocab['trf_raag_name'])}
# Actually we need the TRF label and index
trf_norms = {r.strip().lower().replace(" ", "").replace("_", "").replace("-", ""): (r, i) for i, r in zip(df_trf_vocab['class_index'], df_trf_vocab['trf_raag_name'])}

def normalize_string(s):
    if not isinstance(s, str): return ""
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

# Hand-verified Canonical Aliases mapping
aliases = {
    'sohini': 'sohni',
    'rageshri': 'rageshree',
    'shuddhakalyan': 'suddhakalyan',
    'des': 'desh',
    'lalat': 'lalit',
    'marwa': 'marva',
    'bhairabi': 'bhairavi',
    'bhimpalas': 'bhimpalasi',
    'bhoop': 'bhoopali',
    'kalyan': 'yamanorkalyan',
    'todi': 'miyankitodiortodiordarbaritodi'
}

mapping = []
for _, row in df_saraga.iterrows():
    orig_label = str(row['canonical_raga'])
    norm_label = normalize_string(orig_label)
    
    if norm_label in trf_norms:
        status = "EXACT_MATCH"
        reason = "Normalized string exact match"
        conf = "HIGH"
        mapped_trf, mapped_idx = trf_norms[norm_label]
    elif norm_label in aliases and aliases[norm_label] in trf_norms:
        status = "CANONICAL_ALIAS"
        reason = "Hand-verified canonical alias/spelling variation"
        conf = "HIGH"
        mapped_trf, mapped_idx = trf_norms[aliases[norm_label]]
    elif norm_label == 'lalitpancham':
        status = "AMBIGUOUS"
        reason = "Lalit Pancham contains Pancham which standard Lalit omits. Too ambiguous."
        conf = "LOW"
        mapped_trf, mapped_idx = None, -1
    else:
        status = "NO_MATCH"
        reason = "No matching TRF class found"
        conf = "NONE"
        mapped_trf, mapped_idx = None, -1
            
    mapping.append({
        'saraga_original_label': orig_label,
        'normalized_label': norm_label,
        'trf_label': mapped_trf,
        'trf_class_index': mapped_idx,
        'mapping_status': status,
        'mapping_reason': reason,
        'confidence': conf
    })

df_map = pd.DataFrame(mapping).drop_duplicates(subset=['saraga_original_label'])
df_map.to_csv(os.path.join(EVAL_DIR, "saraga_trf_raag_mapping.csv"), index=False)

eligible_raags = df_map[df_map['mapping_status'].isin(['EXACT_MATCH', 'CANONICAL_ALIAS'])]

print(f"Total SARAGA unique Raags: {len(df_map)}")
print(f"EXACT_MATCH: {len(df_map[df_map['mapping_status'] == 'EXACT_MATCH'])}")
print(f"CANONICAL_ALIAS: {len(df_map[df_map['mapping_status'] == 'CANONICAL_ALIAS'])}")
print(f"Total Eligible Classes: {len(eligible_raags)}")

# Create Saraga Evaluation Manifest
manifest = []
for _, row in df_saraga.iterrows():
    orig_label = str(row['canonical_raga'])
    map_row = df_map[df_map['saraga_original_label'] == orig_label].iloc[0]
    
    eligible = map_row['mapping_status'] in ['EXACT_MATCH', 'CANONICAL_ALIAS']
    reason = "Mapped successfully" if eligible else map_row['mapping_reason']
    
    manifest.append({
        'saraga_recording_id': row['unique_id'],
        'audio_path': row['folder'] + "\\" + row['audio_file'],
        'original_raag': orig_label,
        'mapped_trf_raag': map_row['trf_label'] if eligible else None,
        'mapped_class_index': map_row['trf_class_index'] if eligible else -1,
        'evaluation_eligibility': eligible,
        'exclusion_reason': reason
    })

df_manifest = pd.DataFrame(manifest)
df_manifest.to_csv(os.path.join(EVAL_DIR, "saraga_evaluation_manifest.csv"), index=False)

eligible_recs = len(df_manifest[df_manifest['evaluation_eligibility'] == True])
print(f"Total Eligible SARAGA Recordings: {eligible_recs} / {len(df_manifest)}")

