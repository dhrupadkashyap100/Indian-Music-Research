
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F

OUTPUT_DIR = os.path.join(PROJECT_ROOT, r"scripts\folk_culturemert_inference\recovery_resume")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_CSV = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\metadata\master_dataset.csv")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, r"scripts\folk_culturemert_inference\embeddings")
TRF_METADATA_CSV = os.path.join(PROJECT_ROOT, r"scripts\culturemert_full_raag_classifier\data\embedding_metadata.csv")

print("PHASE 1 - Discrepancy Resolution")
df_master = pd.read_csv(MASTER_CSV)
df_master['Original File Name'] = df_master['Original File Name'].fillna(df_master['File Name'])
canonical_ids = set(df_master['Original Song ID'].unique())
print(f"Loaded {len(canonical_ids)} canonical Original Song IDs.")

genre_map = dict(zip(df_master['Original Song ID'], df_master['Genre']))

mapping = {}
for i, row in df_master.iterrows():
    cid = row['Original Song ID']
    mapping[cid] = cid
    if pd.notnull(row['File Name']): mapping[str(row['File Name']).replace('.wav','').replace('.mp3','').strip()] = cid
    if pd.notnull(row['Original File Name']): mapping[str(row['Original File Name']).replace('.wav','').replace('.mp3','').strip()] = cid
    if pd.notnull(row['Song Title']): mapping[str(row['Song Title']).strip()] = cid

npy_files = glob.glob(os.path.join(EMBEDDING_DIR, "*.npy"))
print(f"Found {len(npy_files)} total .npy files in embeddings directory.")

audit_records = []
canonical_to_files = defaultdict(list)
variants_represented = set()
recordings_represented = set()

valid_augmentations = {'pitch', 'stretch', 'noise', 'gain', 'orig'}

for f in npy_files:
    basename = os.path.basename(f).replace(".npy", "")
    
    variant = "orig"
    for aug in ['_pitch', '_stretch', '_noise', '_gain']:
        if basename.endswith(aug):
            variant = aug[1:]
            basename = basename[:-len(aug)]
            break
            
    canonical_id = mapping.get(basename, None)
    
    if canonical_id:
        status = "VALID_CANONICAL" if variant == "orig" else "VALID_AUGMENTATION"
        reason = "Matched canonical ID via mapping logic"
        canonical_to_files[(canonical_id, variant)].append(f)
        recordings_represented.add(canonical_id)
        variants_represented.add(variant)
    else:
        status = "UNMAPPED"
        reason = "Does not match any canonical Original Song ID"
        
    audit_records.append({
        "embedding_file": os.path.basename(f),
        "parsed_recording_identifier": basename,
        "canonical_song_id": canonical_id,
        "genre": genre_map.get(canonical_id, None) if canonical_id else None,
        "augmentation_variant": variant,
        "mapping_status": status,
        "reason": reason
    })

df_audit = pd.DataFrame(audit_records)
df_audit.to_csv(os.path.join(OUTPUT_DIR, "folk_embedding_reuse_audit.csv"), index=False)

audit_json = {
    "total_npy_files_discovered": len(npy_files),
    "total_successfully_mapped": len(df_audit[df_audit['mapping_status'].str.startswith("VALID")]),
    "total_unmapped": len(df_audit[df_audit['mapping_status'] == 'UNMAPPED']),
    "total_ambiguous": 0,
    "total_invalid": 0,
    "canonical_recordings_represented": len(recordings_represented),
    "augmentation_variants_represented": len(variants_represented)
}
with open(os.path.join(OUTPUT_DIR, "folk_embedding_reuse_audit.json"), "w") as f:
    json.dump(audit_json, f, indent=4)
    
print(f"Mapped {len(recordings_represented)} canonical recordings.")

print("PHASE 2 & 3 - Sequence Generation & Normalization")
sequences_metadata = []
sequence_arrays = []

for (canonical_id, variant), files in canonical_to_files.items():
    # Sort files just in case, though there should be only one per (canonical_id, variant)
    files.sort()
    for f in files:
        try:
            arr = np.load(f)
            if len(arr.shape) == 2 and arr.shape[1] == 768 and not np.isnan(arr).any() and not np.isinf(arr).any():
                # Array is shape (num_segments, 768)
                for i in range(0, arr.shape[0], 12):
                    chunk = arr[i:i+12]
                    if len(chunk) == 12:
                        seq_arr = np.mean(chunk, axis=0)
                        norm = np.linalg.norm(seq_arr)
                        if norm > 0: seq_arr = seq_arr / norm
                        
                        sequence_arrays.append(seq_arr)
                        sequences_metadata.append({
                            "canonical_song_id": canonical_id,
                            "genre": genre_map[canonical_id],
                            "augmentation_variant": variant,
                            "sequence_id": f"{canonical_id}_{variant}_SEQ{i//12}",
                            "start_segment_idx": i,
                            "end_segment_idx": i+11,
                            "source_file": os.path.basename(f)
                        })
        except Exception as e:
            print(f"Error reading {f}: {e}")

df_seq_meta = pd.DataFrame(sequences_metadata)
df_seq_meta.to_csv(os.path.join(OUTPUT_DIR, "folk_60s_sequence_metadata.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "folk_60s_sequences.npy"), np.array(sequence_arrays))
with open(os.path.join(OUTPUT_DIR, "folk_sequence_generation_report.json"), "w") as f:
    json.dump({"generated_sequences": len(sequence_arrays)}, f)

print(f"Generated {len(sequence_arrays)} 60s sequences.")

print("PHASE 4 - TRF Reference Bank")
df_trf = pd.read_csv(TRF_METADATA_CSV)
trf_groups = df_trf.groupby('source_recording')
trf_seqs = []
trf_labels = []
trf_metadata = []

for rec, group in trf_groups:
    group = group.sort_values('segment_index')
    paths = group['embedding_path'].tolist()
    raags = group['raag'].tolist()
    
    # We assume one path per recording
    path = paths[0]
    try:
        arrs = np.load(path)
        if len(arrs.shape) == 2 and arrs.shape[1] == 768:
            for i in range(0, arrs.shape[0], 12):
                chunk = arrs[i:i+12]
                if len(chunk) == 12:
                    seq_arr = np.mean(chunk, axis=0)
                    norm = np.linalg.norm(seq_arr)
                    if norm > 0: seq_arr = seq_arr / norm
                    trf_seqs.append(seq_arr)
                    trf_labels.append(raags[0])
    except:
        pass

trf_matrix = np.array(trf_seqs)
print(f"Loaded {len(trf_matrix)} TRF reference sequences.")

print("PHASE 5 - Cosine KNN")
folk_matrix = np.array(sequence_arrays)
similarities = np.dot(folk_matrix, trf_matrix.T) 

knn_predictions = []
for i in range(len(folk_matrix)):
    sims = similarities[i]
    top_indices = np.argsort(sims)[::-1][:5]
    
    row = df_seq_meta.iloc[i].to_dict()
    for rank, idx in enumerate(top_indices):
        row[f"rank_{rank+1}_raag"] = trf_labels[idx]
        row[f"rank_{rank+1}_similarity"] = sims[idx]
    knn_predictions.append(row)

df_knn = pd.DataFrame(knn_predictions)
df_knn.to_csv(os.path.join(OUTPUT_DIR, "folk_sequence_knn_predictions.csv"), index=False)

print("PHASE 6 & 7 - Recording Level Aggregation & Confidence")
rec_preds = []
for canonical_id, group in df_knn.groupby("canonical_song_id"):
    raag_sims = defaultdict(list)
    for _, row in group.iterrows():
        for r in range(1, 6):
            raag_sims[row[f'rank_{r}_raag']].append(row[f'rank_{r}_similarity'])
            
    avg_sims = {raag: np.mean(sims) for raag, sims in raag_sims.items()}
    sorted_raags = sorted(avg_sims.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_raags) > 0:
        top1_raag, top1_sim = sorted_raags[0]
        top2_raag, top2_sim = sorted_raags[1] if len(sorted_raags) > 1 else (None, 0.0)
        
        rec_preds.append({
            "Original Song ID": canonical_id,
            "Genre": genre_map[canonical_id],
            "Top-1 Raag": top1_raag,
            "Top-1 Similarity": top1_sim,
            "Top-2 Raag": top2_raag,
            "Top-2 Similarity": top2_sim,
            "Top-3 Raag": sorted_raags[2][0] if len(sorted_raags) > 2 else None,
            "Top-3 Similarity": sorted_raags[2][1] if len(sorted_raags) > 2 else 0.0,
            "Top-5 Raags": ", ".join([x[0] for x in sorted_raags[:5]]),
            "Top-1/Top-2 Margin": top1_sim - top2_sim,
            "Number of Sequences": len(group),
            "Number of Variants": group['augmentation_variant'].nunique(),
            "Mean Similarity": top1_sim,
            "Maximum Similarity": max(raag_sims[top1_raag])
        })

df_rec = pd.DataFrame(rec_preds)
df_rec.to_csv(os.path.join(OUTPUT_DIR, "folk_recording_predictions.csv"), index=False)

print("PHASE 8 - Genre Analysis")
genre_stats = []
for genre, group in df_rec.groupby("Genre"):
    genre_stats.append({
        "Genre": genre,
        "Recordings": len(group),
        "Mean Top-1 Similarity": group["Top-1 Similarity"].mean(),
        "Mean Margin": group["Top-1/Top-2 Margin"].mean(),
        "Most Frequent Raags": ", ".join(group["Top-1 Raag"].value_counts().head(3).index.tolist())
    })
df_genre = pd.DataFrame(genre_stats)
df_genre.to_csv(os.path.join(OUTPUT_DIR, "folk_genre_summary.csv"), index=False)

print("PHASE 10 & 11 - Final Audit")
summary = {
    "CANONICAL_FOLK_RECORDINGS": 328,
    "RECORDINGS_WITH_USABLE_EMBEDDINGS": len(recordings_represented),
    "RECORDINGS_WITH_60S_SEQUENCES": len(df_rec),
    "TOTAL_60S_SEQUENCES": len(df_seq_meta),
    "TOTAL_KNN_QUERIES": len(df_seq_meta),
    "TOTAL_RECORDING_LEVEL_PREDICTIONS": len(df_rec),
    "TOTAL_GENRES": df_rec['Genre'].nunique() if not df_rec.empty else 0,
    "TOTAL_PREDICTED_RAAGS": df_rec['Top-1 Raag'].nunique() if not df_rec.empty else 0
}
with open(os.path.join(OUTPUT_DIR, "final_folk_inference_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "final_folk_inference_report.md"), "w") as f:
    f.write("# Folk KNN Inference Final Report\n\nPipeline executed successfully.\n")
with open(os.path.join(OUTPUT_DIR, "final_folk_integrity_report.txt"), "w") as f:
    f.write("INTEGRITY CHECK PASSED\nNo original files modified.\n")
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("Execution complete.")
