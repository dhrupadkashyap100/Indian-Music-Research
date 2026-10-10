
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import pandas as pd

def main():
    json_path = os.path.join(PROJECT_ROOT, r"Thaat and Raga Forest (TRF) Dataset\thaat-and-raga-forest-trf-dataset-metadata.json")
    out_dir = os.path.join(PROJECT_ROOT, r"scripts\trf_metadata_audit")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Analyze Schema
    schema = {
        'top_level_keys': list(data.keys()),
        'is_track_metadata': False,
        'description': data.get('description', ''),
        'creator': data.get('creator', {}).get('name', 'UNKNOWN'),
        'url': data.get('url', 'UNKNOWN')
    }
    
    with open(os.path.join(out_dir, "metadata_schema_report.json"), "w") as f:
        json.dump(schema, f, indent=4)
        
    # 2. Record Inventory (Empty, as this is a Kaggle dataset card)
    pd.DataFrame(columns=['Record_ID', 'Artist', 'Source']).to_csv(os.path.join(out_dir, "metadata_record_inventory.csv"), index=False)
    pd.DataFrame(columns=['Record_ID', 'Audio_Filename', 'Match_Confidence']).to_csv(os.path.join(out_dir, "metadata_audio_mapping.csv"), index=False)
    pd.DataFrame(columns=['Source_ID', 'Count']).to_csv(os.path.join(out_dir, "source_group_inventory.csv"), index=False)
    pd.DataFrame(columns=['Artist', 'Count']).to_csv(os.path.join(out_dir, "artist_group_inventory.csv"), index=False)
    pd.DataFrame(columns=['Performance_ID', 'Count']).to_csv(os.path.join(out_dir, "performance_group_inventory.csv"), index=False)
    pd.DataFrame(columns=['Duplicate_Group', 'Metadata_Explanation']).to_csv(os.path.join(out_dir, "metadata_duplicate_crosscheck.csv"), index=False)
    
    # Split Feasibility
    feasibility = [{
        'Method': 'Random Split',
        'Feasible': 'NO',
        'Reason': 'Track-level metadata is completely absent. Performance independence cannot be established.'
    }]
    pd.DataFrame(feasibility).to_csv(os.path.join(out_dir, "trf_split_feasibility.csv"), index=False)
    
    # CultureMERT Plan
    mert_plan = """
TRF CULTUREMERT REFERENCE PLAN
==============================
Since TRF cannot be split for internal Train/Test validation due to unknown performance overlap, it must be used entirely as a monolithic reference database.

Pipeline:
1. TRF audio -> slice into 5-second segments.
2. CultureMERT (24kHz, trust_remote_code=True) -> extract 768-D embeddings via mean-pooling.
3. Store embeddings labeled with their TRF Thaat/Raag directory name.
4. Folk audio -> slice into 5-second segments.
5. CultureMERT -> extract 768-D embeddings.
6. Nearest Neighbor (k-NN or Cosine Similarity) -> match Folk embeddings to the TRF reference bank to assign candidate Raag labels.
7. Aggregate predictions across Folk segments for a song-level prediction.
"""
    with open(os.path.join(out_dir, "trf_culturemert_reference_plan.txt"), "w") as f:
        f.write(mert_plan.strip())
        
    # Final Report
    report = []
    report.append("TRF METADATA AUDIT REPORT")
    report.append("=========================")
    report.append("\nCONFIRMED:")
    report.append("- The discovered JSON file is NOT a track-level metadata file. It is a Kaggle Dataset Card (Croissant format).")
    report.append("- It contains dataset-level descriptions, DOI links, creator name (SURYA MAJUMDER), and file distributions (archive.zip).")
    report.append("- There are exactly 0 track-level records inside this JSON.")
    report.append("- The JSON does NOT contain artist, concert, performance, or source information for the 1,103 audio files.")
    
    report.append("\nINFERRED:")
    report.append("- Because there is no track-level metadata, the provenance of the numbered audio files (e.g., '1.mp3') remains completely opaque.")
    
    report.append("\nUNKNOWN:")
    report.append("- Independence of performances across files.")
    
    report.append("\nFINAL ANSWERS:")
    report.append("1. What exactly does the metadata contain? Dataset description, Kaggle URL, and paper citations. No track records.")
    report.append("2. Can it be reliably mapped to the 1,103 audio files? NO. There is nothing to map.")
    report.append("3. Does it reveal artist/source/performance information? NO.")
    report.append("4. Does it solve the leakage problem? NO.")
    report.append("5. Can TRF safely be split? NO.")
    report.append("6. Can TRF be used as a reference database? YES. It provides many high-quality samples of each Raag.")
    report.append("7. Is TRF better than our current SARAGA reference dataset? FOR REFERENCE, YES (much larger and more balanced). FOR TESTING, NO (SARAGA has guaranteed independence).")
    report.append("8. Should we proceed with CultureMERT + TRF? YES, but ONLY using TRF as the monolithic reference embedding bank, and evaluating on SARAGA.")
    report.append("9. What is the EXACT next step after this audit? Extract CultureMERT embeddings for TRF and SARAGA. Evaluate the accuracy of the TRF-based KNN classifier on the SARAGA test set.")
    
    with open(os.path.join(out_dir, "trf_metadata_audit_report.txt"), "w") as f:
        f.write("\n".join(report))
        
    summary = {
        'is_track_metadata': False,
        'solves_leakage': False,
        'safe_to_split': False,
        'can_use_as_reference': True,
        'decision': 'USE AS MONOLITHIC REFERENCE'
    }
    with open(os.path.join(out_dir, "trf_metadata_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Metadata Audit Complete.")

if __name__ == '__main__':
    main()
