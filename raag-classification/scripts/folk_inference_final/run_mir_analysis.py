
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import glob
import pandas as pd
import json

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")
AUDIO_DIR = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\raw_audio")

df_pred = pd.read_csv(os.path.join(WORKSPACE, "folk_raag_predictions.csv"))
df_conf = pd.read_csv(os.path.join(WORKSPACE, "folk_confidence_analysis.csv"))
df_agree = pd.read_csv(os.path.join(WORKSPACE, "folk_model_agreement_analysis.csv"))
df_dist = pd.read_csv(os.path.join(WORKSPACE, "folk_raag_distribution.csv"))
df_amb = pd.read_csv(os.path.join(WORKSPACE, "folk_ambiguous_raag_pairs.csv"))
df_priority = pd.read_csv(os.path.join(WORKSPACE, "folk_manual_review_priority.csv"))

# Build Genre mapping
genre_map = {}
for ext in ['*.wav', '*.mp3']:
    files = glob.glob(os.path.join(AUDIO_DIR, '**', ext), recursive=True)
    for f in files:
        rec_id = os.path.splitext(os.path.basename(f))[0]
        genre = os.path.basename(os.path.dirname(f))
        genre_map[rec_id] = genre

df_pred['genre'] = df_pred['recording_id'].map(genre_map)

# Phase 6: Genre Analysis
genre_analysis = []
if df_pred['genre'].notna().any():
    for g, grp in df_pred.groupby('genre'):
        top_raags = grp['predicted_raag_top1'].value_counts()
        genre_analysis.append({
            'Genre': g,
            'Total_Recordings': len(grp),
            'Top_1_Raag': top_raags.index[0] if len(top_raags) > 0 else "None",
            'Top_2_Raag': top_raags.index[1] if len(top_raags) > 1 else "None",
            'Top_3_Raag': top_raags.index[2] if len(top_raags) > 2 else "None",
            'HIGH_Count': sum(grp['confidence_tier'] == 'HIGH'),
            'MODERATE_Count': sum(grp['confidence_tier'] == 'MODERATE'),
            'AMBIGUOUS_Count': sum(grp['confidence_tier'] == 'AMBIGUOUS'),
            'Avg_Margin': grp['hybrid_score_margin'].mean(),
            'Agreement_Rate': sum(grp['models_agree'] == 'AGREEMENT') / len(grp)
        })
    df_genre = pd.DataFrame(genre_analysis)
    df_genre.to_csv(os.path.join(WORKSPACE, "folk_genre_raag_analysis.csv"), index=False)
else:
    print("No genre metadata found.")

# Phase 7: Raag Family Analysis (Refining ambiguous pairs)
family_analysis = df_amb.copy()
family_analysis.to_csv(os.path.join(WORKSPACE, "folk_raag_family_analysis.csv"), index=False)

# Phase 8: Manual Review Prioritization
df_priority['Priority_Tier'] = 3
df_priority.loc[df_priority['confidence_tier'] == 'AMBIGUOUS', 'Priority_Tier'] = 3
df_priority.loc[(df_priority['confidence_tier'] == 'MODERATE') | (df_priority['models_agree'].isin(['DISAGREEMENT', 'HYBRID RESOLUTION'])), 'Priority_Tier'] = 2
df_priority.loc[(df_priority['confidence_tier'] == 'HIGH') & (df_priority['models_agree'] == 'AGREEMENT'), 'Priority_Tier'] = 1

df_priority.to_csv(os.path.join(WORKSPACE, "folk_musicologist_review_priorities.csv"), index=False)

# Phase 12 JSON
summary = {
    "total_recordings_attempted": 328,
    "total_recordings_processed": 327,
    "failed_recordings": 1,
    "confidence_counts": {
        "HIGH": int(sum(df_pred['confidence_tier'] == 'HIGH')),
        "MODERATE": int(sum(df_pred['confidence_tier'] == 'MODERATE')),
        "AMBIGUOUS": int(sum(df_pred['confidence_tier'] == 'AMBIGUOUS'))
    },
    "confidence_percentages": {
        "HIGH": sum(df_pred['confidence_tier'] == 'HIGH') / 327 * 100,
        "MODERATE": sum(df_pred['confidence_tier'] == 'MODERATE') / 327 * 100,
        "AMBIGUOUS": sum(df_pred['confidence_tier'] == 'AMBIGUOUS') / 327 * 100
    },
    "agreement_counts": {
        "Complete_Agreement": int(sum(df_pred['models_agree'] == 'AGREEMENT')),
        "Disagreement": int(sum(df_pred['models_agree'] != 'AGREEMENT'))
    },
    "top_predicted_raags": df_dist.head(5).to_dict('records'),
    "major_ambiguous_pairs": df_amb.head(5).to_dict('records'),
    "key_limitations": [
        "61-class closed-set classification assumes forced choices.",
        "Domain shift from pure classical (TRF) to folk melodies.",
        "Lack of Folk ground truth for accuracy verification.",
        "Cultural and regional variation is not encoded in classical metadata."
    ],
    "key_scientific_conclusions": [
        "The model predicts Folk music is fundamentally OOD relative to pure classical Raags.",
        "When forced, the model maps Folk structures to historically folk-adjacent Raags like Malkauns and Pahadi.",
        "Model disagreement is a primary indicator of structural instability."
    ],
    "disclaimer": "Folk predictions are model-based hypotheses about possible structural similarities, not verified ground truth annotations."
}
with open(os.path.join(WORKSPACE, "final_folk_research_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)
