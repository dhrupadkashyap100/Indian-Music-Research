
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import pandas as pd
import numpy as np

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")

df = pd.read_csv(os.path.join(WORKSPACE, "FINAL_FOLK_RAAG_DATASET.csv"))
df_failed = pd.read_csv(os.path.join(WORKSPACE, "FOLK_INFERENCE_FAILED_RECORDINGS.csv"))

# 1. Record Count Audit
num_success = len(df)
num_failed = len(df_failed)
total_recs = num_success + num_failed
print(f"Total processed: {num_success}, Failed: {num_failed}, Sum: {total_recs}")

# 2. Master CSV Audit
duplicates = df['recording_id'].duplicated().sum()
nans = df['predicted_raag_top1'].isna().sum()
print(f"Duplicates: {duplicates}, NaNs in Top1: {nans}")

# 3. Top-K Consistency
score_order = (df['hybrid_top1_score'] >= df['hybrid_top2_score']).all()
margin_correct = np.allclose(df['hybrid_score_margin'], df['hybrid_top1_score'] - df['hybrid_top2_score'], atol=1e-5)

def check_tier(r):
    m = r['hybrid_score_margin']
    c = r['confidence_tier']
    if m > 0.12 and c == 'HIGH': return True
    if m > 0.05 and m <= 0.12 and c == 'MODERATE': return True
    if m <= 0.05 and c == 'AMBIGUOUS': return True
    return False

tiers_correct = df.apply(check_tier, axis=1).all()
print(f"Score Order: {score_order}, Margin Correct: {margin_correct}, Tiers Correct: {tiers_correct}")

# 4. Confidence Distribution
conf_counts = df['confidence_tier'].value_counts()
conf_chart = pd.DataFrame({
    'Confidence_Tier': ['HIGH', 'MODERATE', 'AMBIGUOUS'],
    'Count': [
        sum(df['confidence_tier'] == 'HIGH'),
        sum(df['confidence_tier'] == 'MODERATE'),
        sum(df['confidence_tier'] == 'AMBIGUOUS')
    ]
})
conf_chart['Percentage'] = conf_chart['Count'] / num_success * 100
conf_chart.to_csv(os.path.join(WORKSPACE, "FINAL_CHART_CONFIDENCE_DISTRIBUTION.csv"), index=False)

# 5. Raag Distribution
raag_counts = df['predicted_raag_top1'].value_counts()
raag_chart = pd.DataFrame({'Raag': raag_counts.index, 'Count': raag_counts.values})
raag_chart['Percentage'] = raag_chart['Count'] / num_success * 100
raag_chart.to_csv(os.path.join(WORKSPACE, "FINAL_CHART_RAAG_DISTRIBUTION.csv"), index=False)

# 6. Genre x Raag
genre_raag = df.groupby(['folk_genre', 'predicted_raag_top1']).size().reset_index(name='Count')
genre_raag = genre_raag.sort_values(['folk_genre', 'Count'], ascending=[True, False])
genre_raag.to_csv(os.path.join(WORKSPACE, "FINAL_CHART_GENRE_RAAG.csv"), index=False)

# 7. Model Agreement
agree_chart = pd.DataFrame(df['model_agreement'].value_counts()).reset_index()
agree_chart.columns = ['Agreement_Type', 'Count']
agree_chart['Percentage'] = agree_chart['Count'] / num_success * 100
agree_chart.to_csv(os.path.join(WORKSPACE, "FINAL_CHART_MODEL_AGREEMENT.csv"), index=False)

print("Visualization CSVs generated successfully.")
