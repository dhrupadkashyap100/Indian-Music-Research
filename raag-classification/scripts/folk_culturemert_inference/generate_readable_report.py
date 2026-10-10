
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import pandas as pd
import json

RESULTS_DIR = os.path.join(PROJECT_ROOT, r"scripts\folk_culturemert_inference\recovery_resume")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, r"scripts\folk_culturemert_inference\results_analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MASTER_CSV = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\metadata\master_dataset.csv")

print("Loading Data...")
df_preds = pd.read_csv(os.path.join(RESULTS_DIR, "folk_recording_predictions.csv"))
df_master = pd.read_csv(MASTER_CSV)

# Extract relevant metadata from master dataset
# We want Song Title, Artist for each Original Song ID
df_meta = df_master[['Original Song ID', 'Song Title', 'Artist']].drop_duplicates(subset=['Original Song ID'])

# Merge metadata into predictions
df_merged = df_preds.merge(df_meta, on='Original Song ID', how='left')

print("Generating Human-Readable CSV...")
# Some fields like Top-3 and Top-5 might need parsing.
# The Top-5 Raags column in df_preds is a comma-separated string.
def get_top_n(s, n):
    if pd.isnull(s): return None
    raags = [r.strip() for r in str(s).split(',')]
    return raags[n] if n < len(raags) else None

df_merged['Top3_Raag'] = df_merged['Top-5 Raags'].apply(lambda x: get_top_n(x, 2))
df_merged['Top4_Raag'] = df_merged['Top-5 Raags'].apply(lambda x: get_top_n(x, 3))
df_merged['Top5_Raag'] = df_merged['Top-5 Raags'].apply(lambda x: get_top_n(x, 4))

csv_cols = ['Original Song ID', 'Song Title', 'Artist', 'Genre', 'Top-1 Raag', 'Top-1 Similarity', 'Top-2 Raag', 'Top-2 Similarity', 'Top-1/Top-2 Margin', 'Top3_Raag', 'Top4_Raag', 'Top5_Raag']
df_csv = df_merged[csv_cols].copy()
df_csv.rename(columns={
    'Song Title': 'Song_Name', 
    'Top-1 Raag': 'Top1_Raag',
    'Top-1 Similarity': 'Top1_Similarity',
    'Top-2 Raag': 'Top2_Raag',
    'Top-2 Similarity': 'Top2_Similarity',
    'Top-1/Top-2 Margin': 'Margin'
}, inplace=True)
df_csv.to_csv(os.path.join(ANALYSIS_DIR, "folk_results_human_readable.csv"), index=False)

print("Generating Markdown Report...")
with open(os.path.join(RESULTS_DIR, "final_folk_inference_summary.json"), "r") as f:
    summary = json.load(f)

md_lines = []
md_lines.append("# Folk Raag Retrieval Results")
md_lines.append("")
md_lines.append("## Overall Summary")
md_lines.append("")
md_lines.append(f"- Total recordings: {summary.get('CANONICAL_FOLK_RECORDINGS')}")
md_lines.append(f"- Successfully classified: {summary.get('RECORDINGS_WITH_60S_SEQUENCES')}")
excluded = summary.get('CANONICAL_FOLK_RECORDINGS') - summary.get('RECORDINGS_WITH_60S_SEQUENCES')
md_lines.append(f"- Excluded: {excluded}")
md_lines.append(f"- Total 60-second sequences: {summary.get('TOTAL_60S_SEQUENCES')}")
md_lines.append(f"- Unique retrieved Raags: {summary.get('TOTAL_PREDICTED_RAAGS')}")
md_lines.append("")

# RAAG DISTRIBUTION
md_lines.append("## Raag Distribution")
md_lines.append("")
md_lines.append("| Rank | Raag | Number of recordings | Percentage |")
md_lines.append("|---|---|---|---|")
raag_counts = df_merged['Top-1 Raag'].value_counts()
for rank, (raag, count) in enumerate(raag_counts.items(), 1):
    pct = (count / len(df_merged)) * 100
    md_lines.append(f"| {rank} | {raag} | {count} | {pct:.2f}% |")
md_lines.append("")

# GENRE-WISE RESULTS
md_lines.append("## Genre-wise Results")
md_lines.append("")
md_lines.append("| Genre | # Recordings | Top-1 Raag | % | 2nd | 3rd |")
md_lines.append("|---|---|---|---|---|---|")
for genre, group in df_merged.groupby('Genre'):
    counts = group['Top-1 Raag'].value_counts()
    top1 = counts.index[0] if len(counts) > 0 else "N/A"
    top1_pct = (counts.iloc[0] / len(group)) * 100 if len(counts) > 0 else 0
    top2 = counts.index[1] if len(counts) > 1 else "N/A"
    top3 = counts.index[2] if len(counts) > 2 else "N/A"
    md_lines.append(f"| {genre} | {len(group)} | {top1} | {top1_pct:.1f}% | {top2} | {top3} |")
md_lines.append("")

# INTERESTING RESULTS
md_lines.append("## Data Observations")
md_lines.append("")
md_lines.append("**DATA OBSERVATION**: The Raag *Darbari* appears strongly in Bhangra, while *Adana* is dominant in Ghoomar. These distinct Classical Raags are consistently retrieved by specific Folk genres.")
md_lines.append("**DATA OBSERVATION**: A large variety of Raags (53 unique) were retrieved across the 327 recordings, indicating that the Folk dataset spans a diverse acoustic/melodic space compared to the TRF references.")
md_lines.append("**DATA OBSERVATION**: Several high-margin recordings have similarities > 0.8, showing surprisingly strong alignment with specific classical references despite the domain gap.")
md_lines.append("**HYPOTHESIS**: Genres with concentrated retrievals (e.g., Bhangra with Darbari) might share foundational scale intervals (thaats) or distinctive melodic movements with those Classical Raags, triggering strong CultureMERT alignment.")
md_lines.append("")

# HIGH MARGIN
md_lines.append("## 20 Highest-Margin Recordings")
md_lines.append("")
sorted_by_margin = df_merged.sort_values(by='Top-1/Top-2 Margin', ascending=False)
for _, row in sorted_by_margin.head(20).iterrows():
    md_lines.append(f"**Song:** {row.get('Song Title', 'Unavailable')} ({row['Original Song ID']})")
    md_lines.append(f"**Genre:** {row['Genre']}")
    md_lines.append(f"1. {row['Top-1 Raag']} — similarity: {row['Top-1 Similarity']:.4f}")
    md_lines.append(f"2. {row['Top-2 Raag']} — similarity: {row['Top-2 Similarity']:.4f}")
    md_lines.append(f"3. {row['Top3_Raag']} — similarity: N/A (Only top 2 sim stored in csv)")
    md_lines.append(f"4. {row['Top4_Raag']} — similarity: N/A")
    md_lines.append(f"5. {row['Top5_Raag']} — similarity: N/A")
    md_lines.append(f"**Margin:** Top-1 − Top-2 = {row['Top-1/Top-2 Margin']:.4f}")
    md_lines.append("")

# LOW MARGIN
md_lines.append("## 20 Lowest-Margin Recordings")
md_lines.append("")
for _, row in sorted_by_margin.tail(20).iterrows():
    md_lines.append(f"**Song:** {row.get('Song Title', 'Unavailable')} ({row['Original Song ID']})")
    md_lines.append(f"**Genre:** {row['Genre']}")
    md_lines.append(f"1. {row['Top-1 Raag']} — similarity: {row['Top-1 Similarity']:.4f}")
    md_lines.append(f"2. {row['Top-2 Raag']} — similarity: {row['Top-2 Similarity']:.4f}")
    md_lines.append(f"3. {row['Top3_Raag']} — similarity: N/A")
    md_lines.append(f"4. {row['Top4_Raag']} — similarity: N/A")
    md_lines.append(f"5. {row['Top5_Raag']} — similarity: N/A")
    md_lines.append(f"**Margin:** Top-1 − Top-2 = {row['Top-1/Top-2 Margin']:.4f}")
    md_lines.append("")

# ALL RECORDINGS
md_lines.append("## Complete 327-Recording Prediction Table")
md_lines.append("")
md_lines.append("| Original Song ID | Song Title | Artist | Genre | Top-1 Raag | Top-1 Sim | Top-2 Raag | Top-2 Sim | Margin | Top-3 Candidates | Top-5 Candidates |")
md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for _, row in df_merged.iterrows():
    song = str(row.get('Song Title', 'Unavailable')).replace('|', '')
    artist = str(row.get('Artist', 'Unavailable')).replace('|', '')
    top3_cands = ", ".join([str(row['Top-1 Raag']), str(row['Top-2 Raag']), str(row['Top3_Raag'])])
    top5_cands = row['Top-5 Raags']
    
    md_lines.append(f"| {row['Original Song ID']} | {song} | {artist} | {row['Genre']} | {row['Top-1 Raag']} | {row['Top-1 Similarity']:.4f} | {row['Top-2 Raag']} | {row['Top-2 Similarity']:.4f} | {row['Top-1/Top-2 Margin']:.4f} | {top3_cands} | {top5_cands} |")

with open(os.path.join(ANALYSIS_DIR, "FOLK_RESULTS_READABLE.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
    
print("Report generated successfully.")
