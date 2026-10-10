
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = os.path.join(PROJECT_ROOT, r"scripts\folk_inference_final")
OUTPUT_DIR = os.path.join(WORKSPACE, "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def plot_raag_distribution():
    df = pd.read_csv(os.path.join(WORKSPACE, "FINAL_CHART_RAAG_DISTRIBUTION.csv"))
    # Top 10 Raags
    df_top = df.head(10)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Count', y='Raag', data=df_top, palette='viridis')
    plt.title('Top 10 Predicted Raags in Indian Folk Music (n=327)')
    plt.xlabel('Number of Recordings')
    plt.ylabel('Predicted Raag Hypothesis')
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_raag_distribution.png"), dpi=300)
    plt.close()

def plot_confidence_distribution():
    df = pd.read_csv(os.path.join(WORKSPACE, "FINAL_CHART_CONFIDENCE_DISTRIBUTION.csv"))
    
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999','#66b3ff','#99ff99']
    # If the order is High, Mod, Amb, we use a specific palette
    # Let's map it safely
    color_map = {'HIGH': '#2ca02c', 'MODERATE': '#ff7f0e', 'AMBIGUOUS': '#d62728'}
    bar_colors = [color_map.get(x, '#333333') for x in df['Confidence_Tier']]
    
    ax = sns.barplot(x='Confidence_Tier', y='Percentage', data=df, palette=bar_colors)
    plt.title('Confidence Distribution (Structural Domain Shift)')
    plt.xlabel('Confidence Tier')
    plt.ylabel('Percentage of Recordings (%)')
    plt.ylim(0, 100)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_confidence_distribution.png"), dpi=300)
    plt.close()

def plot_genre_distribution():
    # We want a heatmap of Top 5 Genres vs Top 5 Raags
    df = pd.read_csv(os.path.join(WORKSPACE, "FINAL_CHART_GENRE_RAAG.csv"))
    
    # Get top 5 genres by total count
    top_genres = df.groupby('folk_genre')['Count'].sum().nlargest(5).index
    # Get top 5 raags by total count
    top_raags = df.groupby('predicted_raag_top1')['Count'].sum().nlargest(5).index
    
    df_filtered = df[df['folk_genre'].isin(top_genres) & df['predicted_raag_top1'].isin(top_raags)]
    
    pivot_df = df_filtered.pivot(index='folk_genre', columns='predicted_raag_top1', values='Count').fillna(0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt='g', cbar_kws={'label': 'Number of Recordings'})
    plt.title('Heatmap of Top 5 Folk Genres vs Top 5 Predicted Raags')
    plt.xlabel('Predicted Raag Hypothesis')
    plt.ylabel('Folk Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_genre_heatmap.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating figures...")
    plot_raag_distribution()
    plot_confidence_distribution()
    plot_genre_distribution()
    print(f"Figures saved to {OUTPUT_DIR}")
