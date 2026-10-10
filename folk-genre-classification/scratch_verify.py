import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

def generate_structure(root_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {".venv", ".git", "__pycache__"}
        
    tree_str = f"Folk_Music_Classifier/\n"
    
    def walk_dir(current_dir, prefix=""):
        try:
            items = sorted(os.listdir(current_dir))
        except PermissionError:
            return
            
        dirs = [i for i in items if os.path.isdir(os.path.join(current_dir, i)) and i not in exclude_dirs]
        files = [i for i in items if os.path.isfile(os.path.join(current_dir, i))]
        
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1 and len(files) == 0)
            connector = "└── " if is_last else "├── "
            tree_str_add = f"{prefix}{connector}{d}/\n"
            nonlocal tree_str
            tree_str += tree_str_add
            new_prefix = prefix + ("    " if is_last else "│   ")
            walk_dir(os.path.join(current_dir, d), new_prefix)
            
        for i, f in enumerate(files):
            is_last = (i == len(files) - 1)
            connector = "└── " if is_last else "├── "
            tree_str += f"{prefix}{connector}{f}\n"
            
    walk_dir(root_dir)
    return tree_str

def check_artifacts():
    artifacts = [
        "results/baseline/best_multi_feature_classifier.pth",
        "results/cbam/best_cbam_classifier.pth",
        "results/reports/cbam_research_report.md",
        "results/reports/cbam_final_engineering_report.md",
        "results/reports/statistical_summary.md",
        "results/figures/accuracy_vs_epoch_comparison.png",
        "results/figures/loss_vs_epoch_comparison.png",
        "results/figures/confusion_matrix.png",
        "results/metrics/cbam_classification_report.json",
        "results/logs/cbam_training_log.csv"
    ]
    
    missing = []
    for art in artifacts:
        if not (PROJECT_ROOT / art).exists():
            missing.append(art)
    
    if missing:
        print("MISSING ARTIFACTS:")
        for m in missing: print(m)
    else:
        print("ALL ARTIFACTS VERIFIED.")

if __name__ == "__main__":
    structure = generate_structure(PROJECT_ROOT)
    with open(PROJECT_ROOT / "PROJECT_STRUCTURE.md", "w", encoding="utf-8") as f:
        f.write("# Project Structure\n\n```text\n")
        f.write(structure)
        f.write("```\n")
    print("Generated PROJECT_STRUCTURE.md")
    
    check_artifacts()
