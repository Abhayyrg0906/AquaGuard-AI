import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def find_metrics_files(experiments_dir: Path) -> List[Path]:
    """Finds all metrics.json files under the experiments directory."""
    if not experiments_dir.exists():
        return []
    return sorted(list(experiments_dir.glob("**/metrics.json")))

def load_metrics(metrics_paths: List[Path]) -> List[Dict[str, Any]]:
    """Loads all metrics JSON files."""
    results = []
    for path in metrics_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load metrics from {path}: {e}")
    return results

def generate_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Generates a Markdown table comparing all experiment results."""
    if not results:
        return "No experiment results found to compare."
        
    headers = [
        "Experiment Run", "Model", "Epochs", "Batch", "Imgsz", 
        "Optimizer", "Mosaic", "Device", "Duration", 
        "Precision", "Recall", "mAP50", "mAP50-95"
    ]
    
    rows = []
    for r in results:
        cfg = r.get("training_configuration", {})
        
        name = r.get("name", "")
        if not name and r.get("output_directory"):
            name = Path(r["output_directory"]).name
            
        row = [
            f"**{name}**",
            r.get("model_name", cfg.get("model_name", "unknown")),
            str(r.get("epochs", cfg.get("epochs", "unknown"))),
            str(r.get("batch", cfg.get("batch", "unknown"))),
            str(r.get("imgsz", cfg.get("imgsz", "unknown"))),
            str(r.get("optimizer", cfg.get("optimizer", "auto"))),
            str(r.get("mosaic", cfg.get("mosaic", "1.0"))),
            r.get("device", cfg.get("device", "unknown")),
            r.get("duration", "unknown"),
            f"{r.get('precision', 0.0):.4f}",
            f"{r.get('recall', 0.0):.4f}",
            f"{r.get('mAP50', 0.0):.4f}",
            f"{r.get('mAP50-95', 0.0):.4f}"
        ]
        rows.append(row)
        
    md = []
    md.append("# AquaGuard AI - Experiment Comparison Report\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - Experiment Comparison Utility"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="artifacts/training/experiments",
        help="Directory containing the experiment subdirectories with metrics.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/training/experiments/comparison_report.md",
        help="Output path for the Markdown comparison report"
    )
    
    args = parser.parse_args()
    experiments_dir = Path(args.dir)
    
    print(f"Scanning for metrics in: {experiments_dir.resolve()}")
    metrics_files = find_metrics_files(experiments_dir)
    print(f"Found {len(metrics_files)} experiment metric files.")
    
    results = load_metrics(metrics_files)
    md_content = generate_comparison_table(results)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md_content)
        
    print(f"\nSaved comparison report to: {out_path.resolve()}\n")
    print(md_content)

if __name__ == "__main__":
    main()
