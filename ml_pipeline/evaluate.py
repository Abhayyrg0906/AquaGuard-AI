import os
import sys
import json
import argparse
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import dependencies safely
try:
    from ultralytics import YOLO, __version__ as ultralytics_version
except ImportError:
    YOLO = None
    ultralytics_version = "unknown"

try:
    import torch
except ImportError:
    torch = None

from ml_pipeline.train import extract_metrics, load_dataset_yaml, detect_device

def run_evaluation_pipeline(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    """Executes model validation and serializes metrics reports."""
    model_path = Path(args.model)
    yaml_path = Path(args.data)
    
    # 1. Path existence validations
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint weights file not found at: {model_path}")
        
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration YAML not found at: {yaml_path}")
        
    if YOLO is None:
        raise ImportError("Ultralytics package is missing. Run 'pip install ultralytics'")
        
    # 2. Hardware device detection
    device = detect_device(args.device)
    
    logger.info("=== YOLO Model Evaluation ===")
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Dataset path: {yaml_path}")
    logger.info(f"Selected Device: {device}")
    logger.info("=============================")
    
    # Load dataset details (throws error if invalid)
    _ = load_dataset_yaml(yaml_path)
    
    # 3. Load YOLO model and execute validation
    logger.info(f"Loading model checkpoint: {model_path}")
    model = YOLO(model_path)
    
    logger.info("Running validation on split...")
    val_results = model.val(
        data=str(yaml_path),
        device=device,
        verbose=False
    )
    
    # Extract validation split metrics
    metrics = extract_metrics(val_results)
    
    # Print/log the results
    logger.info("=== Evaluation Metrics Summary ===")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    logger.info("==================================")
    
    pytorch_version = "unknown"
    if torch is not None:
        pytorch_version = getattr(torch, "__version__", "unknown")
        
    report_data = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "model_path": str(model_path.resolve()).replace("\\", "/"),
        "dataset_path": str(yaml_path.resolve()).replace("\\", "/"),
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics_version,
        "pytorch_version": pytorch_version,
        "os_platform": platform.platform()
    }
    
    md_content = f"""# AquaGuard AI - Model Evaluation Report

**Evaluation Timestamp:** {report_data["timestamp"]}

## 1. Evaluation Configuration
- **Model Path:** `{report_data["model_path"]}`
- **Dataset Configuration:** `{report_data["dataset_path"]}`
- **Hardware/Device:** `{report_data["device"]}`

## 2. Evaluation Metrics (Validation Split)
- **Precision:** {metrics["precision"]:.4f}
- **Recall:** {metrics["recall"]:.4f}
- **mAP@0.5:** {metrics["mAP50"]:.4f}
- **mAP@0.5:0.95:** {metrics["mAP50-95"]:.4f}
"""
    
    # 4. Optional outputs serialization
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_json_path = out_dir / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"JSON evaluation report saved to: {metrics_json_path}")
        
        md_report_path = out_dir / "evaluation_report.md"
        with open(md_report_path, "w") as f:
            f.write(md_content)
        logger.info(f"Markdown evaluation report saved to: {md_report_path}")
        
    return report_data, md_content

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - YOLO Model Evaluation Utility"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint weights file (.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="artifacts/yolo_dataset/dataset.yaml",
        help="Path to dataset.yaml configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Hardware device to run evaluation on (e.g. cpu, cuda)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional directory to save metrics.json and evaluation_report.md files"
    )
    
    args = parser.parse_args()
    
    try:
        run_evaluation_pipeline(args)
    except Exception as e:
        logger.error(f"Evaluation utility failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
