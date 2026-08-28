import os
import sys
import json
import logging
import argparse
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing PyYAML
try:
    import yaml
except ImportError:
    yaml = None

# Try importing torch
try:
    import torch
except ImportError:
    torch = None

# Try importing ultralytics
try:
    from ultralytics import YOLO, __version__ as ultralytics_version
except ImportError:
    YOLO = None
    ultralytics_version = "unknown"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_dataset_yaml(yaml_path: Path) -> dict:
    """Loads and parses the dataset configuration YAML file."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration YAML not found at: {yaml_path}")
        
    logger.info(f"Loading dataset configuration from: {yaml_path}")
    with open(yaml_path, "r") as f:
        if yaml is not None:
            try:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    raise ValueError("YAML content must be a dictionary")
                return data
            except Exception as e:
                raise ValueError(f"Failed to parse YAML: {e}")
        else:
            # Fallback custom parser for environments without PyYAML
            data = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v.startswith(("'", '"')) and v.endswith(("'", '"')):
                        v = v[1:-1]
                    if v.isdigit():
                        data[k] = int(v)
                    elif v.startswith("[") and v.endswith("]"):
                        items = [i.strip().strip("'\"") for i in v[1:-1].split(",")]
                        data[k] = items
                    else:
                        data[k] = v
            return data

def validate_dataset_paths(yaml_data: dict, yaml_path: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Validates and resolves paths defined in the dataset YAML file.
    Returns:
        Tuple of (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)
    """
    for field in ["path", "train", "val"]:
        if field not in yaml_data:
            raise KeyError(f"Missing required field '{field}' in dataset configuration")
            
    base_path = Path(yaml_data["path"])
    
    if not base_path.is_absolute():
        resolved_base = (yaml_path.parent / base_path).resolve()
        if not (resolved_base / yaml_data["train"]).exists():
            resolved_base = (Path(os.getcwd()) / base_path).resolve()
    else:
        resolved_base = base_path.resolve()
        
    train_images = (resolved_base / yaml_data["train"]).resolve()
    val_images = (resolved_base / yaml_data["val"]).resolve()
    
    train_labels = (resolved_base / yaml_data["train"].replace("images", "labels")).resolve()
    val_labels = (resolved_base / yaml_data["val"].replace("images", "labels")).resolve()
    
    for dir_path, dir_name in [
        (train_images, "Train images"),
        (val_images, "Validation images"),
        (train_labels, "Train labels"),
        (val_labels, "Validation labels")
    ]:
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"{dir_name} directory does not exist: {dir_path}")
            
    return train_images, train_labels, val_images, val_labels

def validate_image_label_consistency(images_dir: Path, labels_dir: Path) -> Dict[str, Any]:
    """Validates that the image files and label files match 1-to-1."""
    image_files = {}
    for p in images_dir.glob("*"):
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files[p.stem] = p
            
    label_files = {}
    for p in labels_dir.glob("*.txt"):
        label_files[p.stem] = p
        
    image_stems = set(image_files.keys())
    label_stems = set(label_files.keys())
    
    missing_labels = [str(image_files[stem].name) for stem in sorted(image_stems - label_stems)]
    orphan_labels = [str(label_files[stem].name) for stem in sorted(label_stems - image_stems)]
    
    is_consistent = len(missing_labels) == 0 and len(orphan_labels) == 0
    
    return {
        "is_consistent": is_consistent,
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "missing_labels_count": len(missing_labels),
        "missing_labels": missing_labels,
        "orphan_labels_count": len(orphan_labels),
        "orphan_labels": orphan_labels
    }

def detect_device(requested_device: str = "") -> str:
    """Detects available hardware or validates target device configuration."""
    if torch is None:
        return "cpu"
        
    # Check custom cuda device request
    if requested_device:
        device_lower = requested_device.lower()
        if "cuda" in device_lower or device_lower.isdigit():
            if not torch.cuda.is_available():
                raise ValueError(f"Requested CUDA device '{requested_device}' is not available.")
            return requested_device
        return requested_device
        
    # Auto-detection
    if torch.cuda.is_available():
        return "0"  # default to first cuda gpu
    return "cpu"

def extract_metrics(val_results) -> Dict[str, float]:
    """Safely extracts precision, recall, mAP50, and mAP50-95 from YOLO validation results."""
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0
    }
    if val_results is None:
        return metrics
        
    if hasattr(val_results, "results_dict"):
        d = val_results.results_dict
        metrics["precision"] = d.get("metrics/precision(B)", d.get("metrics/precision", 0.0))
        metrics["recall"] = d.get("metrics/recall(B)", d.get("metrics/recall", 0.0))
        metrics["mAP50"] = d.get("metrics/mAP50(B)", d.get("metrics/mAP50", 0.0))
        metrics["mAP50-95"] = d.get("metrics/mAP50-95(B)", d.get("metrics/mAP50-95", 0.0))
        
    if hasattr(val_results, "box"):
        box = val_results.box
        if box is not None:
            # box.mp (mean precision), box.mr (mean recall), box.map50, box.map
            metrics["precision"] = getattr(box, "mp", metrics["precision"])
            metrics["recall"] = getattr(box, "mr", metrics["recall"])
            metrics["mAP50"] = getattr(box, "map50", metrics["mAP50"])
            metrics["mAP50-95"] = getattr(box, "map", metrics["mAP50-95"])
            
    for k in metrics:
        metrics[k] = float(metrics[k])
        
    return metrics

def run_training_pipeline(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    """Executes dataset verification, YOLO training, validation, and logs reports."""
    # 1. Print and log software/device variables
    python_ver = platform.python_version()
    device = detect_device(args.device)
    
    logger.info("=== Baseline Training Parameters ===")
    logger.info(f"Python Version: {python_ver}")
    logger.info(f"Ultralytics Version: {ultralytics_version}")
    logger.info(f"Selected Device: {device}")
    logger.info(f"Dataset path: {args.data}")
    logger.info(f"Model path: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Batch size: {args.batch}")
    logger.info("====================================")
    
    yaml_path = Path(args.data)
    yaml_data = load_dataset_yaml(yaml_path)
    
    # 2. Validate paths and splits consistency
    train_img, train_lbl, val_img, val_lbl = validate_dataset_paths(yaml_data, yaml_path)
    train_check = validate_image_label_consistency(train_img, train_lbl)
    val_check = validate_image_label_consistency(val_img, val_lbl)
    
    if not train_check["is_consistent"] or not val_check["is_consistent"]:
        err = (
            f"Dataset consistency check failed.\n"
            f"Train: {train_check['total_images']} images, {train_check['total_labels']} labels.\n"
            f"Validation: {val_check['total_images']} images, {val_check['total_labels']} labels."
        )
        logger.error(err)
        raise ValueError(err)
        
    if YOLO is None:
        raise ImportError("Ultralytics package is missing. Run 'pip install ultralytics'")
        
    # Check model weights file path check (unless downloading default yolov8n.pt)
    if args.model != "yolov8n.pt" and not Path(args.model).exists():
        raise FileNotFoundError(f"Selected model file does not exist: {args.model}")
        
    # 3. Train
    logger.info(f"Loading baseline model: {args.model}")
    model = YOLO(args.model)
    
    start_time = time.perf_counter()
    
    # Resolve project path to absolute to avoid relative path resolution issues
    project_abs = Path(args.project).resolve()
    
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_abs),
        "name": args.name,
        "device": device,
        "deterministic": True,
        "exist_ok": True,
        "verbose": False
    }
    if getattr(args, "seed", None) is not None:
        train_kwargs["seed"] = args.seed
    
    logger.info("Executing training...")
    model.train(**train_kwargs)
    
    # 4. Automatically run validation
    logger.info("Executing validation on split...")
    val_results = model.val(data=str(yaml_path), verbose=False)
    
    duration = time.perf_counter() - start_time
    minutes, seconds = divmod(int(duration), 60)
    hours, minutes = divmod(minutes, 60)
    duration_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
    
    metrics = extract_metrics(val_results)
    
    # Derive actual save directory from trainer
    if hasattr(model, "trainer") and model.trainer is not None and getattr(model.trainer, "save_dir", None) is not None:
        run_dir = Path(model.trainer.save_dir).resolve()
    else:
        run_dir = (project_abs / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather environment metadata
    pytorch_version = "unknown"
    if torch is not None:
        pytorch_version = getattr(torch, "__version__", "unknown")
        
    best_model_path = run_dir / "weights" / "best.pt"
    last_model_path = run_dir / "weights" / "last.pt"
    
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Training completed but best checkpoint weights were not found at: {best_model_path}"
        )
    if not last_model_path.exists():
        raise FileNotFoundError(
            f"Training completed but last checkpoint weights were not found at: {last_model_path}"
        )
        
    best_model_path_str = str(best_model_path.resolve()).replace("\\", "/")
    last_model_path_str = str(last_model_path.resolve()).replace("\\", "/")
    
    report_data = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "training_configuration": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "model_name": args.model,
            "device": device,
            "deterministic": True
        },
        "model_name": args.model,
        "dataset_path": str(yaml_path.resolve()).replace("\\", "/"),
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "duration": duration_str,
        "output_directory": str(run_dir.resolve()).replace("\\", "/"),
        
        # Reproducibility / Experiment Metadata
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics_version,
        "pytorch_version": pytorch_version,
        "os_platform": platform.platform(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed if getattr(args, "seed", None) is not None else None,
        "actual_output_directory": str(run_dir.resolve()).replace("\\", "/"),
        "best_checkpoint_path": best_model_path_str,
        "last_checkpoint_path": last_model_path_str
    }
    
    # Save metrics.json
    metrics_json_path = run_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    # Generate Markdown Report
    md_content = f"""# AquaGuard AI - Baseline Training Report

**Report Generation Timestamp:** {report_data["timestamp"]}

## 1. Experiment Configuration
- **Model Baseline:** `{report_data["model_name"]}`
- **Dataset Path:** `{report_data["dataset_path"]}`
- **Hardware/Device:** `{report_data["device"]}`
- **Epochs:** {report_data["training_configuration"]["epochs"]}
- **Image Size:** {report_data["training_configuration"]["imgsz"]}px
- **Batch Size:** {report_data["training_configuration"]["batch"]}
- **Training Duration:** {duration_str}

## 2. Evaluation Metrics (Validation Split)
- **Precision:** {metrics["precision"]:.4f}
- **Recall:** {metrics["recall"]:.4f}
- **mAP@0.5:** {metrics["mAP50"]:.4f}
- **mAP@0.5:0.95:** {metrics["mAP50-95"]:.4f}

## 3. Results & Outputs
- **Best Model Checkpoint:** `{best_model_path_str}`
- **Validation Output Directory:** `{report_data["output_directory"]}`
"""
    md_report_path = run_dir / "training_report.md"
    with open(md_report_path, "w") as f:
        f.write(md_content)
        
    return report_data, md_content

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - YOLO Baseline Training Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="artifacts/yolo_dataset/dataset.yaml",
        help="Path to dataset.yaml configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Lightweight YOLO model suitable for experimentation (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size in pixels"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="artifacts/training",
        help="Output project directory for experiments tracking"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Name of the experiment run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Hardware device (e.g. cpu, cuda, 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic training"
    )
    
    args = parser.parse_args()
    
    try:
        report, _ = run_training_pipeline(args)
        logger.info(f"Training pipeline execution completed successfully. Output saved to: {report['output_directory']}")
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
