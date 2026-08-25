import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing yaml, fallback to simple parser if missing
try:
    import yaml
except ImportError:
    yaml = None

# Try importing ultralytics, but don't fail import if missing
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

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
            # Simple line-by-line fallback parser for environments without PyYAML
            data = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Strip quotes if present
                    if v.startswith(("'", '"')) and v.endswith(("'", '"')):
                        v = v[1:-1]
                    # Simple integer or list parser
                    if v.isdigit():
                        data[k] = int(v)
                    elif v.startswith("[") and v.endswith("]"):
                        # Parse simple list
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
    # 1. Check required fields
    for field in ["path", "train", "val"]:
        if field not in yaml_data:
            raise KeyError(f"Missing required field '{field}' in dataset configuration")
            
    base_path = Path(yaml_data["path"])
    
    # Resolve relative paths
    # Ultralytics resolves paths relative to the directory containing dataset.yaml or Cwd
    if not base_path.is_absolute():
        resolved_base = (yaml_path.parent / base_path).resolve()
        if not (resolved_base / yaml_data["train"]).exists():
            resolved_base = (Path(os.getcwd()) / base_path).resolve()
    else:
        resolved_base = base_path.resolve()
        
    train_images = (resolved_base / yaml_data["train"]).resolve()
    val_images = (resolved_base / yaml_data["val"]).resolve()
    
    # YOLO labels directory is parallel to images (replacing 'images' with 'labels')
    train_labels = (resolved_base / yaml_data["train"].replace("images", "labels")).resolve()
    val_labels = (resolved_base / yaml_data["val"].replace("images", "labels")).resolve()
    
    # 2. Check directories existence
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
    """
    Validates that the image files and label files match 1-to-1.
    """
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

def run_training_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Executes the training pipeline including configuration load, checks, and YOLO training."""
    yaml_path = Path(args.dataset)
    yaml_data = load_dataset_yaml(yaml_path)
    
    # Validate paths
    train_img, train_lbl, val_img, val_lbl = validate_dataset_paths(yaml_data, yaml_path)
    
    # Validate consistency
    train_check = validate_image_label_consistency(train_img, train_lbl)
    val_check = validate_image_label_consistency(val_img, val_lbl)
    
    if not train_check["is_consistent"] or not val_check["is_consistent"]:
        err_msg = (
            f"Dataset consistency check failed.\n"
            f"Train: {train_check['total_images']} images, {train_check['total_labels']} labels. "
            f"Missing labels: {train_check['missing_labels_count']}, Orphans: {train_check['orphan_labels_count']}.\n"
            f"Validation: {val_check['total_images']} images, {val_check['total_labels']} labels. "
            f"Missing labels: {val_check['missing_labels_count']}, Orphans: {val_check['orphan_labels_count']}."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
        
    logger.info("Dataset consistency checks passed. Train and Validation sets are consistent.")
    
    if YOLO is None:
        logger.error("Ultralytics YOLO is not installed. Cannot run training.")
        raise ImportError("Ultralytics package is missing. Run 'pip install ultralytics'")
        
    logger.info(f"Initializing YOLO model: {args.model}")
    model = YOLO(args.model)
    
    logger.info(
        f"Starting YOLO baseline training:\n"
        f"  - Dataset config: {yaml_path}\n"
        f"  - Epochs: {args.epochs}\n"
        f"  - Image size: {args.imgsz}\n"
        f"  - Batch size: {args.batch}\n"
        f"  - Device: {args.device or 'auto'}\n"
        f"  - Output dir: {args.project}/{args.name}"
    )
    
    # Convert args for YOLO training
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    
    if args.device:
        train_kwargs["device"] = args.device
        
    results = model.train(**train_kwargs)
    
    # Save training report details
    report_data = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "model_baseline": args.model,
        "dataset_yaml": str(yaml_path),
        "hyperparameters": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device or "auto"
        },
        "dataset_stats": {
            "train_images": train_check["total_images"],
            "val_images": val_check["total_images"]
        },
        "output_directory": f"{args.project}/{args.name}"
    }
    
    return report_data

def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generates the human-readable Markdown training report."""
    md = []
    md.append("# AquaGuard AI - YOLO Training Report\n")
    md.append(f"**Execution Timestamp:** {data.get('timestamp')}\n")
    
    md.append("## 1. Baseline Experiment Configuration")
    md.append(f"- **Baseline Model:** `{data.get('model_baseline')}`")
    md.append(f"- **Dataset Configuration:** `{data.get('dataset_yaml')}`")
    md.append(f"- **Output Directory:** `{data.get('output_directory')}`\n")
    
    md.append("### Hyperparameters")
    hp = data.get("hyperparameters", {})
    md.append(f"- Epochs: {hp.get('epochs')}")
    md.append(f"- Image Size: {hp.get('imgsz')}px")
    md.append(f"- Batch Size: {hp.get('batch')}")
    md.append(f"- Device: {hp.get('device')}\n")
    
    md.append("## 2. Dataset Split Summary")
    ds = data.get("dataset_stats", {})
    md.append(f"- Training Images: {ds.get('train_images')}")
    md.append(f"- Validation Images: {ds.get('val_images')}\n")
    
    md.append("## 3. Results & Output Checkpoints")
    md.append("Training completed successfully. Checkpoints and performance plots are available under:")
    md.append(f"`{data.get('output_directory')}/`")
    md.append(f"- Best model checkpoint: `{data.get('output_directory')}/weights/best.pt`\n")
    
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - YOLO Baseline Training Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/prepared_dataset/dataset.yaml",
        help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Baseline model to load (e.g. yolov8n.pt)"
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
        "--device",
        type=str,
        default="",
        help="Device (e.g. cpu, cuda, 0)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="artifacts/training",
        help="Project output directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Name of the training run experiment"
    )
    
    args = parser.parse_args()
    
    try:
        report_data = run_training_pipeline(args)
        
        # Save training report files
        base_dir = Path(__file__).resolve().parent.parent
        
        report_json_path = base_dir / "artifacts" / "training_report.json"
        report_md_path = base_dir / "artifacts" / "training_report.md"
        
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_json_path, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"JSON training report written to: {report_json_path}")
        
        md_content = generate_markdown_report(report_data)
        with open(report_md_path, "w") as f:
            f.write(md_content)
        logger.info(f"Markdown training report written to: {report_md_path}")
        
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
