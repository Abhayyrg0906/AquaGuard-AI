import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

from ml_pipeline.config import get_dataset_config
from ml_pipeline.audit import resolve_coco_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def clip_bbox(bbox: List[float], img_w: int, img_h: int) -> Tuple[List[float], bool, bool]:
    """
    Clips bounding box [x, y, width, height] to image boundaries to repair overflow.
    
    Returns:
        repaired_bbox: The clipped [x, y, width, height] bbox.
        was_repaired: True if the bbox coordinates were adjusted.
        was_discarded: True if the resulting width or height is <= 0.
    """
    if len(bbox) != 4:
        raise ValueError("COCO bbox must have exactly 4 values [x, y, width, height]")
        
    x, y, w, h = bbox
    
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(img_w), float(x + w))
    y2 = min(float(img_h), float(y + h))
    
    new_w = x2 - x1
    new_h = y2 - y1
    
    # Check if discarded
    if new_w <= 0.0 or new_h <= 0.0:
        return [0.0, 0.0, 0.0, 0.0], False, True
        
    # Check if repaired
    was_repaired = (
        abs(x1 - x) > 1e-7 or 
        abs(y1 - y) > 1e-7 or 
        abs(new_w - w) > 1e-7 or 
        abs(new_h - h) > 1e-7
    )
    
    return [x1, y1, new_w, new_h], was_repaired, False

def coco_to_yolo(bbox: List[float], img_w: int, img_h: int, class_id: int = 0) -> List[float]:
    """
    Converts clipped COCO [x, y, width, height] to YOLO normalized [class_id, x_center, y_center, width, height].
    Coordinates are normalized to range [0, 1].
    """
    if len(bbox) != 4:
        raise ValueError("COCO bbox must have exactly 4 values [x, y, width, height]")
        
    x, y, w, h = bbox
    
    # Center relative calculation
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    
    # Normalization
    norm_x = x_center / img_w
    norm_y = y_center / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    # Ensure coordinates are bound within [0.0, 1.0]
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    
    return [float(class_id), norm_x, norm_y, norm_w, norm_h]

def load_coco_data(json_path: Path) -> dict:
    """Parses a COCO JSON file and returns the parsed data dictionary."""
    logger.info(f"Loading COCO data from: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)

def filter_coco_by_category(coco_data: dict, category_id: int) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    """
    Filters COCO images and annotations to contain only references to the given category_id.
    Returns:
        filtered_images: A mapping of image_id -> image_metadata dict.
        filtered_annotations: A mapping of image_id -> list of annotation dicts.
    """
    logger.info(f"Filtering COCO data for Category ID: {category_id}")
    
    # 1. Build a map of all images
    images_map = {img["id"]: img for img in coco_data.get("images", [])}
    
    # 2. Filter annotations by category_id and group by image_id
    filtered_annotations: Dict[int, List[dict]] = {}
    for ann in coco_data.get("annotations", []):
        if ann.get("category_id") == category_id:
            img_id = ann["image_id"]
            filtered_annotations.setdefault(img_id, []).append(ann)
            
    # 3. Keep only images that have at least one matching annotation
    filtered_images: Dict[int, dict] = {}
    for img_id in filtered_annotations.keys():
        if img_id in images_map:
            filtered_images[img_id] = images_map[img_id]
        else:
            logger.warning(f"Annotation refers to image_id {img_id} which is missing from COCO images list.")
            
    logger.info(f"Filtered to {len(filtered_images)} images and {sum(len(v) for v in filtered_annotations.values())} annotations.")
    return filtered_images, filtered_annotations

def validate_prepared_dataset(yolo_dir: Path) -> Dict[str, Any]:
    """
    Validates that the prepared YOLO dataset is complete, consistent, and correctly formatted.
    """
    summary = {}
    for split in ["train", "val"]:
        images_dir = yolo_dir / "images" / split
        labels_dir = yolo_dir / "labels" / split
        
        images = {f.stem: f for f in images_dir.glob("*") if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS}
        labels = {f.stem: f for f in labels_dir.glob("*.txt")}
        
        # Verify 1-to-1 matching
        missing_labels = []
        for stem in sorted(images.keys()):
            if stem not in labels:
                missing_labels.append(images[stem].name)
                
        orphan_labels = []
        for stem in sorted(labels.keys()):
            if stem not in images:
                orphan_labels.append(labels[stem].name)
                
        # Validate label values
        invalid_labels = []
        for stem in sorted(labels.keys()):
            label_file = labels[stem]
            with open(label_file, "r") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_labels.append(f"{label_file.name}:{idx+1} - Malformed line: expected 5 elements, got {len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_labels.append(f"{label_file.name}:{idx+1} - Non-numeric coordinates")
                    continue
                
                if class_id != 0:
                    invalid_labels.append(f"{label_file.name}:{idx+1} - Invalid class ID: expected 0, got {class_id}")
                for coord_idx, val in enumerate(coords):
                    if val < 0.0 or val > 1.0:
                        invalid_labels.append(f"{label_file.name}:{idx+1} - Coordinate out of [0, 1] range: {val}")
                        
        summary[split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "missing_labels_count": len(missing_labels),
            "missing_labels": missing_labels,
            "orphan_labels_count": len(orphan_labels),
            "orphan_labels": orphan_labels,
            "invalid_labels_count": len(invalid_labels),
            "invalid_labels": invalid_labels,
            "is_valid": len(missing_labels) == 0 and len(orphan_labels) == 0 and len(invalid_labels) == 0
        }
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - Dataset Preparation Pipeline"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to the TrashCan dataset root. Overrides the AQUAGUARD_DATASET_ROOT environment variable."
    )
    args = parser.parse_args()

    config = get_dataset_config()
    dataset_root_str = args.dataset_root or config["dataset_root"]
    
    if not dataset_root_str:
        logger.error(
            "Dataset root path is not specified. Provide the path using --dataset-root <path> "
            "or set the AQUAGUARD_DATASET_ROOT environment variable."
        )
        sys.exit(1)

    dataset_root = Path(dataset_root_str)
    
    if not dataset_root.exists() or not dataset_root.is_dir():
        logger.error(f"Provided dataset root path is invalid or does not exist: {dataset_root}")
        sys.exit(1)

    coco_dir = resolve_coco_dir(dataset_root)
    train_json = coco_dir / "instances_train_trashcan.json"
    val_json = coco_dir / "instances_val_trashcan.json"
    train_dir = coco_dir / "train"
    val_dir = coco_dir / "val"

    if (not train_json.exists() or not val_json.exists() or 
        not train_dir.exists() or not train_dir.is_dir() or
        not val_dir.exists() or not val_dir.is_dir()):
        logger.error(
            f"The directory {dataset_root} does not contain the required TrashCan structure.\n"
            f"Expected files/folders inside dataset root or 'material_version':\n"
            f"  - train/\n"
            f"  - val/\n"
            f"  - instances_train_trashcan.json\n"
            f"  - instances_val_trashcan.json"
        )
        sys.exit(1)

    logger.info("Validation successful. Starting dataset preparation...")

    base_dir = Path(__file__).resolve().parent.parent
    yolo_dir = base_dir / "artifacts" / "dataset" / "yolo"
    
    # Clean and re-create output folders
    images_train_out = yolo_dir / "images" / "train"
    images_val_out = yolo_dir / "images" / "val"
    labels_train_out = yolo_dir / "labels" / "train"
    labels_val_out = yolo_dir / "labels" / "val"
    
    for folder in [images_train_out, images_val_out, labels_train_out, labels_val_out]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"]
    
    for split in splits:
        logger.info(f"Preparing split: {split}")
        split_json_path = coco_dir / f"instances_{split}_trashcan.json"
        split_img_src_dir = coco_dir / split
        
        coco_data = load_coco_data(split_json_path)
        filtered_imgs, filtered_anns = filter_coco_by_category(coco_data, category_id=14)
        
        split_dest_img_dir = images_train_out if split == "train" else images_val_out
        split_dest_lbl_dir = labels_train_out if split == "train" else labels_val_out
        
        # Sort image IDs for deterministic processing
        for img_id in sorted(filtered_anns.keys()):
            img_meta = filtered_imgs.get(img_id)
            if not img_meta:
                continue
                
            file_name = img_meta["file_name"]
            img_w = img_meta["width"]
            img_h = img_meta["height"]
            
            src_img_path = split_img_src_dir / file_name
            if not src_img_path.exists():
                logger.warning(f"Source image does not exist: {src_img_path}")
                continue
                
            yolo_lines = []
            for ann in filtered_anns[img_id]:
                bbox = ann["bbox"]
                repaired_bbox, was_repaired, was_discarded = clip_bbox(bbox, img_w, img_h)
                
                if was_discarded:
                    continue
                    
                yolo_box = coco_to_yolo(repaired_bbox, img_w, img_h, class_id=0)
                yolo_lines.append(f"0 {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}")
                
            if yolo_lines:
                dest_img_path = split_dest_img_dir / file_name
                dest_lbl_path = split_dest_lbl_dir / Path(file_name).with_suffix(".txt")
                
                shutil.copy2(src_img_path, dest_img_path)
                with open(dest_lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")

    # Write dataset.yaml
    # Resolved absolute path of the generated dataset root
    resolved_yolo_root = str(yolo_dir.resolve()).replace("\\", "/")
    yaml_content = f"""path: {resolved_yolo_root}
train: images/train
val: images/val

names:
  0: plastic
"""
    with open(yolo_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    logger.info(f"dataset.yaml written to: {yolo_dir / 'dataset.yaml'}")

    # Validate output
    logger.info("Verifying prepared dataset splits and labels...")
    validation_summary = validate_prepared_dataset(yolo_dir)
    
    # Logging validation results
    for split, results in validation_summary.items():
        logger.info(
            f"Split '{split}' validation: {results['image_count']} images, {results['label_count']} labels. "
            f"Is valid: {results['is_valid']}"
        )
        if not results["is_valid"]:
            logger.error(f"Validation failed for split '{split}'. Malformed/missing checks: {results}")
            sys.exit(1)

    logger.info("Dataset preparation pipeline completed successfully.")

if __name__ == "__main__":
    main()
