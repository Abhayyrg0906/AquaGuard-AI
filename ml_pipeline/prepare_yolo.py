import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def clip_bbox(bbox: List[float], img_w: int, img_h: int) -> Tuple[List[float], bool, bool]:
    """
    Clips bounding box coordinates [x, y, width, height] to image boundary limits [0, img_w] and [0, img_h].
    
    Returns:
        clipped_bbox: The corrected [x, y, width, height] bbox.
        was_repaired: True if bounding box values were changed.
        was_discarded: True if width or height <= 0 after clipping.
    """
    if len(bbox) != 4:
        raise ValueError("COCO bbox must contain exactly 4 values")
        
    x, y, w, h = bbox
    
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(img_w), float(x + w))
    y2 = min(float(img_h), float(y + h))
    
    new_w = x2 - x1
    new_h = y2 - y1
    
    if new_w <= 0.0 or new_h <= 0.0:
        return [0.0, 0.0, 0.0, 0.0], False, True
        
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
    Normalizes coordinates relative to image width and height to range [0, 1].
    """
    if len(bbox) != 4:
        raise ValueError("COCO bbox must contain exactly 4 values")
        
    x, y, w, h = bbox
    
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    
    norm_x = max(0.0, min(1.0, x_center / img_w))
    norm_y = max(0.0, min(1.0, y_center / img_h))
    norm_w = max(0.0, min(1.0, w / img_w))
    norm_h = max(0.0, min(1.0, h / img_h))
    
    return [float(class_id), norm_x, norm_y, norm_w, norm_h]

def copy_if_different(src: Path, dst: Path):
    """Copies source file to destination only if different (size mismatch or missing)."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)

def write_if_different(content: str, dst: Path):
    """Writes text content to destination only if different or missing."""
    if dst.exists():
        with open(dst, "r") as f:
            existing = f.read()
        if existing == content:
            return
    with open(dst, "w") as f:
        f.write(content)

def find_category_id_by_name(coco_data: dict, name: str = "trash_plastic") -> int:
    """Finds COCO category ID matching name dynamically, falling back to 14 if missing."""
    for cat in coco_data.get("categories", []):
        if cat.get("name") == name:
            return cat.get("id")
    logger.warning(f"Category name '{name}' not found dynamically in COCO categories. Falling back to ID 14.")
    return 14

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - COCO to YOLO Preparation Pipeline"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the TrashCan dataset root directory containing material_version/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/yolo_dataset",
        help="Target output directory for the prepared YOLO dataset"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dir)
    
    # 1. Validate source path layout
    material_version_dir = dataset_root / "material_version"
    if not material_version_dir.exists():
        material_version_dir = dataset_root
        
    train_json_path = material_version_dir / "instances_train_trashcan.json"
    val_json_path = material_version_dir / "instances_val_trashcan.json"
    train_img_dir = material_version_dir / "train"
    val_img_dir = material_version_dir / "val"
    
    if (not train_json_path.exists() or not val_json_path.exists() or
        not train_img_dir.exists() or not train_img_dir.is_dir() or
        not val_img_dir.exists() or not val_img_dir.is_dir()):
        logger.error(
            f"Dataset root is missing required TrashCan split layout or files at: {dataset_root}\n"
            f"Please ensure it contains instances_train_trashcan.json, instances_val_trashcan.json, train/ and val/."
        )
        sys.exit(1)
        
    logger.info("Source dataset validated. Starting YOLO conversion pipeline...")
    
    # Create target folders
    images_train_out = output_root / "images" / "train"
    images_val_out = output_root / "images" / "val"
    labels_train_out = output_root / "labels" / "train"
    labels_val_out = output_root / "labels" / "val"
    
    for folder in [images_train_out, images_val_out, labels_train_out, labels_val_out]:
        folder.mkdir(parents=True, exist_ok=True)
        
    # Metrics trackers
    train_imgs_total = 0
    val_imgs_total = 0
    train_plastic_anns = 0
    val_plastic_anns = 0
    total_converted_anns = 0
    total_clipped_anns = 0
    total_rejected_anns = 0
    
    # Process splits
    splits = [
        ("train", train_json_path, train_img_dir, images_train_out, labels_train_out),
        ("val", val_json_path, val_img_dir, images_val_out, labels_val_out)
    ]
    
    for split_name, json_path, img_src_dir, img_dest_dir, lbl_dest_dir in splits:
        logger.info(f"Processing split: {split_name}")
        with open(json_path, "r") as f:
            coco_data = json.load(f)
            
        if split_name == "train":
            train_imgs_total = len(coco_data.get("images", []))
        else:
            val_imgs_total = len(coco_data.get("images", []))
            
        # Discover category ID
        plastic_cat_id = find_category_id_by_name(coco_data, "trash_plastic")
        
        # Build image metadata maps and annotations maps
        images_map = {img["id"]: img for img in coco_data.get("images", [])}
        anns_map = {}
        for ann in coco_data.get("annotations", []):
            if ann.get("category_id") == plastic_cat_id:
                img_id = ann["image_id"]
                anns_map.setdefault(img_id, []).append(ann)
                if split_name == "train":
                    train_plastic_anns += 1
                else:
                    val_plastic_anns += 1
                    
        # Process images containing plastic (sorted for deterministic iteration)
        for img_id in sorted(anns_map.keys()):
            img_meta = images_map.get(img_id)
            if not img_meta:
                continue
                
            file_name = img_meta["file_name"]
            img_w = img_meta["width"]
            img_h = img_meta["height"]
            
            src_img_path = img_src_dir / file_name
            if not src_img_path.exists():
                logger.warning(f"Source image not found: {src_img_path}")
                continue
                
            yolo_lines = []
            for ann in anns_map[img_id]:
                bbox = ann["bbox"]
                clipped, was_repaired, was_discarded = clip_bbox(bbox, img_w, img_h)
                
                if was_discarded:
                    total_rejected_anns += 1
                    continue
                    
                if was_repaired:
                    total_clipped_anns += 1
                    
                yolo_box = coco_to_yolo(clipped, img_w, img_h, class_id=0)
                yolo_lines.append(f"0 {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}")
                total_converted_anns += 1
                
            if yolo_lines:
                # Copy image file idempotently
                dest_img_path = img_dest_dir / file_name
                copy_if_different(src_img_path, dest_img_path)
                
                # Write label text file idempotently
                dest_lbl_path = lbl_dest_dir / Path(file_name).with_suffix(".txt")
                content_str = "\n".join(yolo_lines) + "\n"
                write_if_different(content_str, dest_lbl_path)
                
    # Count final files inside target splits
    final_train_imgs = len([f for f in images_train_out.glob("*") if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
    final_val_imgs = len([f for f in images_val_out.glob("*") if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
    final_train_lbls = len(list(labels_train_out.glob("*.txt")))
    final_val_lbls = len(list(labels_val_out.glob("*.txt")))
    
    # Save dataset.yaml
    resolved_output_root = str(output_root.resolve()).replace("\\", "/")
    yaml_content = f"""path: {resolved_output_root}
train: images/train
val: images/val

names:
  0: plastic
"""
    write_if_different(yaml_content, output_root / "dataset.yaml")
    
    # Build report dict
    report_data = {
        "source_dataset": str(dataset_root.resolve()).replace("\\", "/"),
        "train_image_count": train_imgs_total,
        "validation_image_count": val_imgs_total,
        "train_plastic_annotation_count": train_plastic_anns,
        "validation_plastic_annotation_count": val_plastic_anns,
        "converted_annotation_count": total_converted_anns,
        "clipped_annotation_count": total_clipped_anns,
        "rejected_annotation_count": total_rejected_anns,
        "output_image_count": final_train_imgs + final_val_imgs,
        "output_label_count": final_train_lbls + final_val_lbls
    }
    
    # Write JSON report
    report_json_path = output_root / "preparation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"JSON summary report written to: {report_json_path}")
    
    # Write Markdown report
    md_content = f"""# AquaGuard AI - YOLO Dataset Preparation Summary

**Source Dataset:** `{report_data["source_dataset"]}`

## 1. Split Totals
- **Training Images Total:** {report_data["train_image_count"]}
- **Validation Images Total:** {report_data["validation_image_count"]}
- **Training Plastic Annotations:** {report_data["train_plastic_annotation_count"]}
- **Validation Plastic Annotations:** {report_data["validation_plastic_annotation_count"]}

## 2. Bounding Box Correction Summary
- **Converted Annotations:** {report_data["converted_annotation_count"]}
- **Clipped Boundary Overflow Annotations:** {report_data["clipped_annotation_count"]}
- **Discarded Invalid Annotations:** {report_data["rejected_annotation_count"]}

## 3. Final YOLO Target Output
- **Target Output Directory:** `{resolved_output_root}`
- **Output Images Count:** {report_data["output_image_count"]}
- **Output Labels Count:** {report_data["output_label_count"]}
"""
    write_if_different(md_content, output_root / "preparation_report.md")
    logger.info(f"Markdown summary report written to: {output_root / 'preparation_report.md'}")
    logger.info("COCO to YOLO dataset preparation completed successfully.")

if __name__ == "__main__":
    main()
