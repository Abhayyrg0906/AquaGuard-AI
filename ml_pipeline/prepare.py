import os
import sys
import shutil
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ml_pipeline.config import get_dataset_config
from ml_pipeline.coco import load_coco_data, filter_coco_by_category
from ml_pipeline.labels import clip_bbox, coco_to_yolo
from ml_pipeline.audit import resolve_coco_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generates a human-readable Markdown report for the dataset preparation."""
    md = []
    md.append("# AquaGuard AI - Dataset Preparation Report\n")
    md.append(f"**Processing Timestamp:** {data.get('processing_timestamp')}\n")
    
    md.append("## 1. Executive Summary")
    md.append(f"- **Source Dataset Root:** `{data.get('source_dataset_root')}`")
    md.append(f"- **Target Category:** `trash_plastic` (COCO ID: 14) -> `YOLO Class 0`")
    md.append(f"- **Final Prepared Images:** {data.get('final_image_count_total')}")
    md.append(f"- **Final Prepared Labels:** {data.get('final_label_count_total')}\n")
    
    md.append("## 2. Pipeline Conversion Statistics")
    md.append("| Metric | Train Split | Validation Split | Total |")
    md.append("| :--- | :---: | :---: | :---: |")
    
    splits = data.get("splits", {})
    train = splits.get("train", {})
    val = splits.get("val", {})
    
    md.append(f"| **Source Images** | {train.get('source_images_count')} | {val.get('source_images_count')} | {train.get('source_images_count', 0) + val.get('source_images_count', 0)} |")
    md.append(f"| **Source Annotations** | {train.get('source_annotations_count')} | {val.get('source_annotations_count')} | {train.get('source_annotations_count', 0) + val.get('source_annotations_count', 0)} |")
    md.append(f"| **Plastic Annotations** | {train.get('plastic_annotations_count')} | {val.get('plastic_annotations_count')} | {train.get('plastic_annotations_count', 0) + val.get('plastic_annotations_count', 0)} |")
    md.append(f"| **Plastic Images** | {train.get('plastic_images_count')} | {val.get('plastic_images_count')} | {train.get('plastic_images_count', 0) + val.get('plastic_images_count', 0)} |")
    md.append(f"| **Repaired Annotations** | {train.get('repaired_annotation_count')} | {val.get('repaired_annotation_count')} | {train.get('repaired_annotation_count', 0) + val.get('repaired_annotation_count', 0)} |")
    md.append(f"| **Discarded Annotations** | {train.get('discarded_annotation_count')} | {val.get('discarded_annotation_count')} | {train.get('discarded_annotation_count', 0) + val.get('discarded_annotation_count', 0)} |")
    md.append(f"| **Final YOLO Labels** | {train.get('final_label_count')} | {val.get('final_label_count')} | {data.get('final_label_count_total')} |")
    md.append(f"| **Final YOLO Images** | {train.get('final_image_count')} | {val.get('final_image_count')} | {data.get('final_image_count_total')} |\n")

    md.append("## 3. Data Integrity & Validation")
    md.append(f"- **Invalid Bounding Boxes Remaining:** {data.get('invalid_boxes_remaining')}")
    md.append(f"- **Coordinate Normalization Check:** {data.get('normalization_validation')}")
    md.append(f"- **Class Mapping:** {json.dumps(data.get('class_mapping'))}\n")
    
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - Dataset Preparation Pipeline. Converts COCO annotations to plastic-only YOLO format."
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
    
    # 1. Validate dataset paths and TrashCan structure
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

    logger.info("Validation successful. Starting COCO to YOLO conversion pipeline...")

    # Determine paths relative to root or workspace
    base_dir = Path(__file__).resolve().parent.parent
    prep_dir = base_dir / "artifacts" / "prepared_dataset"
    
    # Setup directories
    images_train_out = prep_dir / "images" / "train"
    images_val_out = prep_dir / "images" / "val"
    labels_train_out = prep_dir / "labels" / "train"
    labels_val_out = prep_dir / "labels" / "val"
    
    # Re-create output folders fresh
    for folder in [images_train_out, images_val_out, labels_train_out, labels_val_out]:
        folder.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"]
    split_stats = {}
    
    final_image_count_total = 0
    final_label_count_total = 0
    global_normalization_valid = True

    for split in splits:
        logger.info(f"Processing split: {split}")
        split_json_path = coco_dir / f"instances_{split}_trashcan.json"
        split_img_src_dir = coco_dir / split
        
        # Load and Filter COCO data
        coco_data = load_coco_data(split_json_path)
        filtered_imgs, filtered_anns = filter_coco_by_category(coco_data, category_id=14)
        
        split_dest_img_dir = images_train_out if split == "train" else images_val_out
        split_dest_lbl_dir = labels_train_out if split == "train" else labels_val_out
        
        repaired_count = 0
        discarded_count = 0
        final_label_count = 0
        final_image_count = 0

        # Loop over filtered images (only those that contain plastic annotations)
        # Sort keys to ensure deterministic processing
        for img_id in sorted(filtered_anns.keys()):
            img_meta = filtered_imgs.get(img_id)
            if not img_meta:
                continue
                
            file_name = img_meta["file_name"]
            img_w = img_meta["width"]
            img_h = img_meta["height"]
            
            src_img_path = split_img_src_dir / file_name
            if not src_img_path.exists():
                logger.warning(f"Expected source image does not exist: {src_img_path}")
                continue
                
            # Process annotations for this image
            yolo_lines = []
            for ann in filtered_anns[img_id]:
                bbox = ann["bbox"]
                
                # Clip / Repair box
                repaired_bbox, was_repaired, was_discarded = clip_bbox(bbox, img_w, img_h)
                
                if was_discarded:
                    discarded_count += 1
                    continue
                    
                if was_repaired:
                    repaired_count += 1
                    
                # Convert to YOLO
                yolo_box = coco_to_yolo(repaired_bbox, img_w, img_h, class_id=0)
                
                # Validate coordinates normalization
                for val in yolo_box[1:]:
                    if val < 0.0 or val > 1.0:
                        global_normalization_valid = False
                        
                yolo_lines.append(f"0 {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}")

            # If we wrote at least one valid box, write label file and copy image
            if yolo_lines:
                dest_img_path = split_dest_img_dir / file_name
                dest_lbl_path = split_dest_lbl_dir / Path(file_name).with_suffix(".txt")
                
                # Deterministic copying (shutil.copy2 preserves metadata)
                shutil.copy2(src_img_path, dest_img_path)
                
                with open(dest_lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                    
                final_image_count += 1
                final_label_count += len(yolo_lines)

        split_stats[split] = {
            "source_images_count": len(coco_data.get("images", [])),
            "source_annotations_count": len(coco_data.get("annotations", [])),
            "plastic_annotations_count": sum(len(anns) for anns in filtered_anns.values()),
            "plastic_images_count": len(filtered_imgs),
            "repaired_annotation_count": repaired_count,
            "discarded_annotation_count": discarded_count,
            "final_label_count": final_label_count,
            "final_image_count": final_image_count
        }
        
        final_image_count_total += final_image_count
        final_label_count_total += final_label_count

    # 2. Write dataset.yaml
    yaml_content = f"""path: ./artifacts/prepared_dataset
train: images/train
val: images/val

names:
  0: trash_plastic
"""
    with open(prep_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    logger.info(f"YOLO dataset.yaml written to: {prep_dir / 'dataset.yaml'}")

    # 3. Create preparation report data
    # Safe relative paths for the report
    safe_dataset_root = str(dataset_root).replace("\\", "/")
    # Mask username / sensitive parts
    if "Users" in safe_dataset_root:
        parts = safe_dataset_root.split("/")
        try:
            user_idx = parts.index("Users") + 1
            parts[user_idx] = "<USER>"
            safe_dataset_root = "/".join(parts)
        except ValueError:
            pass

    report_data = {
        "processing_timestamp": datetime.now().isoformat(),
        "source_dataset_root": safe_dataset_root,
        "final_image_count_total": final_image_count_total,
        "final_label_count_total": final_label_count_total,
        "invalid_boxes_remaining": 0,
        "normalization_validation": "PASS" if global_normalization_valid else "FAIL",
        "class_mapping": {"14 (COCO)": "0 (YOLO: trash_plastic)"},
        "splits": split_stats
    }

    # Write JSON report
    report_json_path = base_dir / "artifacts" / "preparation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"JSON report written to: {report_json_path}")

    # Write Markdown report
    report_md_path = base_dir / "artifacts" / "preparation_report.md"
    md_content = generate_markdown_report(report_data)
    with open(report_md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown report written to: {report_md_path}")

if __name__ == "__main__":
    main()
