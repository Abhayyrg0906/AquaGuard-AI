import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from ml_pipeline.config import get_dataset_config
from ml_pipeline.audit import audit_coco_dataset, resolve_coco_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generates the human-readable Markdown report from audit data."""
    md = []
    md.append("# AquaGuard AI - Dataset Audit Report\n")
    md.append("This report contains the static analysis of the dataset configured for the AquaGuard AI project.\n")
    
    md.append("## 1. Summary Statistics")
    md.append(f"Dataset health: {data.get('health', 'Unknown')}\n")
    
    splits = data.get("splits", {})
    
    # Train Split Summary
    train_stats = splits.get("train", {})
    md.append("### Train:")
    md.append(f"{train_stats.get('total_images_in_json', 0)} images")
    md.append(f"{train_stats.get('total_annotations', 0)} annotations\n")
    
    # Validation Split Summary
    val_stats = splits.get("val", {})
    md.append("### Validation:")
    md.append(f"{val_stats.get('total_images_in_json', 0)} images")
    md.append(f"{val_stats.get('total_annotations', 0)} annotations\n")
    
    # Plastic Stats Summary
    train_plastic = train_stats.get("plastic_stats", {})
    md.append("### Plastic:")
    md.append(f"ID {train_plastic.get('category_id', 14)}")
    md.append(f"{train_plastic.get('annotations_count', 0)} train annotations")
    md.append(f"{train_plastic.get('images_containing_plastic', 0)} train images containing plastic\n")
    
    md.append("---")
    md.append("## 2. Detailed Split Statistics")
    
    for name, stats in splits.items():
        md.append(f"### Split: `{name}`")
        md.append(f"- **Images in JSON:** {stats.get('total_images_in_json', 0)}")
        md.append(f"- **Images verified on disk:** {stats.get('images_on_disk', 0)}")
        md.append(f"- **Missing image files:** {stats.get('missing_images_count', 0)}")
        md.append(f"- **Corrupt image files:** {stats.get('corrupted_images_count', 0)}")
        md.append(f"- **Duplicate image files:** {stats.get('duplicate_risk_count', 0)}")
        md.append(f"- **Orphan annotations:** {stats.get('orphan_annotations_count', 0)}")
        md.append(f"- **Bounding Box annotations:** {stats.get('total_annotations', 0)}")
        md.append(f"- **Segmentation annotations:** {stats.get('segmentation_annotations_count', 0)}")
        md.append(f"- **Class Imbalance Ratio (Max/Min):** {stats.get('class_imbalance_ratio', 1.0):.2f}")
        
        md.append("\n#### Class/Category Distribution:")
        md.append("| Category | Annotations Count | Images Count |")
        md.append("| :--- | :--- | :--- |")
        ann_per_cat = stats.get("annotations_per_category", {})
        img_per_cat = stats.get("images_per_category", {})
        for cat, count in ann_per_cat.items():
            img_cnt = img_per_cat.get(cat, 0)
            md.append(f"| {cat} | {count} | {img_cnt} |")
        
        md.append("")
        
        # Format/Resolution
        formats = stats.get("format_distribution", {})
        md.append(f"- **Formats:** " + ", ".join([f"{k}: {v}" for k, v in formats.items()]))
        r_stats = stats.get("resolution_stats", {})
        if r_stats:
            md.append(f"- **Resolutions:** Unique count = {r_stats.get('unique_count')}")
            md.append(f"- **Width Range:** {r_stats.get('min_width')}px to {r_stats.get('max_width')}px (average: {r_stats.get('avg_width'):.1f}px)")
            md.append(f"- **Height Range:** {r_stats.get('min_height')}px to {r_stats.get('max_height')}px (average: {r_stats.get('avg_height'):.1f}px)")
        md.append("")
        
        if stats.get("missing_images_count", 0) > 0:
            md.append("> [!CAUTION]")
            md.append("> **Missing Image Files (first 10 shown):**")
            for path in stats.get("missing_images", [])[:10]:
                md.append(f"> - `{path}`")
            md.append("")
            
        if stats.get("corrupted_images_count", 0) > 0:
            md.append("> [!CAUTION]")
            md.append("> **Corrupted Image Files:**")
            for entry in stats.get("corrupted_images", []):
                md.append(f"> - `{entry.get('file')}`: {entry.get('error')}")
            md.append("")

        if stats.get("invalid_category_ids_count", 0) > 0:
            md.append("> [!CAUTION]")
            md.append("> **Invalid Category ID Annotations:**")
            for entry in stats.get("invalid_category_ids", []):
                md.append(f"> - Annotation ID `{entry.get('annotation_id')}` has Category ID `{entry.get('category_id')}`")
            md.append("")

        if stats.get("invalid_bboxes_count", 0) > 0:
            md.append("> [!CAUTION]")
            md.append("> **Invalid Bounding Box Annotations (first 10 shown):**")
            for entry in stats.get("invalid_bboxes", [])[:10]:
                md.append(f"> - Annotation ID `{entry.get('annotation_id')}` in file `{entry.get('file')}`: {entry.get('reason')}")
            md.append("")
            
    if data.get("global_duplicates_count", 0) > 0:
        md.append("## 3. Global Cross-Split Duplicates")
        md.append(f"Found **{data['global_duplicates_count']}** duplicate image references across splits/folders. A sample of duplicate groups:")
        for group in data.get("global_duplicate_groups", [])[:5]:
            md.append("- Duplicate copies:")
            for path in group:
                md.append(f"  - `{path}`")
        md.append("")

    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - Dataset Audit Pipeline. Audits COCO annotations and image folders of the TrashCan dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to the TrashCan dataset root. Overrides the AQUAGUARD_DATASET_ROOT environment variable."
    )
    args = parser.parse_args()

    config = get_dataset_config()
    
    # Hierarchy to resolve dataset root
    dataset_root_str = args.dataset_root or config["dataset_root"]
    
    if not dataset_root_str:
        logger.error(
            "Dataset root path is not specified. Provide the path using --dataset-root <path> "
            "or set the AQUAGUARD_DATASET_ROOT environment variable."
        )
        sys.exit(1)

    dataset_root = Path(dataset_root_str)
    
    # Validate path exists
    if not dataset_root.exists() or not dataset_root.is_dir():
        logger.error(f"Provided dataset root path is invalid or does not exist: {dataset_root}")
        sys.exit(1)

    # Validate COCO structure (either in root or in 'material_version' subfolder)
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
            f"  - instances_val_trashcan.json\n"
            f"Please double check the directory layout."
        )
        sys.exit(1)

    logger.info("Validation successful. Initiating TrashCan dataset static analysis...")
    
    report_data = audit_coco_dataset(dataset_root)
    
    if report_data.get("status") != "success":
        logger.error(f"Audit pipeline failed: {report_data.get('error')}")
        sys.exit(1)

    # Write report files
    json_out = Path(config["report_json_path"])
    md_out = Path(config["report_md_path"])
    
    # Ensure parent folders exist
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_out, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"JSON report written to: {json_out}")
    
    md_content = generate_markdown_report(report_data)
    with open(md_out, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown report written to: {md_out}")

if __name__ == "__main__":
    main()
