import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

EXPECTED_CATEGORIES = {
    1: "rov",
    2: "plant",
    3: "animal_fish",
    4: "animal_starfish",
    5: "animal_shells",
    6: "animal_crab",
    7: "animal_eel",
    8: "animal_etc",
    9: "trash_etc",
    10: "trash_fabric",
    11: "trash_fishing_gear",
    12: "trash_metal",
    13: "trash_paper",
    14: "trash_plastic",
    15: "trash_rubber",
    16: "trash_wood",
}

def calculate_md5(file_path: Path) -> str:
    """Calculates MD5 hash of a file to detect duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_image(file_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """
    Verifies if an image is valid and reads its dimensions and format.
    Returns (is_valid, error_message, dimensions, format).
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open because verify() closes the file but invalidates the object
        with Image.open(file_path) as img:
            return True, None, img.size, img.format
    except Exception as e:
        return False, str(e), None, None

def make_relative(file_path: Path, base_path: Path) -> str:
    """Helper to convert paths to relative strings with unified forward slashes."""
    try:
        return str(file_path.relative_to(base_path)).replace("\\", "/")
    except ValueError:
        return str(file_path).replace("\\", "/")

def resolve_coco_dir(dataset_root: Path) -> Path:
    """
    Resolves the directory containing TrashCan COCO files.
    Checks inside 'material_version' subdirectory, otherwise uses dataset_root directly.
    """
    mv_dir = dataset_root / "material_version"
    if mv_dir.exists() and mv_dir.is_dir():
        if (mv_dir / "instances_train_trashcan.json").exists():
            return mv_dir
    return dataset_root

def audit_coco_dataset(dataset_root: Path) -> Dict[str, Any]:
    """
    Audits a COCO-formatted dataset (like TrashCan) and checks all requirements:
    category mapping, image referencing, dimensions validation, bounding box limits,
    duplicates, corruption, segmentation presence, split metrics.
    """
    if not dataset_root.exists() or not dataset_root.is_dir():
        logger.error(f"Dataset root does not exist: {dataset_root}")
        return {"status": "error", "error": f"Path {dataset_root} is not a valid directory."}

    coco_dir = resolve_coco_dir(dataset_root)
    logger.info(f"Auditing COCO dataset at: {coco_dir}")

    # We will audit 'train' and 'val' splits
    splits = ["train", "val"]
    split_results = {}
    
    # Store all image hashes globally to detect cross-split duplicates
    global_hashes: Dict[str, List[str]] = {}
    global_has_missing_or_corrupt = False

    for split in splits:
        json_filename = f"instances_{split}_trashcan.json"
        json_path = coco_dir / json_filename
        split_img_dir = coco_dir / split

        if not json_path.exists():
            logger.error(f"Expected annotation file missing: {json_path}")
            return {
                "status": "error",
                "error": f"Missing COCO annotation file {json_filename} in {coco_dir}"
            }
        
        if not split_img_dir.exists() or not split_img_dir.is_dir():
            logger.error(f"Expected image directory missing: {split_img_dir}")
            return {
                "status": "error",
                "error": f"Missing image folder '{split}' in {coco_dir}"
            }

        # Parse COCO JSON
        try:
            with open(json_path, "r") as f:
                coco_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse COCO JSON {json_path}: {e}")
            return {
                "status": "error",
                "error": f"Malformed JSON in {json_filename}: {e}"
            }

        # 1. Categories verification
        coco_categories = coco_data.get("categories", [])
        categories_map = {cat["id"]: cat["name"] for cat in coco_categories}
        
        missing_expected_categories = []
        unexpected_categories = []
        id_mismatches = []
        
        for exp_id, exp_name in EXPECTED_CATEGORIES.items():
            if exp_id not in categories_map:
                missing_expected_categories.append({"id": exp_id, "name": exp_name})
            elif categories_map[exp_id] != exp_name:
                id_mismatches.append({
                    "id": exp_id,
                    "expected_name": exp_name,
                    "found_name": categories_map[exp_id]
                })
        
        for found_id, found_name in categories_map.items():
            if found_id not in EXPECTED_CATEGORIES:
                unexpected_categories.append({"id": found_id, "name": found_name})

        # 2. Images verification
        coco_images = coco_data.get("images", [])
        images_map = {img["id"]: img for img in coco_images}
        
        # Cache to prevent double processing of the same physical file
        processed_paths: Dict[str, Tuple[bool, Optional[Tuple[int, int]], Optional[str]]] = {}
        
        missing_images = []
        corrupted_images = []
        dimension_mismatches = []
        resolutions = []
        formats_counts = {}
        file_hashes = {}
        
        # Track valid image references
        valid_image_ids = set()

        for img in coco_images:
            img_id = img["id"]
            file_name = img["file_name"]
            expected_width = img.get("width")
            expected_height = img.get("height")
            
            img_path = split_img_dir / file_name
            rel_img_path = make_relative(img_path, dataset_root)

            if rel_img_path in processed_paths:
                is_valid, dims, img_fmt = processed_paths[rel_img_path]
                if not is_valid:
                    continue
                valid_image_ids.add(img_id)
                # Compare dimensions
                if expected_width is not None and expected_height is not None and dims:
                    if dims[0] != expected_width or dims[1] != expected_height:
                        dimension_mismatches.append({
                            "file": rel_img_path,
                            "metadata_dims": (expected_width, expected_height),
                            "actual_dims": dims
                        })
                        global_has_missing_or_corrupt = True
                continue

            if not img_path.exists():
                missing_images.append(rel_img_path)
                global_has_missing_or_corrupt = True
                processed_paths[rel_img_path] = (False, None, None)
                continue

            # Verify image file integrity
            is_valid, err_msg, dims, img_fmt = verify_image(img_path)
            if not is_valid:
                corrupted_images.append({"file": rel_img_path, "error": str(err_msg)})
                global_has_missing_or_corrupt = True
                processed_paths[rel_img_path] = (False, None, None)
                continue

            valid_image_ids.add(img_id)
            processed_paths[rel_img_path] = (True, dims, img_fmt)

            if img_fmt:
                formats_counts[img_fmt] = formats_counts.get(img_fmt, 0) + 1
            if dims:
                resolutions.append(dims)
                # Compare dimensions
                if expected_width is not None and expected_height is not None:
                    if dims[0] != expected_width or dims[1] != expected_height:
                        dimension_mismatches.append({
                            "file": rel_img_path,
                            "metadata_dims": (expected_width, expected_height),
                            "actual_dims": dims
                        })
                        global_has_missing_or_corrupt = True

            # Calculate MD5 for duplicates
            try:
                f_hash = calculate_md5(img_path)
                file_hashes.setdefault(f_hash, []).append(rel_img_path)
                global_hashes.setdefault(f_hash, []).append(rel_img_path)
            except Exception as e:
                logger.warning(f"Failed to calculate hash for {img_path}: {e}")

        # Compute duplicate stats for split
        split_duplicates_list = [paths for h, paths in file_hashes.items() if len(paths) > 1]
        split_duplicates_count = sum(len(paths) - 1 for paths in split_duplicates_list)

        # Resolution statistics
        unique_resolutions = set(resolutions)
        res_stats = {}
        if resolutions:
            widths = [r[0] for r in resolutions]
            heights = [r[1] for r in resolutions]
            res_stats = {
                "min_width": min(widths),
                "max_width": max(widths),
                "avg_width": sum(widths) / len(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "avg_height": sum(heights) / len(heights),
                "unique_count": len(unique_resolutions)
            }

        # 3. Annotations verification
        coco_annotations = coco_data.get("annotations", [])
        
        annotations_per_category = {name: 0 for name in EXPECTED_CATEGORIES.values()}
        images_per_category_set = {name: set() for name in EXPECTED_CATEGORIES.values()}
        
        invalid_category_annotations = []
        invalid_bboxes = []
        orphan_annotations = []
        segmentation_annotations_count = 0
        
        for ann in coco_annotations:
            ann_id = ann.get("id")
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            bbox = ann.get("bbox")  # [x, y, width, height]
            segmentation = ann.get("segmentation")

            # Check for orphan annotations (image_id not defined in images map or image file missing)
            if image_id not in images_map:
                orphan_annotations.append({
                    "annotation_id": ann_id,
                    "image_id": image_id,
                    "reason": "image_id not defined in JSON images list"
                })
                continue
            
            img_meta = images_map[image_id]
            file_name = img_meta["file_name"]
            img_path = split_img_dir / file_name
            rel_img_path = make_relative(img_path, dataset_root)
            
            if image_id not in valid_image_ids:
                orphan_annotations.append({
                    "annotation_id": ann_id,
                    "image_id": image_id,
                    "file_name": rel_img_path,
                    "reason": "referenced image is missing or corrupt on disk"
                })
                continue

            # Validate category ID
            if category_id not in EXPECTED_CATEGORIES:
                invalid_category_annotations.append({
                    "annotation_id": ann_id,
                    "category_id": category_id
                })
                global_has_missing_or_corrupt = True
                continue

            cat_name = EXPECTED_CATEGORIES[category_id]
            annotations_per_category[cat_name] += 1
            images_per_category_set[cat_name].add(image_id)

            # Validate bounding box
            if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                invalid_bboxes.append({
                    "annotation_id": ann_id,
                    "file": rel_img_path,
                    "bbox": bbox,
                    "reason": "bbox must be list/tuple of length 4"
                })
                global_has_missing_or_corrupt = True
            else:
                x, y, w, h = bbox
                img_w = img_meta.get("width") or 0
                img_h = img_meta.get("height") or 0
                
                # Check for positive dimensions
                if w <= 0 or h <= 0:
                    invalid_bboxes.append({
                        "annotation_id": ann_id,
                        "file": rel_img_path,
                        "bbox": bbox,
                        "reason": f"bbox width ({w}) and height ({h}) must be positive"
                    })
                    global_has_missing_or_corrupt = True
                # Check within image boundaries
                elif x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                    invalid_bboxes.append({
                        "annotation_id": ann_id,
                        "file": rel_img_path,
                        "bbox": bbox,
                        "reason": f"bbox [{x}, {y}, {w}, {h}] out of image bounds ({img_w}x{img_h})"
                    })
                    global_has_missing_or_corrupt = True

            # Inspect segmentation
            if segmentation and (isinstance(segmentation, list) and len(segmentation) > 0 or isinstance(segmentation, dict)):
                segmentation_annotations_count += 1

        # Count images containing each category
        images_per_category = {k: len(v) for k, v in images_per_category_set.items()}

        # 4. Orphan images on disk check
        referenced_files = {img["file_name"] for img in coco_images}
        orphan_images_on_disk = []
        for ext in SUPPORTED_FORMATS:
            for p in split_img_dir.glob(f"*{ext}"):
                if p.name not in referenced_files:
                    orphan_images_on_disk.append(make_relative(p, dataset_root))

        # Plastic stats (category ID 14)
        plastic_name = EXPECTED_CATEGORIES[14]
        plastic_stats = {
            "category_id": 14,
            "annotations_count": annotations_per_category.get(plastic_name, 0),
            "images_containing_plastic": images_per_category.get(plastic_name, 0)
        }

        # Class imbalance calculation
        imbalance_ratio = 1.0
        active_counts = [count for count in annotations_per_category.values() if count > 0]
        if active_counts:
            imbalance_ratio = max(active_counts) / min(active_counts)

        split_results[split] = {
            "total_images_in_json": len(coco_images),
            "total_annotations": len(coco_annotations),
            "images_on_disk": len(valid_image_ids) + len(corrupted_images),
            "missing_images_count": len(missing_images),
            "missing_images": missing_images,
            "corrupted_images_count": len(corrupted_images),
            "corrupted_images": corrupted_images,
            "duplicate_risk_count": split_duplicates_count,
            "duplicate_groups": split_duplicates_list,
            "resolution_stats": res_stats,
            "format_distribution": formats_counts,
            "annotations_per_category": annotations_per_category,
            "images_per_category": images_per_category,
            "plastic_stats": plastic_stats,
            "segmentation_annotations_count": segmentation_annotations_count,
            "invalid_category_ids_count": len(invalid_category_annotations),
            "invalid_category_ids": invalid_category_annotations,
            "invalid_bboxes_count": len(invalid_bboxes),
            "invalid_bboxes": invalid_bboxes,
            "dimension_mismatch_count": len(dimension_mismatches),
            "dimension_mismatches": dimension_mismatches,
            "orphan_annotations_count": len(orphan_annotations),
            "orphan_annotations": orphan_annotations,
            "orphan_images_on_disk_count": len(orphan_images_on_disk),
            "orphan_images_on_disk": orphan_images_on_disk,
            "class_imbalance_ratio": imbalance_ratio,
            "category_verification": {
                "missing_expected_categories": missing_expected_categories,
                "unexpected_categories": unexpected_categories,
                "id_mismatches": id_mismatches
            }
        }

    # Cross-split duplicate verification
    global_duplicates_list = [paths for h, paths in global_hashes.items() if len(paths) > 1]
    global_duplicates_count = sum(len(paths) - 1 for paths in global_duplicates_list)

    # Determine dataset health
    # HEALTH: PASS, WARNING, or FAIL
    # FAIL if we have missing files, corrupt files, invalid categories, or invalid bounding boxes.
    # WARNING if we have duplicates, dimension mismatch warnings, or orphan images/annotations.
    # PASS otherwise.
    health = "PASS"
    if global_has_missing_or_corrupt:
        health = "FAIL"
    else:
        # Check if any split has missing/corrupted files or invalid bounding boxes
        for split, res in split_results.items():
            if (res["missing_images_count"] > 0 or 
                res["corrupted_images_count"] > 0 or 
                res["invalid_category_ids_count"] > 0 or 
                res["invalid_bboxes_count"] > 0):
                health = "FAIL"
                break
        
        if health != "FAIL":
            for split, res in split_results.items():
                if (res["duplicate_risk_count"] > 0 or 
                    res["orphan_images_on_disk_count"] > 0 or 
                    res["orphan_annotations_count"] > 0 or 
                    res["dimension_mismatch_count"] > 0 or
                    len(res["category_verification"]["missing_expected_categories"]) > 0 or
                    len(res["category_verification"]["id_mismatches"]) > 0):
                    health = "WARNING"
                    break
            
            if health != "WARNING" and global_duplicates_count > 0:
                health = "WARNING"

    return {
        "status": "success",
        "health": health,
        "splits": split_results,
        "global_duplicates_count": global_duplicates_count,
        "global_duplicate_groups": global_duplicates_list
    }
