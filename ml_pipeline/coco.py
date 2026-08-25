import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

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
