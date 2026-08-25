import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

def clip_bbox(bbox: List[float], img_w: int, img_h: int) -> Tuple[List[float], bool, bool]:
    """
    Clips bounding box [x, y, width, height] to image boundaries.
    
    Args:
        bbox: COCO format [x, y, width, height].
        img_w: Image width.
        img_h: Image height.
        
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
