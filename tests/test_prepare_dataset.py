import pytest
import tempfile
from pathlib import Path
from ml_pipeline.prepare_dataset import (
    clip_bbox,
    coco_to_yolo,
    filter_coco_by_category,
    validate_prepared_dataset
)

def test_clip_bbox_no_repair():
    # Normal coordinates inside image limits (640x480)
    bbox = [10.0, 20.0, 100.0, 150.0]
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 640, 480)
    assert repaired == [10.0, 20.0, 100.0, 150.0]
    assert not was_repaired
    assert not was_discarded

def test_clip_bbox_with_repair():
    # Floating overflow box that exceeds image limits
    bbox = [-1.0, 2.0, 650.0, 100.0]  # width goes to 649, exceeds 640
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 640, 480)
    # Expected clipped to [0, 2, 640, 100] -> x1=0, y1=2, x2=min(640, 649)=640 -> w = 640 - 0 = 640
    assert repaired == [0.0, 2.0, 640.0, 100.0]
    assert was_repaired
    assert not was_discarded

def test_clip_bbox_invalid_discarded():
    # Negative dimensions or coordinate fully out of bounds resulting in width/height <= 0
    bbox = [650.0, 500.0, 10.0, 10.0]  # fully out of bounds
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 640, 480)
    assert was_discarded

    bbox_neg = [10.0, 20.0, -5.0, 15.0]  # negative width
    repaired, was_repaired, was_discarded = clip_bbox(bbox_neg, 640, 480)
    assert was_discarded

def test_coco_to_yolo_conversion():
    # Convert COCO [10, 20, 100, 150] in 640x480 to YOLO normalized
    # COCO: [x, y, w, h]
    # YOLO: [class_id, x_center, y_center, w, h] normalized
    # x_center = 10 + 100/2 = 60 -> 60/640 = 0.09375
    # y_center = 20 + 150/2 = 95 -> 95/480 = 0.19791667
    # w = 100/640 = 0.15625
    # h = 150/480 = 0.3125
    yolo_box = coco_to_yolo([10.0, 20.0, 100.0, 150.0], 640, 480, class_id=0)
    assert yolo_box[0] == 0.0
    assert pytest.approx(yolo_box[1]) == 0.09375
    assert pytest.approx(yolo_box[2]) == 0.19791667
    assert pytest.approx(yolo_box[3]) == 0.15625
    assert pytest.approx(yolo_box[4]) == 0.3125

def test_filter_coco_by_category():
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 14, "bbox": [10, 20, 30, 40]},
            {"id": 102, "image_id": 2, "category_id": 99, "bbox": [50, 60, 70, 80]}
        ]
    }
    
    filtered_imgs, filtered_anns = filter_coco_by_category(coco_data, category_id=14)
    assert len(filtered_imgs) == 1
    assert 1 in filtered_imgs
    assert len(filtered_anns) == 1
    assert 1 in filtered_anns
    assert filtered_anns[1][0]["id"] == 101

def test_validate_prepared_dataset():
    # Setup a dummy structured directory using tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create directories
        for split in ["train", "val"]:
            (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
            
        # 1. Valid case
        # Create a matching image and label file
        img_file = tmp_path / "images" / "train" / "test_img.jpg"
        img_file.touch()
        lbl_file = tmp_path / "labels" / "train" / "test_img.txt"
        with open(lbl_file, "w") as f:
            f.write("0 0.500000 0.500000 0.200000 0.200000\n")
            
        summary = validate_prepared_dataset(tmp_path)
        assert summary["train"]["is_valid"]
        assert summary["train"]["image_count"] == 1
        assert summary["train"]["label_count"] == 1
        
        # 2. Missing label file case
        missing_lbl_img = tmp_path / "images" / "train" / "no_lbl.jpg"
        missing_lbl_img.touch()
        summary = validate_prepared_dataset(tmp_path)
        assert not summary["train"]["is_valid"]
        assert summary["train"]["missing_labels_count"] == 1
        assert "no_lbl.jpg" in summary["train"]["missing_labels"]
        
        # Cleanup missing label image
        missing_lbl_img.unlink()
        
        # 3. Invalid class ID label file case
        invalid_lbl = tmp_path / "labels" / "train" / "invalid_class.txt"
        img_for_invalid = tmp_path / "images" / "train" / "invalid_class.jpg"
        img_for_invalid.touch()
        with open(invalid_lbl, "w") as f:
            f.write("1 0.500000 0.500000 0.200000 0.200000\n")  # Expected class 0
        summary = validate_prepared_dataset(tmp_path)
        assert not summary["train"]["is_valid"]
        assert summary["train"]["invalid_labels_count"] == 1
