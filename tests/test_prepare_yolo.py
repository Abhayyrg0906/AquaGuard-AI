import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

from ml_pipeline.prepare_yolo import (
    clip_bbox,
    coco_to_yolo,
    find_category_id_by_name,
    main
)

def test_clip_bbox():
    # Inside boundaries (no repair)
    bbox = [10.0, 20.0, 100.0, 150.0]
    clipped, was_repaired, was_discarded = clip_bbox(bbox, 640, 480)
    assert clipped == [10.0, 20.0, 100.0, 150.0]
    assert not was_repaired
    assert not was_discarded

    # Over boundary limits (repair required)
    bbox_overflow = [-2.0, 5.0, 650.0, 100.0]
    clipped, was_repaired, was_discarded = clip_bbox(bbox_overflow, 640, 480)
    assert clipped == [0.0, 5.0, 640.0, 100.0]
    assert was_repaired
    assert not was_discarded

    # Fully out of bounds (rejected)
    bbox_out = [645.0, 490.0, 20.0, 20.0]
    clipped, was_repaired, was_discarded = clip_bbox(bbox_out, 640, 480)
    assert was_discarded

    # Negative width (rejected)
    bbox_neg = [10.0, 20.0, -5.0, 15.0]
    clipped, was_repaired, was_discarded = clip_bbox(bbox_neg, 640, 480)
    assert was_discarded

def test_coco_to_yolo():
    # Normal bounds conversion test
    # COCO bbox: [x, y, w, h] = [10, 20, 100, 150] inside 640x480
    # x_center = 10 + 50 = 60 -> 60/640 = 0.09375
    # y_center = 20 + 75 = 95 -> 95/480 = 0.19791667
    yolo_box = coco_to_yolo([10.0, 20.0, 100.0, 150.0], 640, 480, class_id=0)
    assert yolo_box[0] == 0
    assert pytest.approx(yolo_box[1]) == 0.09375
    assert pytest.approx(yolo_box[2]) == 0.19791667
    assert pytest.approx(yolo_box[3]) == 0.15625
    assert pytest.approx(yolo_box[4]) == 0.3125

def test_find_category_id_by_name():
    coco_data = {
        "categories": [
            {"id": 14, "name": "trash_plastic"},
            {"id": 1, "name": "rov"}
        ]
    }
    assert find_category_id_by_name(coco_data, "trash_plastic") == 14
    assert find_category_id_by_name(coco_data, "nonexistent") == 14  # Default fallback

def test_prepare_yolo_end_to_end_mock():
    # Setup mock file structure in temp folder
    with tempfile.TemporaryDirectory() as tmp_dataset_root, tempfile.TemporaryDirectory() as tmp_out_dir:
        root_path = Path(tmp_dataset_root)
        out_path = Path(tmp_out_dir)
        
        # Create splits structure inside mock source
        (root_path / "train").mkdir()
        (root_path / "val").mkdir()
        
        # Touch mock image files
        (root_path / "train" / "img1.jpg").touch()
        (root_path / "val" / "img2.jpg").touch()
        
        # Mock COCO JSONs
        train_coco = {
            "categories": [{"id": 14, "name": "trash_plastic"}],
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 101, "image_id": 1, "category_id": 14, "bbox": [10, 20, 100, 150]}]
        }
        val_coco = {
            "categories": [{"id": 14, "name": "trash_plastic"}],
            "images": [{"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 102, "image_id": 2, "category_id": 14, "bbox": [50, 60, 70, 80]}]
        }
        
        with open(root_path / "instances_train_trashcan.json", "w") as f:
            json.dump(train_coco, f)
        with open(root_path / "instances_val_trashcan.json", "w") as f:
            json.dump(val_coco, f)
            
        test_args = [
            "prepare_yolo.py",
            "--dataset-root", str(root_path),
            "--output-dir", str(out_path)
        ]
        
        with patch.object(sys, "argv", test_args):
            main()
            
        # Verify output structures
        assert (out_path / "images" / "train" / "img1.jpg").exists()
        assert (out_path / "images" / "val" / "img2.jpg").exists()
        assert (out_path / "labels" / "train" / "img1.txt").exists()
        assert (out_path / "labels" / "val" / "img2.txt").exists()
        assert (out_path / "dataset.yaml").exists()
        assert (out_path / "preparation_report.json").exists()
        assert (out_path / "preparation_report.md").exists()
        
        # Read dataset.yaml
        with open(out_path / "dataset.yaml", "r") as f:
            yaml_content = f.read()
        assert "plastic" in yaml_content
        assert "path: " in yaml_content
        
        # Read JSON report
        with open(out_path / "preparation_report.json", "r") as f:
            report = json.load(f)
        assert report["train_image_count"] == 1
        assert report["validation_image_count"] == 1
        assert report["train_plastic_annotation_count"] == 1
        assert report["validation_plastic_annotation_count"] == 1
        assert report["output_image_count"] == 2
