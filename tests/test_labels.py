import pytest

from ml_pipeline.labels import clip_bbox, coco_to_yolo

def test_clip_bbox_no_repair():
    # Box fully inside bounds
    bbox = [10.0, 20.0, 50.0, 60.0]
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 100, 100)
    
    assert repaired == [10.0, 20.0, 50.0, 60.0]
    assert was_repaired is False
    assert was_discarded is False

def test_clip_bbox_repaired_out_of_bounds():
    # Box exceeding W and H
    bbox = [90.0, 80.0, 30.0, 40.0] # exceeds 100x100
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 100, 100)
    
    # Expected: x1=90, y1=80, x2=100, y2=100
    # New box: [90.0, 80.0, 10.0, 20.0]
    assert repaired == [90.0, 80.0, 10.0, 20.0]
    assert was_repaired is True
    assert was_discarded is False

def test_clip_bbox_negative_start():
    # Box starting at negative coordinates
    bbox = [-10.0, -5.0, 50.0, 40.0]
    repaired, was_repaired, was_discarded = clip_bbox(bbox, 100, 100)
    
    # Expected: x1=0, y1=0, x2=40, y2=35
    # New box: [0.0, 0.0, 40.0, 35.0]
    assert repaired == [0.0, 0.0, 40.0, 35.0]
    assert was_repaired is True
    assert was_discarded is False

def test_clip_bbox_zero_or_negative_dims_discarded():
    # Box with width or height <= 0
    bbox1 = [10.0, 10.0, -5.0, 20.0]
    repaired1, was_repaired1, was_discarded1 = clip_bbox(bbox1, 100, 100)
    assert was_discarded1 is True

    # Box starting out of bounds (so new dimensions will be <= 0)
    bbox2 = [110.0, 120.0, 10.0, 10.0]
    repaired2, was_repaired2, was_discarded2 = clip_bbox(bbox2, 100, 100)
    assert was_discarded2 is True

def test_coco_to_yolo_conversion():
    # Image size 640x480
    bbox = [100.0, 120.0, 80.0, 60.0]
    # Center relative: x_center = 100 + 40 = 140, y_center = 120 + 30 = 150
    # Normalized: 140/640 = 0.21875, 150/480 = 0.3125, w = 80/640 = 0.125, h = 60/480 = 0.125
    yolo_box = coco_to_yolo(bbox, 640, 480, class_id=0)
    
    assert yolo_box[0] == 0.0
    assert yolo_box[1] == pytest.approx(0.21875)
    assert yolo_box[2] == pytest.approx(0.3125)
    assert yolo_box[3] == pytest.approx(0.125)
    assert yolo_box[4] == pytest.approx(0.125)

def test_coco_to_yolo_normalization_limits():
    # Coordinates exceeding limits should be clipped to [0, 1]
    # For instance if bounding box math is slightly off, ensure no values are outside [0, 1]
    bbox = [0.0, 0.0, 1000.0, 1000.0]
    yolo_box = coco_to_yolo(bbox, 100, 100, class_id=0)
    
    for val in yolo_box[1:]:
        assert 0.0 <= val <= 1.0

def test_deterministic_processing():
    # Verify that the calculation remains completely stable and identical for multiple runs
    bbox = [45.123456, 78.654321, 12.345678, 56.789012]
    run1 = coco_to_yolo(bbox, 640, 480)
    run2 = coco_to_yolo(bbox, 640, 480)
    assert run1 == run2
