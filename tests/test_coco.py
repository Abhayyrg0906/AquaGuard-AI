import pytest
import json
import tempfile
from pathlib import Path

from ml_pipeline.coco import load_coco_data, filter_coco_by_category

@pytest.fixture
def sample_coco_json():
    """Creates a temporary sample COCO JSON dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "sample_coco.json"
        
        data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "img3.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 101, "image_id": 1, "category_id": 14, "bbox": [100, 100, 50, 50]},
                {"id": 102, "image_id": 1, "category_id": 2, "bbox": [200, 200, 30, 30]},
                {"id": 103, "image_id": 2, "category_id": 14, "bbox": [150, 150, 40, 40]}
            ],
            "categories": [
                {"id": 14, "name": "trash_plastic"},
                {"id": 2, "name": "plant"}
            ]
        }
        
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            
        yield tmp_path

def test_load_coco_data(sample_coco_json):
    coco_data = load_coco_data(sample_coco_json)
    assert "images" in coco_data
    assert "annotations" in coco_data
    assert len(coco_data["images"]) == 3

def test_filter_coco_by_category(sample_coco_json):
    coco_data = load_coco_data(sample_coco_json)
    filtered_imgs, filtered_anns = filter_coco_by_category(coco_data, category_id=14)
    
    # img1 and img2 contain category 14, img3 does not.
    assert len(filtered_imgs) == 2
    assert 1 in filtered_imgs
    assert 2 in filtered_imgs
    assert 3 not in filtered_imgs
    
    assert len(filtered_anns) == 2
    assert len(filtered_anns[1]) == 1
    assert filtered_anns[1][0]["id"] == 101
    assert filtered_anns[2][0]["id"] == 103
