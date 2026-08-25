import pytest
import json
import sys
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from ml_pipeline.audit import (
    calculate_md5,
    verify_image,
    audit_coco_dataset,
    resolve_coco_dir
)
from ml_pipeline.run_audit import generate_markdown_report, main

@pytest.fixture
def synthetic_coco_dataset():
    """Fixture to set up a small synthetic COCO dataset in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create standard TrashCan material-version structure
        mv_dir = tmp_path / "material_version"
        train_dir = mv_dir / "train"
        val_dir = mv_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Create valid dummy image files (1x1 pixel)
        img1_path = train_dir / "img1.jpg"
        img = Image.new("RGB", (480, 360), color="blue")
        img.save(img1_path)

        # Create a duplicate image (same contents, different name)
        img2_path = train_dir / "img2.jpg"
        img.save(img2_path)

        # Valid image in validation split
        img_val_path = val_dir / "val_img1.jpg"
        img_val = Image.new("RGB", (480, 360), color="green")
        img_val.save(img_val_path)

        # Create a corrupt image file (non-image contents)
        corrupted_path = train_dir / "corrupted.jpg"
        with open(corrupted_path, "w") as f:
            f.write("not an image header at all")

        # Define synthetic categories mapping (contains trash_plastic, rov, and an unexpected one)
        categories = [
            {"id": 1, "name": "rov", "supercategory": "rov"},
            {"id": 14, "name": "trash_plastic", "supercategory": "trash_plastic"},
            {"id": 99, "name": "unexpected_class", "supercategory": "unexpected_class"}
        ]

        # Define synthetic train images metadata
        train_images = [
            {"id": 1, "file_name": "img1.jpg", "width": 480, "height": 360},
            {"id": 2, "file_name": "img2.jpg", "width": 480, "height": 360},
            {"id": 3, "file_name": "missing.jpg", "width": 480, "height": 360}, # Missing from disk
            {"id": 4, "file_name": "corrupted.jpg", "width": 480, "height": 360}, # Corrupted
            {"id": 5, "file_name": "img1.jpg", "width": 100, "height": 100}  # Dimension mismatch
        ]

        # Define train annotations with various validation conditions
        train_annotations = [
            # 1. Valid plastic annotation with segmentation
            {
                "id": 101,
                "image_id": 1,
                "category_id": 14,
                "bbox": [10, 20, 50, 60],
                "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]],
                "area": 3000,
                "iscrowd": 0
            },
            # 2. Bounding box out of image bounds (480x360)
            {
                "id": 102,
                "image_id": 1,
                "category_id": 1,
                "bbox": [400, 300, 100, 100],
                "segmentation": [],
                "area": 10000,
                "iscrowd": 0
            },
            # 3. Bounding box with negative dimensions
            {
                "id": 103,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, -5, 10],
                "segmentation": None,
                "area": 0,
                "iscrowd": 0
            },
            # 4. Invalid category ID
            {
                "id": 104,
                "image_id": 2,
                "category_id": 999, # Doesn't exist
                "bbox": [10, 20, 30, 40],
                "segmentation": [],
                "area": 1200,
                "iscrowd": 0
            },
            # 5. Orphan annotation (references non-existent image_id)
            {
                "id": 105,
                "image_id": 999,
                "category_id": 14,
                "bbox": [5, 5, 20, 20],
                "segmentation": [],
                "area": 400,
                "iscrowd": 0
            }
        ]

        # Define validation images metadata
        val_images = [
            {"id": 1, "file_name": "val_img1.jpg", "width": 480, "height": 360}
        ]

        # Define validation annotations
        val_annotations = [
            {
                "id": 201,
                "image_id": 1,
                "category_id": 14,
                "bbox": [5, 5, 10, 10],
                "segmentation": [],
                "area": 100,
                "iscrowd": 0
            }
        ]

        # Write train JSON file
        train_json_data = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": train_images,
            "annotations": train_annotations
        }
        with open(mv_dir / "instances_train_trashcan.json", "w") as f:
            json.dump(train_json_data, f)

        # Write val JSON file
        val_json_data = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": val_images,
            "annotations": val_annotations
        }
        with open(mv_dir / "instances_val_trashcan.json", "w") as f:
            json.dump(val_json_data, f)

        yield tmp_path

def test_calculate_md5(synthetic_coco_dataset):
    img1 = synthetic_coco_dataset / "material_version" / "train" / "img1.jpg"
    img2 = synthetic_coco_dataset / "material_version" / "train" / "img2.jpg"
    
    hash1 = calculate_md5(img1)
    hash2 = calculate_md5(img2)
    
    assert len(hash1) == 32
    assert hash1 == hash2

def test_verify_image_valid(synthetic_coco_dataset):
    img1 = synthetic_coco_dataset / "material_version" / "train" / "img1.jpg"
    is_valid, err_msg, dims, img_fmt = verify_image(img1)
    
    assert is_valid is True
    assert err_msg is None
    assert dims == (480, 360)
    assert img_fmt == "JPEG"

def test_verify_image_corrupted(synthetic_coco_dataset):
    corrupted = synthetic_coco_dataset / "material_version" / "train" / "corrupted.jpg"
    is_valid, err_msg, dims, img_fmt = verify_image(corrupted)
    
    assert is_valid is False
    assert err_msg is not None
    assert dims is None
    assert img_fmt is None

def test_resolve_coco_dir(synthetic_coco_dataset):
    # Should resolve to the material_version folder inside the dataset root
    resolved = resolve_coco_dir(synthetic_coco_dataset)
    assert resolved == synthetic_coco_dataset / "material_version"
    
    # Should fallback to the root if material_version is not there
    assert resolve_coco_dir(Path("C:/nonexistent")) == Path("C:/nonexistent")

def test_audit_coco_dataset(synthetic_coco_dataset):
    res = audit_coco_dataset(synthetic_coco_dataset)
    
    assert res["status"] == "success"
    assert res["health"] == "FAIL" # Since we have missing, corrupt, and out-of-bounds annotations
    
    splits = res["splits"]
    
    # Check train split stats
    train = splits["train"]
    assert train["total_images_in_json"] == 5
    assert train["missing_images_count"] == 1
    assert train["missing_images"] == ["material_version/train/missing.jpg"]
    assert train["corrupted_images_count"] == 1
    assert train["corrupted_images"][0]["file"] == "material_version/train/corrupted.jpg"
    assert train["duplicate_risk_count"] == 1 # img1 and img2 are duplicates
    assert train["dimension_mismatch_count"] == 1 # image 5 has 100x100 metadata but 480x360 actual size
    
    # Category Extraction
    assert train["category_verification"]["unexpected_categories"] == [{"id": 99, "name": "unexpected_class"}]
    # ID 14 is trash_plastic, ID 1 is rov, ID 99 is unexpected. Others in 1-16 are missing.
    assert len(train["category_verification"]["missing_expected_categories"]) == 14
    
    # Plastic category identification (ID 14)
    plastic = train["plastic_stats"]
    assert plastic["category_id"] == 14
    assert plastic["annotations_count"] == 1
    assert plastic["images_containing_plastic"] == 1
    
    # Bounding Box Validation
    # Valid plastic (101) + Out of bounds (102) + Negative (103) + Invalid category (104) + Orphan (105)
    # 102 and 103 are invalid bboxes.
    assert train["invalid_bboxes_count"] == 2
    assert train["invalid_bboxes"][0]["annotation_id"] == 102
    assert "out of image bounds" in train["invalid_bboxes"][0]["reason"]
    assert train["invalid_bboxes"][1]["annotation_id"] == 103
    assert "must be positive" in train["invalid_bboxes"][1]["reason"]
    
    # Orphan annotations detection
    # 105 has image_id 999 which is not in the JSON images.
    # 104 is valid image reference but has invalid category.
    assert train["orphan_annotations_count"] == 1
    assert train["orphan_annotations"][0]["annotation_id"] == 105
    
    # Segmentation presence
    # Only annotation 101 has valid segmentation.
    assert train["segmentation_annotations_count"] == 1

def test_generate_markdown_report():
    data = {
        "health": "WARNING",
        "splits": {
            "train": {
                "total_images_in_json": 100,
                "total_annotations": 200,
                "plastic_stats": {
                    "category_id": 14,
                    "annotations_count": 50,
                    "images_containing_plastic": 45
                },
                "annotations_per_category": {"trash_plastic": 50, "rov": 150},
                "images_per_category": {"trash_plastic": 45, "rov": 80},
                "format_distribution": {"JPEG": 100},
                "resolution_stats": {"unique_count": 1, "min_width": 480, "max_width": 480, "avg_width": 480.0, "min_height": 360, "max_height": 360, "avg_height": 360.0}
            },
            "val": {
                "total_images_in_json": 20,
                "total_annotations": 40,
                "plastic_stats": {
                    "category_id": 14,
                    "annotations_count": 10,
                    "images_containing_plastic": 9
                },
                "annotations_per_category": {"trash_plastic": 10, "rov": 30},
                "images_per_category": {"trash_plastic": 9, "rov": 18},
                "format_distribution": {"JPEG": 20},
                "resolution_stats": {"unique_count": 1, "min_width": 480, "max_width": 480, "avg_width": 480.0, "min_height": 360, "max_height": 360, "avg_height": 360.0}
            }
        }
    }
    
    report_md = generate_markdown_report(data)
    
    assert "Dataset health: WARNING" in report_md
    assert "Train:" in report_md
    assert "100 images" in report_md
    assert "200 annotations" in report_md
    assert "Validation:" in report_md
    assert "20 images" in report_md
    assert "40 annotations" in report_md
    assert "ID 14" in report_md
    assert "50 train annotations" in report_md
    assert "45 train images containing plastic" in report_md

def test_run_audit_invalid_path():
    with pytest.raises(SystemExit) as exc_info:
        with patch.object(sys, "argv", ["run_audit.py", "--dataset-root", "C:/nonexistent_dataset_root_dir"]):
            main()
    assert exc_info.value.code == 1

def test_run_audit_invalid_structure(synthetic_coco_dataset):
    # Remove one of the required JSON files to make the structure invalid
    json_file = synthetic_coco_dataset / "material_version" / "instances_train_trashcan.json"
    json_file.unlink()
    
    with pytest.raises(SystemExit) as exc_info:
        with patch.object(sys, "argv", ["run_audit.py", "--dataset-root", str(synthetic_coco_dataset)]):
            main()
    assert exc_info.value.code == 1
