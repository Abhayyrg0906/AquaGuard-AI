import pytest
import tempfile
import json
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_pipeline.train import (
    load_dataset_yaml,
    validate_dataset_paths,
    validate_image_label_consistency,
    run_training_pipeline,
    generate_markdown_report
)

@pytest.fixture
def temp_dataset_yaml():
    """Fixture to create a temporary dataset.yaml and directory structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        yaml_file = tmp_path / "dataset.yaml"
        
        # Write yaml content
        yaml_content = f"""path: {tmp_path.as_posix()}
train: images/train
val: images/val
names:
  0: trash_plastic
"""
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
            
        # Create directories
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        yield yaml_file

def test_load_dataset_yaml(temp_dataset_yaml):
    data = load_dataset_yaml(temp_dataset_yaml)
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"][0] == "trash_plastic"

def test_load_dataset_yaml_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_dataset_yaml(Path("C:/nonexistent_file_path_xyz.yaml"))

def test_validate_dataset_paths(temp_dataset_yaml):
    yaml_data = load_dataset_yaml(temp_dataset_yaml)
    train_img, train_lbl, val_img, val_lbl = validate_dataset_paths(yaml_data, temp_dataset_yaml)
    
    assert train_img == temp_dataset_yaml.parent / "images" / "train"
    assert train_lbl == temp_dataset_yaml.parent / "labels" / "train"
    assert val_img == temp_dataset_yaml.parent / "images" / "val"
    assert val_lbl == temp_dataset_yaml.parent / "labels" / "val"

def test_validate_dataset_paths_missing(temp_dataset_yaml):
    yaml_data = load_dataset_yaml(temp_dataset_yaml)
    
    # Remove validation folder to force an error
    val_dir = temp_dataset_yaml.parent / "images" / "val"
    val_dir.rmdir()
    
    with pytest.raises(FileNotFoundError):
        validate_dataset_paths(yaml_data, temp_dataset_yaml)

def test_validate_image_label_consistency(temp_dataset_yaml):
    train_img = temp_dataset_yaml.parent / "images" / "train"
    train_lbl = temp_dataset_yaml.parent / "labels" / "train"
    
    # 1. Empty folders (should be consistent but total 0)
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is True
    assert res["total_images"] == 0
    
    # 2. Add images and matching labels
    (train_img / "img1.jpg").touch()
    (train_lbl / "img1.txt").touch()
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is True
    assert res["total_images"] == 1
    
    # 3. Mismatch - missing label file
    (train_img / "img2.png").touch()
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is False
    assert res["missing_labels"] == ["img2.png"]
    
    # 4. Mismatch - orphan label file
    (train_lbl / "img2.txt").touch()
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is True  # Now both have img2
    
    (train_lbl / "orphan.txt").touch()
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is False
    assert res["orphan_labels"] == ["orphan.txt"]

@patch("ml_pipeline.train.YOLO")
def test_run_training_pipeline_success(mock_yolo, temp_dataset_yaml):
    # Setup image and label files
    train_img = temp_dataset_yaml.parent / "images" / "train"
    train_lbl = temp_dataset_yaml.parent / "labels" / "train"
    val_img = temp_dataset_yaml.parent / "images" / "val"
    val_lbl = temp_dataset_yaml.parent / "labels" / "val"
    
    (train_img / "img1.jpg").touch()
    (train_lbl / "img1.txt").touch()
    (val_img / "val1.jpg").touch()
    (val_lbl / "val1.txt").touch()
    
    # Mock model instance and model.train
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    mock_model_instance.train.return_value = "training_results"
    
    args = argparse.Namespace(
        dataset=str(temp_dataset_yaml),
        model="yolov8n.pt",
        epochs=3,
        imgsz=640,
        batch=16,
        device="cpu",
        project="artifacts/training",
        name="baseline"
    )
    
    report = run_training_pipeline(args)
    
    # Verify YOLO calls
    mock_yolo.assert_called_once_with("yolov8n.pt")
    mock_model_instance.train.assert_called_once_with(
        data=str(temp_dataset_yaml),
        epochs=3,
        imgsz=640,
        batch=16,
        project="artifacts/training",
        name="baseline",
        device="cpu"
    )
    
    # Verify report output values
    assert report["status"] == "success"
    assert report["model_baseline"] == "yolov8n.pt"
    assert report["hyperparameters"]["epochs"] == 3
    assert report["dataset_stats"]["train_images"] == 1
    assert report["dataset_stats"]["val_images"] == 1

def test_generate_markdown_report():
    data = {
        "timestamp": "2026-08-25T10:00:00",
        "model_baseline": "yolov8n.pt",
        "dataset_yaml": "artifacts/prepared_dataset/dataset.yaml",
        "output_directory": "artifacts/training/baseline",
        "hyperparameters": {
            "epochs": 3,
            "imgsz": 640,
            "batch": 16,
            "device": "cpu"
        },
        "dataset_stats": {
            "train_images": 1412,
            "val_images": 341
        }
    }
    
    report_md = generate_markdown_report(data)
    
    assert "YOLO Training Report" in report_md
    assert "yolov8n.pt" in report_md
    assert "Epochs: 3" in report_md
    assert "Training Images: 1412" in report_md
    assert "Validation Images: 341" in report_md
    assert "artifacts/training/baseline/weights/best.pt" in report_md
