import pytest
import json
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_pipeline.train import (
    load_dataset_yaml,
    validate_dataset_paths,
    validate_image_label_consistency,
    detect_device,
    extract_metrics,
    run_training_pipeline
)

@pytest.fixture
def temp_dataset_yaml():
    """Fixture to create a temporary dataset.yaml and directory structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        yaml_file = tmp_path / "dataset.yaml"
        
        yaml_content = f"""path: {tmp_path.as_posix()}
train: images/train
val: images/val
names:
  0: plastic
"""
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
            
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        yield yaml_file

def test_load_dataset_yaml(temp_dataset_yaml):
    data = load_dataset_yaml(temp_dataset_yaml)
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"][0] == "plastic"

def test_load_dataset_yaml_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_dataset_yaml(Path("C:/nonexistent_file_path_xyz.yaml"))

def test_validate_dataset_paths(temp_dataset_yaml):
    yaml_data = load_dataset_yaml(temp_dataset_yaml)
    train_img, train_lbl, val_img, val_lbl = validate_dataset_paths(yaml_data, temp_dataset_yaml)
    assert train_img == temp_dataset_yaml.parent / "images" / "train"
    assert train_lbl == temp_dataset_yaml.parent / "labels" / "train"

def test_validate_dataset_paths_missing(temp_dataset_yaml):
    yaml_data = load_dataset_yaml(temp_dataset_yaml)
    val_dir = temp_dataset_yaml.parent / "images" / "val"
    val_dir.rmdir()
    with pytest.raises(FileNotFoundError):
        validate_dataset_paths(yaml_data, temp_dataset_yaml)

def test_validate_image_label_consistency(temp_dataset_yaml):
    train_img = temp_dataset_yaml.parent / "images" / "train"
    train_lbl = temp_dataset_yaml.parent / "labels" / "train"
    
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is True
    
    (train_img / "img1.jpg").touch()
    res = validate_image_label_consistency(train_img, train_lbl)
    assert res["is_consistent"] is False
    assert res["missing_labels"] == ["img1.jpg"]

def test_detect_device():
    with patch("ml_pipeline.train.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        assert detect_device("") == "cpu"
        
        mock_torch.cuda.is_available.return_value = False
        with pytest.raises(ValueError):
            detect_device("cuda:0")
            
        mock_torch.cuda.is_available.return_value = True
        assert detect_device("cuda:0") == "cuda:0"
        
        assert detect_device("") == "0"

def test_extract_metrics():
    assert extract_metrics(None) == {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
    
    val_results = MagicMock()
    mock_box = MagicMock()
    mock_box.mp = 0.85
    mock_box.mr = 0.75
    mock_box.map50 = 0.80
    mock_box.map = 0.55
    val_results.box = mock_box
    
    metrics = extract_metrics(val_results)
    assert metrics["precision"] == 0.85
    assert metrics["recall"] == 0.75
    assert metrics["mAP50"] == 0.80
    assert metrics["mAP50-95"] == 0.55

@patch("ml_pipeline.train.YOLO")
def test_run_training_pipeline_success(mock_yolo, temp_dataset_yaml):
    train_img = temp_dataset_yaml.parent / "images" / "train"
    train_lbl = temp_dataset_yaml.parent / "labels" / "train"
    val_img = temp_dataset_yaml.parent / "images" / "val"
    val_lbl = temp_dataset_yaml.parent / "labels" / "val"
    (train_img / "img1.jpg").touch()
    (train_lbl / "img1.txt").touch()
    (val_img / "val1.jpg").touch()
    (val_lbl / "val1.txt").touch()
    
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    mock_model_instance.train.return_value = "training"
    
    mock_val_results = MagicMock()
    mock_box = MagicMock()
    mock_box.mp = 0.90
    mock_box.mr = 0.80
    mock_box.map50 = 0.85
    mock_box.map = 0.60
    mock_val_results.box = mock_box
    mock_model_instance.val.return_value = mock_val_results
    
    with tempfile.TemporaryDirectory() as tmp_out:
        # Mock the YOLO trainer's save_dir dynamically
        mock_trainer = MagicMock()
        mock_trainer.save_dir = Path(tmp_out) / "baseline"
        mock_model_instance.trainer = mock_trainer
        
        # Touch mock weight files inside the expected output directory
        (Path(tmp_out) / "baseline" / "weights").mkdir(parents=True, exist_ok=True)
        (Path(tmp_out) / "baseline" / "weights" / "best.pt").touch()
        (Path(tmp_out) / "baseline" / "weights" / "last.pt").touch()
        
        args = argparse.Namespace(
            data=str(temp_dataset_yaml),
            model="yolov8n.pt",
            epochs=3,
            imgsz=640,
            batch=16,
            project=tmp_out,
            name="baseline",
            device="cpu"
        )
        
        report, md = run_training_pipeline(args)
        
        run_dir = Path(tmp_out) / "baseline"
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "training_report.md").exists()
        assert report["precision"] == 0.90
        assert report["mAP50"] == 0.85
        assert "0.8500" in md
        
        # Artifact consistency validation: verify reported path points to a file that actually exists
        assert Path(report["best_checkpoint_path"]).exists() is True
        assert Path(report["last_checkpoint_path"]).exists() is True
