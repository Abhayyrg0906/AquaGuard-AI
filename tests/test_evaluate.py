import pytest
import json
import argparse
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_pipeline.evaluate import run_evaluation_pipeline, main

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

@pytest.fixture
def temp_model_pt():
    """Fixture to create a dummy model checkpoint file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(b"mock weights")
        path = Path(tmp.name)
    yield path
    path.unlink()

def test_nonexistent_model(temp_dataset_yaml):
    args = argparse.Namespace(
        model="nonexistent_model_file.pt",
        data=str(temp_dataset_yaml),
        device="cpu",
        output=None
    )
    with pytest.raises(FileNotFoundError):
        run_evaluation_pipeline(args)

def test_nonexistent_dataset(temp_model_pt):
    args = argparse.Namespace(
        model=str(temp_model_pt),
        data="nonexistent_dataset.yaml",
        device="cpu",
        output=None
    )
    with pytest.raises(FileNotFoundError):
        run_evaluation_pipeline(args)

@patch("ml_pipeline.evaluate.YOLO")
def test_evaluation_success(mock_yolo, temp_model_pt, temp_dataset_yaml):
    # Mock YOLO instance and model.val()
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    
    mock_val_results = MagicMock()
    mock_box = MagicMock()
    mock_box.mp = 0.92
    mock_box.mr = 0.82
    mock_box.map50 = 0.87
    mock_box.map = 0.62
    mock_val_results.box = mock_box
    mock_model_instance.val.return_value = mock_val_results
    
    with tempfile.TemporaryDirectory() as tmp_out:
        args = argparse.Namespace(
            model=str(temp_model_pt),
            data=str(temp_dataset_yaml),
            device="cpu",
            output=tmp_out
        )
        
        report, md = run_evaluation_pipeline(args)
        
        # Check files exist
        assert (Path(tmp_out) / "metrics.json").exists()
        assert (Path(tmp_out) / "evaluation_report.md").exists()
        
        # Check report metrics
        assert report["precision"] == 0.92
        assert report["recall"] == 0.82
        assert report["mAP50"] == 0.87
        assert report["mAP50-95"] == 0.62
        
        # Check Markdown format
        assert "AquaGuard AI - Model Evaluation Report" in md
        assert f"Model Path:** `{report['model_path']}`" in md

@patch("ml_pipeline.evaluate.YOLO")
def test_evaluate_cli_help(mock_yolo):
    test_args = ["evaluate.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
