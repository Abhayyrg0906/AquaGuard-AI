import pytest
import json
import tempfile
from pathlib import Path
from ml_pipeline.compare_experiments import (
    find_metrics_files,
    load_metrics,
    generate_comparison_table
)

def test_compare_experiments():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create mock experiment directories
        exp1_dir = tmp_path / "exp-01-baseline"
        exp2_dir = tmp_path / "exp-02-longer"
        exp1_dir.mkdir(parents=True, exist_ok=True)
        exp2_dir.mkdir(parents=True, exist_ok=True)
        
        # Write dummy metrics.json files
        metric1 = {
            "name": "exp-01-baseline",
            "model_name": "yolov8n.pt",
            "epochs": 10,
            "batch": 8,
            "imgsz": 640,
            "device": "cpu",
            "duration": "10m",
            "precision": 0.8344,
            "recall": 0.6765,
            "mAP50": 0.7941,
            "mAP50-95": 0.5490,
            "training_configuration": {
                "optimizer": "auto",
                "mosaic": 1.0
            }
        }
        metric2 = {
            "name": "exp-02-longer",
            "model_name": "yolov8n.pt",
            "epochs": 30,
            "batch": 8,
            "imgsz": 640,
            "device": "cpu",
            "duration": "30m",
            "precision": 0.8500,
            "recall": 0.7000,
            "mAP50": 0.8100,
            "mAP50-95": 0.5600,
            "training_configuration": {
                "optimizer": "auto",
                "mosaic": 1.0
            }
        }
        
        with open(exp1_dir / "metrics.json", "w") as f:
            json.dump(metric1, f)
        with open(exp2_dir / "metrics.json", "w") as f:
            json.dump(metric2, f)
            
        # Run comparison functions
        files = find_metrics_files(tmp_path)
        assert len(files) == 2
        
        metrics = load_metrics(files)
        assert len(metrics) == 2
        assert metrics[0]["name"] == "exp-01-baseline"
        assert metrics[1]["name"] == "exp-02-longer"
        
        table = generate_comparison_table(metrics)
        assert "exp-01-baseline" in table
        assert "exp-02-longer" in table
        assert "0.8344" in table
        assert "0.8500" in table

def test_compare_experiments_empty():
    assert "No experiment results found" in generate_comparison_table([])
    with tempfile.TemporaryDirectory() as tmp_dir:
        files = find_metrics_files(Path(tmp_dir))
        assert files == []
        metrics = load_metrics(files)
        assert metrics == []
