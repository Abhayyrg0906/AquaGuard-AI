import pytest
import tempfile
import argparse
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from ml_pipeline.video_inference import process_video, main


@pytest.fixture
def dummy_model_file():
    """Creates a temporary dummy model file to pass initialization validation."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(b"mock weights")
        path = Path(tmp.name)
    yield path
    path.unlink()


@pytest.fixture
def dummy_video_file():
    """Creates a temporary dummy empty file representing a video."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = Path(tmp.name)
    yield path
    path.unlink()


def test_process_video_missing_model(dummy_video_file):
    with pytest.raises(FileNotFoundError):
        process_video(
            model_path="nonexistent_model.pt",
            source_path=str(dummy_video_file),
            output_path="out.mp4"
        )


def test_process_video_missing_source(dummy_model_file):
    with pytest.raises(FileNotFoundError):
        process_video(
            model_path=str(dummy_model_file),
            source_path="nonexistent_video.mp4",
            output_path="out.mp4"
        )


@patch("ml_pipeline.video_inference.cv2.VideoCapture")
@patch("ml_pipeline.video_inference.cv2.VideoWriter")
@patch("ml_pipeline.video_inference.PlasticDetector")
def test_process_video_success(mock_detector_class, mock_video_writer, mock_video_capture, dummy_model_file, dummy_video_file):
    # Setup mock VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Read returns (True, frame) once, then (False, None) to end loop
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, mock_frame), (False, None)]
    mock_cap.get.side_effect = lambda prop: {
        3: 640.0, # CAP_PROP_FRAME_WIDTH
        4: 480.0, # CAP_PROP_FRAME_HEIGHT
        5: 30.0   # CAP_PROP_FPS
    }.get(prop, 0.0)
    mock_video_capture.return_value = mock_cap
    
    # Setup mock VideoWriter
    mock_out = MagicMock()
    mock_out.isOpened.return_value = True
    mock_video_writer.return_value = mock_out
    
    # Setup mock PlasticDetector
    mock_detector = MagicMock()
    mock_detector.predict.return_value = {
        "detection_count": 1,
        "detections": [{
            "class_id": 0,
            "class_name": "plastic",
            "confidence": 0.95,
            "x1": 50.0,
            "y1": 50.0,
            "x2": 150.0,
            "y2": 150.0
        }],
        "inference_time_ms": 12.5
    }
    mock_detector_class.return_value = mock_detector
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_video_path = Path(tmp_dir) / "output.mp4"
        
        metrics = process_video(
            model_path=str(dummy_model_file),
            source_path=str(dummy_video_file),
            output_path=str(output_video_path),
            confidence=0.25,
            iou=0.45,
            device="cpu"
        )
        
        # Verify call arguments and metrics
        assert metrics["frames_processed"] == 1
        assert metrics["total_detections"] == 1
        assert metrics["average_inference_time_ms"] == 12.5
        assert metrics["processing_fps"] > 0
        
        # Verify mock writes occurred
        mock_cap.read.assert_called()
        mock_out.write.assert_called_once()
        mock_cap.release.assert_called_once()
        mock_out.release.assert_called_once()


@patch("ml_pipeline.video_inference.process_video")
def test_video_cli_parser(mock_process):
    test_args = [
        "video_inference.py",
        "--model", "model.pt",
        "--source", "video.mp4",
        "--output", "out.mp4",
        "--confidence", "0.35",
        "--iou", "0.55",
        "--device", "cuda:0"
    ]
    with patch.object(sys, "argv", test_args):
        main()
        mock_process.assert_called_once_with(
            model_path="model.pt",
            source_path="video.mp4",
            output_path="out.mp4",
            confidence=0.35,
            iou=0.55,
            device="cuda:0"
        )
