"""Integration and unit tests for AquaGuard-AI web application."""

import os
import io
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app, resolve_model_path, DEFAULT_MODEL_PATH


@pytest.fixture
def client():
    """Provides a FastAPI test client instance."""
    return TestClient(app)


def test_app_import_and_metadata():
    """Verify application instance metadata."""
    assert app.title == "AquaGuard AI"
    assert app.version == "1.0.0"


def test_serve_dashboard(client):
    """Verify root URL serves the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AquaGuard" in response.text
    assert "Floating Plastic Waste Detection" in response.text


def test_health_endpoint(client):
    """Verify /api/v1/health returns structured status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert data["version"] == "1.0.0"


def test_model_info_endpoint(client):
    """Verify /api/v1/model-info returns verified baseline metrics."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "YOLOv8n"
    assert data["training_epochs"] == 10
    assert data["image_size"] == 640
    assert data["device"] == "CPU"
    assert data["precision"] == 0.8344
    assert data["recall"] == 0.6765
    assert data["map50"] == 0.7941
    assert data["map50_95"] == 0.5490
    assert "Baseline Evaluation" in data["metrics_type"]


def test_samples_endpoint(client):
    """Verify /api/v1/samples lists available assets."""
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)
    if len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "id" in sample
        assert "name" in sample
        assert "type" in sample
        assert sample["type"] in {"image", "video"}


@patch("app.main.PlasticDetector")
def test_predict_image_upload_success(mock_detector_cls, client, tmp_path):
    """Verify /api/v1/predict/image with valid image upload."""
    mock_instance = MagicMock()
    mock_instance.annotate.return_value = {
        "source": "test.jpg",
        "image_width": 640,
        "image_height": 640,
        "detection_count": 2,
        "detections": [
            {
                "class_id": 0,
                "class_name": "plastic",
                "confidence": 0.88,
                "x1": 100.0,
                "y1": 150.0,
                "x2": 200.0,
                "y2": 250.0
            },
            {
                "class_id": 0,
                "class_name": "plastic",
                "confidence": 0.74,
                "x1": 300.0,
                "y1": 350.0,
                "x2": 400.0,
                "y2": 450.0
            }
        ],
        "inference_time_ms": 112.5
    }
    mock_detector_cls.return_value = mock_instance

    # Create dummy image bytes
    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")
    files = {"file": ("test.jpg", fake_img, "image/jpeg")}
    data = {"confidence": "0.30", "iou": "0.50"}

    response = client.post("/api/v1/predict/image", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert res_json["detection_count"] == 2
    assert len(res_json["detections"]) == 2
    assert res_json["detections"][0]["class_name"] == "plastic"
    assert res_json["inference_time_ms"] == 112.5
    assert "annotated_image_url" in res_json


@patch("app.main.PlasticDetector")
def test_predict_image_no_detections(mock_detector_cls, client):
    """Verify /api/v1/predict/image handling when 0 objects detected."""
    mock_instance = MagicMock()
    mock_instance.annotate.return_value = {
        "source": "clear_water.jpg",
        "image_width": 640,
        "image_height": 640,
        "detection_count": 0,
        "detections": [],
        "inference_time_ms": 95.0
    }
    mock_detector_cls.return_value = mock_instance

    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")
    files = {"file": ("clear_water.jpg", fake_img, "image/jpeg")}

    response = client.post("/api/v1/predict/image", files=files)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert res_json["detection_count"] == 0
    assert len(res_json["detections"]) == 0
    assert "No floating plastic waste detected" in res_json["message"]


def test_predict_image_invalid_extension(client):
    """Verify /api/v1/predict/image rejects invalid file extensions."""
    fake_txt = io.BytesIO(b"not an image file")
    files = {"file": ("invalid.txt", fake_txt, "text/plain")}

    response = client.post("/api/v1/predict/image", files=files)
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


def test_predict_image_missing_input(client):
    """Verify /api/v1/predict/image handles missing file and sample."""
    response = client.post("/api/v1/predict/image", data={})
    assert response.status_code == 400
    assert "No image provided" in response.json()["detail"]


@patch("app.main.process_video")
def test_predict_video_upload_success(mock_process_video, client):
    """Verify /api/v1/predict/video with valid video upload."""
    mock_process_video.return_value = {
        "frames_processed": 30,
        "total_detections": 28,
        "average_inference_time_ms": 125.4,
        "average_processing_time_ms": 132.8,
        "total_processing_time_s": 3.98,
        "processing_fps": 7.54,
        "output_path": "artifacts/predictions/web/annotated_video_test.mp4"
    }

    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42")
    files = {"file": ("test_clip.mp4", fake_video, "video/mp4")}

    response = client.post("/api/v1/predict/video", files=files)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "success"
    assert res_json["frames_processed"] == 30
    assert res_json["total_detections"] == 28
    assert res_json["processing_fps"] == 7.54
    assert "output_video_url" in res_json


def test_predict_video_invalid_extension(client):
    """Verify /api/v1/predict/video rejects invalid formats."""
    fake_pdf = io.BytesIO(b"%PDF-1.4")
    files = {"file": ("doc.pdf", fake_pdf, "application/pdf")}

    response = client.post("/api/v1/predict/video", files=files)
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_resolve_model_path_custom_and_fallback(tmp_path):
    """Verify model resolution helper function."""
    # Test non-existent custom path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_model_path("non_existent_model_weights.pt")

    # Test existing custom model
    dummy_model = tmp_path / "dummy_weights.pt"
    dummy_model.write_bytes(b"dummy")
    resolved = resolve_model_path(str(dummy_model))
    assert resolved == dummy_model


@patch("app.main.PlasticDetector")
def test_predict_image_threshold_forwarding(mock_detector_cls, client):
    """Verify default (0.25/0.45) and custom threshold values forwarded to PlasticDetector."""
    mock_instance = MagicMock()
    mock_instance.annotate.return_value = {
        "source": "test.jpg",
        "image_width": 640,
        "image_height": 640,
        "detection_count": 0,
        "detections": [],
        "inference_time_ms": 100.0
    }
    mock_detector_cls.return_value = mock_instance

    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

    # 1. Test Default Thresholds
    res_default = client.post("/api/v1/predict/image", files={"file": ("test.jpg", fake_img, "image/jpeg")})
    assert res_default.status_code == 200
    _, default_kwargs = mock_detector_cls.call_args
    assert default_kwargs["confidence"] == 0.25
    assert default_kwargs["iou"] == 0.45

    # 2. Test Custom Thresholds
    fake_img.seek(0)
    res_custom = client.post(
        "/api/v1/predict/image",
        files={"file": ("test.jpg", fake_img, "image/jpeg")},
        data={"confidence": "0.75", "iou": "0.60"}
    )
    assert res_custom.status_code == 200
    _, custom_kwargs = mock_detector_cls.call_args
    assert custom_kwargs["confidence"] == 0.75
    assert custom_kwargs["iou"] == 0.60


@patch("app.main.process_video")
def test_predict_video_threshold_forwarding(mock_process_video, client):
    """Verify default and custom threshold values forwarded to process_video."""
    mock_process_video.return_value = {
        "frames_processed": 10,
        "total_detections": 5,
        "average_inference_time_ms": 120.0,
        "average_processing_time_ms": 130.0,
        "total_processing_time_s": 1.3,
        "processing_fps": 7.69,
        "output_path": "artifacts/predictions/web/video_out.mp4"
    }

    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42")

    # 1. Test Default Thresholds
    res_default = client.post("/api/v1/predict/video", files={"file": ("clip.mp4", fake_video, "video/mp4")})
    assert res_default.status_code == 200
    _, default_kwargs = mock_process_video.call_args
    assert default_kwargs["confidence"] == 0.25
    assert default_kwargs["iou"] == 0.45

    # 2. Test Custom Thresholds
    fake_video.seek(0)
    res_custom = client.post(
        "/api/v1/predict/video",
        files={"file": ("clip.mp4", fake_video, "video/mp4")},
        data={"confidence": "0.85", "iou": "0.55"}
    )
    assert res_custom.status_code == 200
    _, custom_kwargs = mock_process_video.call_args
    assert custom_kwargs["confidence"] == 0.85
    assert custom_kwargs["iou"] == 0.55


def test_dashboard_slider_defaults_and_hints(client):
    """Verify dashboard HTML contains default slider values (0.25 / 0.45) and guidance hints."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert 'value="0.25"' in html
    assert 'value="0.45"' in html
    assert "Lower threshold (e.g. 0.25) reveals weaker" in html
    assert "Non-Maximum Suppression" in html

