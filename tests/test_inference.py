import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_pipeline.inference import PlasticDetector

@pytest.fixture
def dummy_model_file():
    """Creates a temporary dummy model file to pass initialization validation."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(b"mock weights")
        path = Path(tmp.name)
    yield path
    path.unlink()

@pytest.fixture
def dummy_image_file():
    """Creates a temporary valid image file using PIL."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        path = Path(tmp.name)
    img = Image.new("RGB", (640, 480), color="blue")
    img.save(path)
    yield path
    path.unlink()

def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        PlasticDetector("nonexistent_model_file_path.pt")

@patch("ml_pipeline.inference.YOLO")
def test_detector_initialization(mock_yolo, dummy_model_file):
    detector = PlasticDetector(dummy_model_file, confidence=0.3, iou=0.5)
    assert detector.confidence == 0.3
    assert detector.iou == 0.5
    mock_yolo.assert_called_once_with(dummy_model_file)

@patch("ml_pipeline.inference.YOLO")
def test_invalid_image_inputs(mock_yolo, dummy_model_file):
    detector = PlasticDetector(dummy_model_file)
    
    # 1. Non-existent image file
    with pytest.raises(FileNotFoundError):
        detector.predict("nonexistent_image_path.jpg")
        
    # 2. Empty numpy array
    with pytest.raises(ValueError):
        detector.predict(np.array([]))
        
    # 3. Invalid source type
    with pytest.raises(ValueError):
        detector.predict(123)

@patch("ml_pipeline.inference.YOLO")
def test_predict_empty_detections(mock_yolo, dummy_model_file, dummy_image_file):
    # Setup mock YOLO instance predict result
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    
    # Mock return list of Result objects
    mock_result = MagicMock()
    mock_result.orig_shape = (480, 640)
    mock_result.boxes = None  # No boxes detected
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = PlasticDetector(dummy_model_file, confidence=0.25, iou=0.45)
    res = detector.predict(dummy_image_file)
    
    assert res["source"] == str(dummy_image_file)
    assert res["image_width"] == 640
    assert res["image_height"] == 480
    assert res["detection_count"] == 0
    assert res["detections"] == []
    assert "inference_time_ms" in res
    assert isinstance(res["inference_time_ms"], float)

@patch("ml_pipeline.inference.YOLO")
def test_predict_structured_detections(mock_yolo, dummy_model_file, dummy_image_file):
    # Setup mock YOLO instance predict result
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "trash_plastic"}
    mock_yolo.return_value = mock_model_instance
    
    # Mock box objects
    mock_box = MagicMock()
    mock_box.xyxy = [[10.5, 20.5, 100.5, 200.5]]
    mock_box.conf = [0.85]
    mock_box.cls = [0]
    
    mock_result = MagicMock()
    mock_result.orig_shape = (480, 640)
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = PlasticDetector(dummy_model_file, confidence=0.25, iou=0.45)
    res = detector.predict(dummy_image_file)
    
    assert res["detection_count"] == 1
    det = res["detections"][0]
    assert det["class_id"] == 0
    assert det["class_name"] == "trash_plastic"
    assert det["confidence"] == 0.85
    assert det["x1"] == 10.5
    assert det["y1"] == 20.5
    assert det["x2"] == 100.5
    assert det["y2"] == 200.5

@patch("ml_pipeline.inference.YOLO")
def test_numpy_array_input(mock_yolo, dummy_model_file):
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "trash_plastic"}
    mock_yolo.return_value = mock_model_instance
    
    mock_result = MagicMock()
    mock_result.orig_shape = (480, 640)
    mock_result.boxes = []
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = PlasticDetector(dummy_model_file)
    # Create random image numpy array (480 height, 640 width, 3 channels)
    numpy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.predict(numpy_image)
    
    assert res["source"] == "numpy_array"
    assert res["image_width"] == 640
    assert res["image_height"] == 480

@patch("ml_pipeline.inference.YOLO")
def test_annotate_method(mock_yolo, dummy_model_file, dummy_image_file):
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "trash_plastic"}
    mock_yolo.return_value = mock_model_instance
    
    # Mock single box detection
    mock_box = MagicMock()
    mock_box.xyxy = [[50.0, 50.0, 150.0, 150.0]]
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    
    mock_result = MagicMock()
    mock_result.orig_shape = (480, 640)
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = PlasticDetector(dummy_model_file)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_image_path = Path(tmp_dir) / "output.jpg"
        res = detector.annotate(dummy_image_file, output_image_path)
        
        assert output_image_path.exists()
        assert res["detection_count"] == 1

@patch("ml_pipeline.inference.YOLO")
def test_multi_image_inference(mock_yolo, dummy_model_file):
    # Mock YOLO instance and prediction results
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "plastic"}
    mock_yolo.return_value = mock_model_instance
    
    mock_box = MagicMock()
    mock_box.xyxy = [[10.0, 20.0, 100.0, 200.0]]
    mock_box.conf = [0.8]
    mock_box.cls = [0]
    
    mock_result = MagicMock()
    mock_result.orig_shape = (480, 640)
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]
    
    detector = PlasticDetector(dummy_model_file)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_dir = Path(tmp_dir) / "source"
        output_dir = Path(tmp_dir) / "output"
        source_dir.mkdir()
        
        # Touch mock image files
        from PIL import Image
        img = Image.new("RGB", (640, 480), color="blue")
        img.save(source_dir / "img1.jpg")
        img.save(source_dir / "img2.png")
        # Touch a non-image file
        (source_dir / "dummy.txt").touch()
        
        from ml_pipeline.inference import run_multi_image_inference
        
        res = run_multi_image_inference(
            detector=detector,
            source_dir=source_dir,
            output_dir=output_dir,
            max_images=None
        )
        
        assert res["summary"]["images_processed"] == 2
        assert res["summary"]["total_detections"] == 2
        assert res["summary"]["images_with_detections"] == 2
        assert (output_dir / "predictions.json").exists()
        assert (output_dir / "images" / "img1.jpg").exists()
        assert (output_dir / "images" / "img2.png").exists()
        assert not (output_dir / "images" / "dummy.txt").exists()
        
        # Test max_images limit
        res_limit = run_multi_image_inference(
            detector=detector,
            source_dir=source_dir,
            output_dir=output_dir,
            max_images=1
        )
        assert res_limit["summary"]["images_processed"] == 1
