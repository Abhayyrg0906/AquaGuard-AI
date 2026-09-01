"""FastAPI backend application for AquaGuard-AI.

Provides REST API endpoints and web interface for floating plastic waste detection
in images and videos using the verified baseline YOLOv8 model.
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aquaguard_app")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions" / "web"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "training" / "baseline-10ep" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# Ensure output directories exist
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Import schemas
from app.schemas import (
    DetectionItem,
    ImagePredictionResponse,
    VideoPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    SampleItem,
    SamplesResponse,
)

# Import ML Pipeline inference modules safely
try:
    from ml_pipeline.inference import PlasticDetector
except ImportError:
    PlasticDetector = None
    logger.warning("PlasticDetector could not be imported from ml_pipeline.inference")

try:
    from ml_pipeline.video_inference import process_video
except ImportError:
    process_video = None
    logger.warning("process_video could not be imported from ml_pipeline.video_inference")

# Initialize FastAPI application
app = FastAPI(
    title="AquaGuard AI",
    description="AI-Based Plastic Waste Detection in Water Bodies Using Computer Vision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and media directories
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if PREDICTIONS_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(PREDICTIONS_DIR)), name="media")


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Resolves active model checkpoint with appropriate fallbacks."""
    if custom_path:
        p = Path(custom_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified model path does not exist: {custom_path}")

    if DEFAULT_MODEL_PATH.exists() and DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    if FALLBACK_MODEL_PATH.exists() and FALLBACK_MODEL_PATH.is_file():
        logger.warning(f"Baseline checkpoint not found at {DEFAULT_MODEL_PATH}. Using fallback {FALLBACK_MODEL_PATH}.")
        return FALLBACK_MODEL_PATH

    raise FileNotFoundError(
        f"No model checkpoint found at default path ({DEFAULT_MODEL_PATH}) or fallback ({FALLBACK_MODEL_PATH})."
    )


# -----------------------------------------------------------------------------
# Web Interface Route
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the AquaGuard-AI interactive demo dashboard."""
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>AquaGuard AI</h1><p>Demo interface template not found. Please verify app/templates/index.html exists.</p>",
        status_code=404
    )


# -----------------------------------------------------------------------------
# REST API Routes
# -----------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint validating application status and model availability."""
    model_loaded = False
    model_path_str = str(DEFAULT_MODEL_PATH)
    try:
        resolved = resolve_model_path()
        model_loaded = resolved.exists()
        model_path_str = str(resolved)
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=model_path_str,
        device="CPU",
        version="1.0.0"
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Returns metadata and verified baseline metrics for the active model."""
    try:
        resolved = resolve_model_path()
        model_path_str = str(resolved).replace("\\", "/")
    except Exception:
        model_path_str = str(DEFAULT_MODEL_PATH).replace("\\", "/")

    return ModelInfoResponse(
        model_name="YOLOv8n",
        training_epochs=10,
        image_size=640,
        device="CPU",
        precision=0.8344,
        recall=0.6765,
        map50=0.7941,
        map50_95=0.5490,
        metrics_type="Baseline Evaluation (10 Epochs)",
        model_path=model_path_str
    )


@app.get("/api/v1/samples", response_model=SamplesResponse)
async def get_samples():
    """Returns curated sample images and videos available for testing."""
    samples: List[SampleItem] = []

    # Check for sample video
    sample_video = PROJECT_ROOT / "artifacts" / "sample_data" / "synthetic_test.mp4"
    if sample_video.exists():
        samples.append(SampleItem(
            id="sample_video_1",
            name="Synthetic Water Body Video",
            type="video",
            path=str(sample_video.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            description="Simulated aquatic video with floating plastic bottles and debris."
        ))

    # Check for unseen sample test image
    sample_new_test = PROJECT_ROOT / "artifacts" / "sample_data" / "new_test.jpg"
    if sample_new_test.exists():
        samples.append(SampleItem(
            id="sample_image_unseen",
            name="Unseen Test Scene (new_test.jpg)",
            type="image",
            path=str(sample_new_test.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            description="Unseen aquatic scene with floating plastic waste."
        ))

    # Check for sample validation images
    val_images_dir = PROJECT_ROOT / "artifacts" / "yolo_dataset" / "images" / "val"
    if val_images_dir.exists():
        val_images = list(val_images_dir.glob("*.jpg"))[:5]
        for idx, img_p in enumerate(val_images, start=1):
            samples.append(SampleItem(
                id=f"sample_image_{idx}",
                name=f"Validation Sample {idx} ({img_p.name})",
                type="image",
                path=str(img_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                description=f"Real water surface scene from validation split: {img_p.name}"
            ))

    return SamplesResponse(samples=samples)


@app.post("/api/v1/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    sample_path: Optional[str] = Form(None),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    model_path: Optional[str] = Form(None)
):
    """Executes plastic waste detection on an uploaded image or sample image."""
    if PlasticDetector is None:
        raise HTTPException(status_code=500, detail="PlasticDetector inference engine is not available.")

    temp_input_path: Optional[Path] = None
    input_image_path: Optional[Path] = None

    try:
        resolved_model = resolve_model_path(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # 1. Resolve source image
        if file is not None and file.filename:
            # Validate extension
            ext = Path(file.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format '{ext}'. Allowed formats are .jpg, .jpeg, .png."
                )

            unique_id = uuid.uuid4().hex[:8]
            temp_input_path = PREDICTIONS_DIR / f"upload_{unique_id}{ext}"
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_image_path = temp_input_path
            source_name = file.filename

        elif sample_path:
            p = PROJECT_ROOT / sample_path if not Path(sample_path).is_absolute() else Path(sample_path)
            if not p.exists() or not p.is_file():
                raise HTTPException(status_code=404, detail=f"Sample image path does not exist: {sample_path}")
            input_image_path = p
            source_name = p.name

        elif sample_id:
            # Look up sample by ID
            samples_resp = await get_samples()
            matched = next((s for s in samples_resp.samples if s.id == sample_id and s.type == "image"), None)
            if not matched:
                raise HTTPException(status_code=404, detail=f"Sample image with id '{sample_id}' not found.")
            input_image_path = PROJECT_ROOT / matched.path
            source_name = input_image_path.name
        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Please upload a file or select a sample image."
            )

        # 2. Initialize detector and run inference
        detector = PlasticDetector(
            model_path=resolved_model,
            confidence=confidence,
            iou=iou,
            device="cpu"
        )

        out_filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        annotated_output_path = PREDICTIONS_DIR / out_filename

        result = detector.annotate(
            source=input_image_path,
            output_path=annotated_output_path
        )

        # 3. Format response
        detections = [
            DetectionItem(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                x1=d["x1"],
                y1=d["y1"],
                x2=d["x2"],
                y2=d["y2"],
            )
            for d in result.get("detections", [])
        ]

        detection_count = len(detections)
        message = (
            f"Successfully detected {detection_count} floating plastic waste object(s)."
            if detection_count > 0
            else "No floating plastic waste detected above the confidence threshold."
        )

        annotated_url = f"/media/{out_filename}"

        return ImagePredictionResponse(
            status="success",
            source=source_name,
            image_width=result.get("image_width", 0),
            image_height=result.get("image_height", 0),
            detection_count=detection_count,
            detections=detections,
            inference_time_ms=result.get("inference_time_ms", 0.0),
            annotated_image_url=annotated_url,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing image prediction")
        raise HTTPException(status_code=500, detail=f"Image inference failed: {str(e)}")


@app.post("/api/v1/predict/video", response_model=VideoPredictionResponse)
async def predict_video(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    sample_path: Optional[str] = Form(None),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    model_path: Optional[str] = Form(None)
):
    """Executes plastic waste detection on an uploaded video or sample video."""
    if process_video is None:
        raise HTTPException(status_code=500, detail="Video inference engine is not available.")

    temp_input_path: Optional[Path] = None
    input_video_path: Optional[Path] = None

    try:
        resolved_model = resolve_model_path(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # 1. Resolve source video
        if file is not None and file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported video format '{ext}'. Allowed formats are .mp4, .avi, .mov, .mkv."
                )

            unique_id = uuid.uuid4().hex[:8]
            temp_input_path = PREDICTIONS_DIR / f"upload_{unique_id}{ext}"
            with open(temp_input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_video_path = temp_input_path

        elif sample_path:
            p = PROJECT_ROOT / sample_path if not Path(sample_path).is_absolute() else Path(sample_path)
            if not p.exists() or not p.is_file():
                raise HTTPException(status_code=404, detail=f"Sample video path does not exist: {sample_path}")
            input_video_path = p

        elif sample_id:
            samples_resp = await get_samples()
            matched = next((s for s in samples_resp.samples if s.id == sample_id and s.type == "video"), None)
            if not matched:
                raise HTTPException(status_code=404, detail=f"Sample video with id '{sample_id}' not found.")
            input_video_path = PROJECT_ROOT / matched.path
        else:
            raise HTTPException(
                status_code=400,
                detail="No video provided. Please upload a video or select a sample."
            )

        # 2. Run video inference pipeline
        out_filename = f"annotated_video_{uuid.uuid4().hex[:8]}.mp4"
        annotated_video_path = PREDICTIONS_DIR / out_filename

        metrics = process_video(
            model_path=str(resolved_model),
            source_path=str(input_video_path),
            output_path=str(annotated_video_path),
            confidence=confidence,
            iou=iou,
            device="cpu"
        )

        output_video_url = f"/media/{out_filename}"
        frames_processed = metrics.get("frames_processed", 0)
        total_detections = metrics.get("total_detections", 0)

        message = (
            f"Successfully processed {frames_processed} frames with {total_detections} total plastic detections."
            if frames_processed > 0
            else "Video processing completed with 0 frames processed."
        )

        return VideoPredictionResponse(
            status="success",
            frames_processed=frames_processed,
            total_detections=total_detections,
            average_inference_time_ms=metrics.get("average_inference_time_ms", 0.0),
            average_processing_time_ms=metrics.get("average_processing_time_ms", 0.0),
            total_processing_time_s=metrics.get("total_processing_time_s", 0.0),
            processing_fps=metrics.get("processing_fps", 0.0),
            output_video_url=output_video_url,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing video prediction")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting AquaGuard-AI application server at http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
