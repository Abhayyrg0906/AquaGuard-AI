"""Pydantic schemas for AquaGuard-AI API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Details of an individual detected plastic waste object."""
    class_id: int = Field(..., description="Class index identifier")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    x1: float = Field(..., description="Top-left X bounding coordinate")
    y1: float = Field(..., description="Top-left Y bounding coordinate")
    x2: float = Field(..., description="Bottom-right X bounding coordinate")
    y2: float = Field(..., description="Bottom-right Y bounding coordinate")


class ImagePredictionResponse(BaseModel):
    """Response payload for image inference."""
    status: str = Field("success", description="Execution status")
    source: str = Field(..., description="Original image filename or source identifier")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    detection_count: int = Field(..., description="Total number of plastic items detected")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected objects")
    inference_time_ms: float = Field(..., description="Model inference latency in milliseconds")
    annotated_image_url: Optional[str] = Field(None, description="URL or relative path to annotated result image")
    message: Optional[str] = Field(None, description="Informative status or warning message")


class VideoPredictionResponse(BaseModel):
    """Response payload for video inference."""
    status: str = Field("success", description="Execution status")
    frames_processed: int = Field(..., description="Total number of video frames processed")
    total_detections: int = Field(..., description="Cumulative detections across all frames")
    average_inference_time_ms: float = Field(..., description="Average inference latency per frame in ms")
    average_processing_time_ms: float = Field(..., description="Average total processing time per frame in ms")
    total_processing_time_s: float = Field(..., description="Total elapsed pipeline execution duration in seconds")
    processing_fps: float = Field(..., description="Processing throughput in frames per second")
    output_video_url: Optional[str] = Field(None, description="URL or relative path to the annotated video")
    message: Optional[str] = Field(None, description="Informative status or warning message")


class ModelInfoResponse(BaseModel):
    """Model status and verified baseline evaluation metrics."""
    model_name: str = Field("YOLOv8n", description="Model architecture variant")
    training_epochs: int = Field(10, description="Number of training epochs")
    image_size: int = Field(640, description="Input image resolution in pixels")
    device: str = Field("CPU", description="Active inference device")
    precision: float = Field(0.8344, description="Precision score on validation split")
    recall: float = Field(0.6765, description="Recall score on validation split")
    map50: float = Field(0.7941, description="mAP@0.5 score on validation split")
    map50_95: float = Field(0.5490, description="mAP@0.5:0.95 score on validation split")
    metrics_type: str = Field("Baseline Evaluation (10 Epochs)", description="Benchmark status description")
    model_path: str = Field(..., description="Filesystem location of active checkpoint")


class HealthResponse(BaseModel):
    """System health check payload."""
    status: str = Field("healthy", description="Application health status")
    model_loaded: bool = Field(..., description="Indicates whether model checkpoint is present and loadable")
    model_path: str = Field(..., description="Path to active model checkpoint")
    device: str = Field("cpu", description="Configured inference device")
    version: str = Field("1.0.0", description="Application version")


class SampleItem(BaseModel):
    """Pre-packaged sample media asset for quick demo evaluation."""
    id: str = Field(..., description="Unique sample identifier")
    name: str = Field(..., description="Display title of the sample")
    type: str = Field(..., description="Asset type ('image' or 'video')")
    path: str = Field(..., description="Relative filesystem path")
    description: str = Field(..., description="Brief scene summary")


class SamplesResponse(BaseModel):
    """List of available sample media assets."""
    samples: List[SampleItem] = Field(default_factory=list, description="Available sample files")
