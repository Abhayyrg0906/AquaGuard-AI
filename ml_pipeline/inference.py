import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Union, List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import YOLO safely
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PlasticDetector:
    """
    Production-grade inference engine wrapper for AquaGuard AI waste detection.
    Loads trained YOLO models and outputs structured dictionaries.
    """
    def __init__(self, model_path: Union[str, Path], confidence: float = 0.25, iou: float = 0.45):
        self.model_path = Path(model_path)
        
        # Validate model checkpoint existence
        if not self.model_path.exists() or not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model weights file not found at: {self.model_path}")
            
        self.confidence = confidence
        self.iou = iou
        
        if YOLO is None:
            raise ImportError("Ultralytics package is missing. Run 'pip install ultralytics'")
            
        logger.info(f"Loading YOLO model weights from: {self.model_path}")
        self.model = YOLO(self.model_path)

    def predict(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Performs inference on the image source (file path or numpy array)
        and returns a structured prediction dictionary.
        """
        source_name = "numpy_array"
        img_width = 0
        img_height = 0
        
        # 1. Input type and path validation
        if isinstance(source, (str, Path)):
            img_path = Path(source)
            if not img_path.exists() or not img_path.is_file():
                raise FileNotFoundError(f"Input image path does not exist: {img_path}")
            source_name = str(img_path)
            
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                raise ValueError(f"Failed to read image at {img_path}: {e}")
                
        elif isinstance(source, np.ndarray):
            if source.size == 0 or len(source.shape) < 2:
                raise ValueError("Input numpy array is empty or has invalid shape.")
            # numpy shape is (H, W, C) or (H, W)
            img_height, img_width = source.shape[:2]
        else:
            raise ValueError("Input source must be an image file path or a numpy array.")
            
        # 2. Run model prediction with latency measuring
        start_time = time.perf_counter()
        
        results = self.model.predict(
            source,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000.0
        
        # 3. Extract and structure results
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            for box in result.boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names.get(class_id, f"class_{class_id}")
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(round(conf, 4)),
                    "x1": float(round(xyxy[0], 2)),
                    "y1": float(round(xyxy[1], 2)),
                    "x2": float(round(xyxy[2], 2)),
                    "y2": float(round(xyxy[3], 2))
                })
                
        return {
            "source": source_name,
            "image_width": img_width,
            "image_height": img_height,
            "detection_count": len(detections),
            "detections": detections,
            "inference_time_ms": float(round(inference_time_ms, 2))
        }

    def annotate(self, source: Union[str, Path, np.ndarray], output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs prediction, draws colored boxes and class labels, saves the image,
        and returns the structured detection result dictionary.
        """
        # Run prediction
        pred_res = self.predict(source)
        
        # Open source image for drawing
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source).convert("RGB")
        else:
            raise ValueError("Input source must be an image file path or a numpy array.")
            
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
            
        # Draw each bounding box
        for det in pred_res["detections"]:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Format text label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Calculate text size for label background block
            try:
                if hasattr(draw, "textsize"):
                    text_w, text_h = draw.textsize(label, font=font)
                else:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = len(label) * 6, 12
                
            # Draw label block above bounding box
            label_y = max(0, y1 - text_h - 4)
            draw.rectangle([x1, label_y, x1 + text_w + 4, y1], fill="red")
            draw.text((x1 + 2, label_y + 2), label, fill="white", font=font)
            
        # Save annotated image
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_p)
        logger.info(f"Annotated output image saved to: {out_p}")
        
        return pred_res

def main():
    parser = argparse.ArgumentParser(
        description="AquaGuard AI - single image inference engine runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model weights (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output image"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Intersection-over-Union threshold"
    )
    
    args = parser.parse_args()
    
    try:
        detector = PlasticDetector(
            model_path=args.model,
            confidence=args.confidence,
            iou=args.iou
        )
        
        if args.output:
            logger.info(f"Running prediction and annotation on: {args.source}")
            result = detector.annotate(args.source, args.output)
        else:
            logger.info(f"Running prediction on: {args.source}")
            result = detector.predict(args.source)
            
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        logger.error(f"Inference run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
