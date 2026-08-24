# Machine Learning Specification: AquaGuard AI

This document outlines the machine learning task definitions, baseline model specifications, training tracking, and evaluation metrics for AquaGuard AI.

---

## 1. Machine Learning Tasks

The platform comprises two distinct ML tasks targeting environmental monitoring:

### Task A: Scene Classification (Environmental Context)
- **Objective:** Determine the overall environment type or pollution status of a scene.
- **Inputs:** Raw RGB images of water bodies.
- **Outputs:** Categorical class labels (e.g., `Vegetation`, `Sand`, `Water`, `Highly Polluted`).
- **Baseline Model:** ResNet-18 or EfficientNet-B0 pretrained on ImageNet, fine-tuned on the PLD dataset.

### Task B: Object Detection (Plastic Detection & Classification)
- **Objective:** Locate and classify individual floating debris items.
- **Inputs:** Raw RGB images or video frames.
- **Outputs:** Bounding boxes `[x_center, y_center, width, height]` (normalized), class labels, and confidence scores.
- **Baseline Model:** YOLOv8-nano (Ultralytics) trained on annotated PLQ dataset segments.

---

## 2. Model Input & Output Architectures

```text
[Input Image (e.g., 640x640x3)]
       │
       ├─► Task A Classifier ──► Class Probabilities: [Water: 0.1, Polluted: 0.9]
       │
       └─► Task B Detector   ──► Detection Array:
                                  [ [x_min, y_min, x_max, y_max, conf, class_id], ... ]
```

---

## 3. Evaluation Metrics Specification

To guarantee scientific and engineering integrity, the model performance will be reported using the following metrics. No target thresholds are claimed until verified experimentally.

### 3.1 Object Detection Metrics (Task B)
- **mAP@0.5:** Mean Average Precision calculated at an Intersection over Union (IoU) threshold of 0.5. Useful for evaluating general localization accuracy.
- **mAP@0.5:0.95:** Mean Average Precision averaged over IoU thresholds ranging from 0.5 to 0.95 in steps of 0.05. Standard COCO metric for precise localization evaluation.
- **Precision (P) & Recall (R):** Measures model precision (avoiding false positives) and recall (avoiding missed plastics).
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Per-Class Metrics:** Track AP for each material category (e.g., plastic bottle vs. plastic bag) to identify weak spots.

### 3.2 Operational Performance Metrics
- **Inference Latency (ms):** End-to-end processing time of a single frame (preprocessing + forward pass + postprocessing).
- **Frames Per Second (FPS):** Latency translated to throughput (1000 / latency_ms). Crucial for real-time video processing.
- **Model Size (MB):** File size of weights (critical for memory footprint in container environments).

---

## 4. Experiment Tracking with MLflow

During model training, the following variables will be logged deterministically to MLflow:

```text
MLflow Experiment: aquaguard-waste-detection
├── Run Name: yolov8n-baseline-epoch100
│   ├── Parameters:
│   │   ├── lr0: 0.01
│   │   ├── batch: 16
│   │   ├── imgsz: 640
│   │   └── optimizer: SGD
│   ├── Metrics ( logged per epoch ):
│   │   ├── train/box_loss, train/cls_loss
│   │   ├── val/box_loss, val/cls_loss
│   │   ├── metrics/precision, metrics/recall
│   │   └── metrics/mAP50, metrics/mAP50-95
│   └── Artifacts:
│       ├── weights/best.pt (PyTorch Model)
│       ├── weights/best.onnx (Exported Model)
│       ├── PR_curve.png
│       └── confusion_matrix.png
```

---

## 5. Model Export & Runtime Optimization

To meet the non-functional requirement of single-image latency `< 200ms` on typical server CPUs:

1. **ONNX Export:** Post-training, models will be exported from PyTorch (`.pt`) to ONNX (`.onnx`) with constant input shapes:
   ```bash
   yolo export model=best.pt format=onnx imgsz=640 half=false
   ```
2. **ONNX Runtime (ORT):** The FastAPI production server will execute the model using `onnxruntime` instead of PyTorch, bypassing the large PyTorch library footprint and speeding up CPU inference.
3. **NMS Post-Processing:** Non-Maximum Suppression (NMS) will be handled in standard NumPy/Python or bundled directly in the ONNX graph to reduce serving latency.
