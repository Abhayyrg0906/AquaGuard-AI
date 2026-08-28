# YOLO Model Evaluation: Baseline Experiment

This document details the systematic baseline training, standalone evaluation, multi-image inference testing, and video inference validation results of the AquaGuard-AI floating plastic waste detection model.

---

## 1. Current Status

The YOLO-based plastic-only detection pipeline is fully implemented, verified, and integrated. A reproducible 10-epoch baseline model has been trained on the CPU, and the generated checkpoints have been validated through standalone evaluation, batch image predictions, and video processing. All code passes the automated test suite (51 tests passed).

---

## 2. Baseline Experiment Specification

The baseline training was executed to establish a reproducible benchmark under CPU conditions.

- **Model Architecture:** YOLOv8-nano (`yolov8n.pt`)
- **Dataset Configuration:** `artifacts/yolo_dataset/dataset.yaml` (custom YOLO format, class 0: `plastic`)
- **Image Resolution:** 640px
- **Device Configuration:** CPU (auto-detected fallback)
- **Baseline Hyperparameters:**
  - Epochs: 10
  - Batch Size: 8
  - Device: CPU
  - Deterministic: true

---

## 3. Quantitative Performance Analysis

To guarantee scientific and engineering integrity, we report the actual verified metrics below. These are divided into training metrics and standalone evaluation metrics.

### 3.1 Training Metrics (baseline-10ep)
The model was trained for 10 epochs on a CPU. The training metrics recorded at convergence are:

- **Precision:** 0.8344406477893705
- **Recall:** 0.6764705882352942
- **mAP@0.5:** 0.7940974168866262
- **mAP@0.5:0.95:** 0.5489786373921522
- **Training Duration:** Approximately 1h 27m
- **Artifact Output Directory:** `artifacts/training/baseline-10ep/`
- **Best Weights Checkpoint:** `artifacts/training/baseline-10ep/weights/best.pt`
- **Last Weights Checkpoint:** `artifacts/training/baseline-10ep/weights/last.pt`

### 3.2 Standalone Evaluation Metrics
A separate standalone evaluation utility was executed directly against the best model checkpoint (`best.pt`) on the validation dataset split.

- **Model Evaluated:** `artifacts/training/baseline-10ep/weights/best.pt`
- **Dataset Evaluated:** `artifacts/yolo_dataset/dataset.yaml`
- **Validation Images:** 341
- **Validation Instances:** 374
- **Evaluation Results:**
  - **Precision:** 0.8344
  - **Recall:** 0.6765
  - **mAP@0.5:** 0.7941
  - **mAP@0.5:0.95:** 0.5490
- **Artifact Output Directory:** `artifacts/evaluation/baseline-10ep/`

---

## 4. Multi-Image Inference Testing

Batch inference was tested on a representative sample of 20 images from the validation split.

- **Images Processed:** 20
- **Source Directory:** `artifacts/yolo_dataset/images/val`
- **Output Directory:** `artifacts/predictions/`
- **Annotated Image Files:** `test-1.jpg` through `test-20.jpg` under `artifacts/predictions/`
- **Result Output:** The annotated validation images confirm localized bounding box annotations around floating plastic targets.

---

## 5. Video Inference Testing

Video inference throughput and localization accuracy were validated using a synthetic test video.

- **Input Video:** `artifacts/sample_data/synthetic_test.mp4`
- **Output Video:** `artifacts/predictions/baseline-10ep-video.mp4`
- **Frames Processed:** 30
- **Total Objects Detected:** 30
- **Average Inference Latency:** Approximately 127 ms
- **Overall Processing Throughput:** Approximately 7.51 FPS
- **Device:** CPU

---

## 6. Qualitative Observation Summary

Reviewing prediction annotations from the inference testing reveals several qualitative aspects of the model's current capability:
1. **High Localization Accuracy:** Bounding boxes align well with visible floating plastic bottles and containers in open water under clear lighting.
2. **Submerged/Reflection Challenges:** Glare on water surfaces can sometimes introduce minor background confusion, and partially submerged objects show reduced recall.

---

## 7. Known Limitations / Next Improvements

1. **CPU Throughput Constraint:** The model achieves approximately 7.51 FPS on CPU, which is insufficient for real-time video processing (>30 FPS). Deploying on GPU hardware is necessary to unlock real-time throughput.
2. **Single-Class Limitation:** The model currently only detects "plastic". Future work includes expanding the taxonomy to detect other categories of marine debris (e.g., cans, wood, organic waste).
3. **Occlusions and Glare:** Performance drops under heavy aquatic vegetation clutter and extreme water surface reflection glares. Implementing data augmentations mimicking reflections and debris-vegetation overlays would improve robustness.

---

## 8. Reproducible Execution Instructions

Below are the exact commands to reproduce the baseline training, evaluation, and inference tests locally.

### 8.1 Running Training
To run the 10-epoch baseline training:
```bash
python -m ml_pipeline.train \
    --data "artifacts/yolo_dataset/dataset.yaml" \
    --model "yolov8n.pt" \
    --epochs 10 \
    --imgsz 640 \
    --batch 8 \
    --device cpu \
    --project "artifacts/training" \
    --name "baseline-10ep"
```

### 8.2 Running Evaluation
To run standalone evaluation on the best weights file:
```bash
python -m ml_pipeline.evaluate \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --data "artifacts/yolo_dataset/dataset.yaml" \
    --device cpu \
    --output "artifacts/evaluation/baseline-10ep"
```

### 8.3 Running Single-Image Inference
To run inference on a single image and save the annotated output:
```bash
python -m ml_pipeline.inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "path/to/image.jpg" \
    --output "artifacts/predictions/annotated_single.jpg"
```

### 8.4 Running Multi-Image Inference
To run batch inference on 20 validation images:
```bash
python -m ml_pipeline.inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "artifacts/yolo_dataset/images/val" \
    --output "artifacts/predictions" \
    --max-images 20
```

### 8.5 Running Video Inference
To run inference on a video and generate the annotated output mp4:
```bash
python -m ml_pipeline.video_inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "artifacts/sample_data/synthetic_test.mp4" \
    --output "artifacts/predictions/baseline-10ep-video.mp4" \
    --device cpu
```
