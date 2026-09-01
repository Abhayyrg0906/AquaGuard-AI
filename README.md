# AquaGuard AI

AI-Based Plastic Waste Detection in Water Bodies Using Computer Vision

---

## Project Overview

AquaGuard AI is a software system designed to automate the detection, localization, and classification of floating plastic waste in water bodies (rivers, oceans, lakes, and canals). By utilizing computer vision and machine learning models, the system processes image and video inputs to identify floating debris, helping environmental organizations, local authorities, and automated cleanup systems target pollution hotspots.

---

## Planned Architecture

The project is structured as a multi-component system designed for scalability, ease of deployment, and high-performance inference:

```
                  ┌────────────────────────────────────────┐
                  │          React + TS Frontend           │
                  │   (Dashboard, Analytics, Maps, UI)     │
                  └───────────────────▲────────────────────┘
                                      │  REST API
                                      │  WebSockets
                  ┌───────────────────▼────────────────────┐
                  │            FastAPI Backend             │
                  │      (Inference Service & Data API)    │
                  └───────▲────────────────────────▲───────┘
                          │ DB Connection          │ Run Inference
                          │                        │ / Load Model
  ┌───────────────────────▼───────┐        ┌───────▼───────────────────────┐
  │      PostgreSQL Database      │        │       Python ML Pipeline      │
  │ (Hotspots, Alerts, Analytics) │        │  (Model Training & Artifacts) │
  └───────────────────────────────┘        └───────────────────────────────┘
```

### Components

1. **Python ML Pipeline**
   - Implements model training, evaluation, and exporting using modern deep learning/computer vision frameworks.
   - Manages data ingestion, preprocessing, and augmentation.
   - Integrates ML experiment tracking (e.g., MLflow, Weights & Biases) to monitor runs, hyperparameters, and artifacts.

2. **FastAPI Backend**
   - Serving layer exposing REST API endpoints for image/video upload and automated classification.
   - Real-time notification endpoints (WebSockets) for incoming detections.
   - Integrates with the database to log location coordinates, classification labels, timestamps, and confidence scores.

3. **React + TypeScript Frontend**
   - A dashboard featuring live tracking maps, charts showing pollution levels, hotspots, and historical trends.
   - Image/video analysis interface where users can drag-and-drop media to preview detections in real time.

4. **PostgreSQL Database**
   - Storage for detection history, environmental metadata, geospatial data, and system configurations.

5. **CI/CD & DevOps**
   - **Docker:** Multi-stage container setup for easy local orchestrations (`docker-compose`) and production deployment.
   - **GitHub Actions:** CI workflow to run linters, formatters, and automated testing suites across the stack.

---

## Planned Directory Structure

Once development begins, the repository is planned to follow this structure:

```text
AquaGuard-AI/
├── .github/              # GitHub Actions workflows and templates
├── backend/              # FastAPI application source code
├── frontend/             # React + TypeScript application source code
├── ml_pipeline/          # Python training scripts, data prep, and model exports
├── tests/                # Core test suite (pytest, frontend tests)
├── docker/               # Dockerfiles and orchestration files
├── .env.example          # Environment variables template
├── .gitignore            # Git exclusion rules
├── LICENSE               # MIT License file
├── CONTRIBUTING.md       # Contribution guidelines
└── README.md             # Project documentation
```

---

## ML Pipeline Workflow

The ML Pipeline in this repository operates in a sequential flow from dataset preparation to inference:

```
    Dataset Preparation
            ↓
      Model Training
            ↓
     Model Evaluation
            ↓
  Batch Image Inference
            ↓
  Video Stream Inference
```

### 1. Dataset Preparation
Ensure the raw COCO format dataset is downloaded and prepared. Run the preparation script:
```bash
python -m ml_pipeline.prepare_dataset
```

### 2. Model Training
To train the baseline model (10 epochs on CPU):
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

### 3. Model Evaluation
To evaluate the trained model on the validation split:
```bash
python -m ml_pipeline.evaluate \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --data "artifacts/yolo_dataset/dataset.yaml" \
    --device cpu \
    --output "artifacts/evaluation/baseline-10ep"
```

### 4. Image Inference
To perform inference on a single image and save the annotated output:
```bash
python -m ml_pipeline.inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "path/to/image.jpg" \
    --output "artifacts/predictions/annotated_single.jpg"
```
To run batch inference on 20 validation images:
```bash
python -m ml_pipeline.inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "artifacts/yolo_dataset/images/val" \
    --output "artifacts/predictions" \
    --max-images 20
```

### 5. Video Inference
To run sequential inference on an input video:
```bash
python -m ml_pipeline.video_inference \
    --model "artifacts/training/baseline-10ep/weights/best.pt" \
    --source "artifacts/sample_data/synthetic_test.mp4" \
    --output "artifacts/predictions/baseline-10ep-video.mp4" \
    --device cpu
```

---

## Web Application Demo Interface

AquaGuard AI includes an integrated web application powered by **FastAPI** that serves both a REST API and a responsive demo dashboard.

### 1. Starting the Application

Launch the local web server:
```bash
python -m app.main
```
Or with Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to: `http://localhost:8000`

### 2. Web Interface Features

- **Image Detection Tab:**
  - Drag-and-drop custom aquatic images or select pre-loaded validation samples.
  - Dynamically adjust Confidence and NMS IoU thresholds.
  - View real-time annotated bounding boxes, item count, inference latency, and detection tables.
- **Video Analysis Tab:**
  - Upload video clips or execute detection on pre-loaded synthetic water body feeds.
  - View sequential frame processing metrics (FPS, total detections, average latency).
  - Stream and playback annotated output videos directly in the browser.
- **Model Baseline & Metrics Tab:**
  - View verified baseline metrics: **Precision (83.44%)**, **Recall (67.65%)**, **mAP@0.5 (79.41%)**, **mAP@0.5:0.95 (54.90%)**.
  - Inspect model architecture parameters and hardware performance notes.

### 3. Core REST API Endpoints

- `GET /` - Interactive Demo Web Interface.
- `GET /api/v1/health` - System health and active model loading status.
- `GET /api/v1/model-info` - Active model metadata and verified baseline metrics.
- `GET /api/v1/samples` - Curated sample images and videos for rapid testing.
- `POST /api/v1/predict/image` - Executes plastic detection on uploaded/sample images.
- `POST /api/v1/predict/video` - Executes sequential frame detection on uploaded/sample videos.
- Interactive Swagger API documentation: `http://localhost:8000/docs`

---

## Current Trained Baseline

- **Model Checkpoint:** `artifacts/training/baseline-10ep/weights/best.pt`
- **Validation Dataset:** 341 images, 374 instances
- **Baseline Metrics:**
  - **Precision:** 0.8344
  - **Recall:** 0.6765
  - **mAP@0.5:** 0.7941
  - **mAP@0.5:0.95:** 0.5490

---

## Current Inference & Video Artifacts

- **Batch Predictions:** Annotated validation images `test-1.jpg` to `test-20.jpg` are generated under `artifacts/predictions/` alongside a detection report at `artifacts/predictions/predictions.json`.
- **Processed Video:** The processed validation video is output to `artifacts/predictions/baseline-10ep-video.mp4` showing frame-by-frame annotations.

---

## Current Limitations & Next Improvements

### Current Limitations:
1. **Single-Class Baseline:** The model is currently a plastic-only detection model (class 0: `plastic`).
2. **CPU Throughput Constraint:** Video processing runs at approximately ~7.16 - 7.51 FPS with an average latency of ~127 - 136 ms per frame on CPU, which does not meet real-time edge video requirements (>30 FPS).
3. **Deferred Experiments:** Optimization experiments (Experiment C with AdamW/mosaic tuning and Experiment D with YOLOv8-small) are planned and documented in `docs/experiment_plan.md` but deferred for GPU execution to prevent long CPU compute runs.

### Next Improvements:
1. **GPU Scale Training:** Fine-tune for longer epochs (30–100) using GPU acceleration.
2. **Multi-Class Detection:** Extend the model taxonomy to other forms of marine pollution (e.g. glass, metal, wood, organic debris).
3. **Data Augmentation:** Integrate brightness, contrast, water reflections, and occlusion data augmentations to increase model robustness.
4. **Model Optimization:** Export to TensorRT or ONNX Runtime to optimize throughput for real-time edge deployments.

