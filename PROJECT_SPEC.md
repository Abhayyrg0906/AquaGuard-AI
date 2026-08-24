# Project Specification: AquaGuard AI

AI-Based Plastic Waste Detection in Water Bodies Using Computer Vision

---

## 1. Problem Statement

Plastic pollution in marine and fresh water ecosystems is one of the most critical environmental challenges of our time. Large amounts of plastic waste enter rivers, lakes, and oceans daily, degrading ecosystems, harming wildlife, and entering the human food chain. 

Current monitoring methods are mostly manual, localized, and slow. There is a pressing need for automated, scalable computer vision solutions that can:
1. Detect floating plastic objects in real-time or batch video feeds.
2. Classify the type of plastic waste to identify sources.
3. Calculate pollution indices to help environmental groups and municipal authorities prioritize cleanup efforts.

AquaGuard AI addresses this need by providing an end-to-end, production-grade platform combining a computer vision pipeline, a robust backend API, a real-time web dashboard, and experiment tracking to manage the ML lifecycle.

---

## 2. Target Users

- **Environmental NGOs & Researchers:** Need to track pollution trends, identify pollution sources, and gather quantitative data on aquatic litter.
- **Municipal & Port Authorities:** Need real-time alerts or reports to schedule cleanup vessels/booms and identify critical pollution hotspots.
- **ML Engineers & Data Scientists:** Need to evaluate model updates, analyze detection errors, and audit dataset quality dynamically.

---

## 3. Functional Requirements

### 3.1 Data & ML Core
- **Task A: Scene Classification:** Classify the water body scene's overall environmental state (e.g., heavily polluted, clear water, vegetation-heavy, sandy shore).
- **Task B: Object Detection & Classification:** Detect and draw bounding boxes around specific floating plastic classes (e.g., plastic bottles, canisters, polystyrene, bags) with confidence scores.
- **Metrics Extraction:** Calculate detection statistics per image/video frame, including object counts, label distributions, and average confidence.
- **Environmental Analytics:** Calculate a custom **Pollution Index (PI)** and **Pollution Severity Category** based on bounding box count and density.

### 3.2 Backend API (FastAPI)
- **Image Inference:** Endpoint `POST /api/v1/predict/image` to process uploaded images and return detected objects, bounding boxes, and metadata.
- **Video Inference:** Endpoint `POST /api/v1/predict/video` to process uploaded videos asynchronously, returning a job ID to poll or get live updates.
- **Historical Analysis:** Endpoints to query past detections, look up individual records, and retrieve aggregated historical statistics.
- **Model Registry Access:** Endpoint `GET /api/v1/models` to display currently active models and metadata.
- **System Health:** Endpoint `GET /api/v1/health` for monitoring database connectivity and model status.

### 3.3 Web Dashboard (React + TS)
- **Interactive Overview:** View high-level metrics (total items detected, temporal pollution trends, distribution of plastic types).
- **Single Image Analysis:** Upload interface where users can preview images with interactive bounding boxes and confidence score tooltips.
- **Video Processing Center:** Upload videos, track asynchronous processing status, and view overlay video streams.
- **Historical Logs:** Browse past runs, search by date, model version, or pollution index, and view saved prediction images.
- **Model Hub:** Compare active models and review validation/metrics charts directly from the UI.

### 3.4 ML Lifecycle & MLOps
- **Experiment Tracking:** Integration with **MLflow** to track dataset versions, model training hyperparameters, validation curves, and model weights.
- **Offline Evaluation:** Automatic evaluation scripts generating confusion matrices, Precision-Recall curves, and per-class mAP metrics.

---

## 4. Non-Functional Requirements

- **Performance & Latency:**
  - Single-image API inference latency should be `< 200ms` on CPU baseline.
  - Video processing must support at least `15 FPS` processing throughput on a standard GPU or optimized runtime.
- **Security:**
  - Validate all file uploads (enforce extensions like `.jpg`, `.jpeg`, `.png`, `.mp4` and maximum file sizes, e.g., 10MB for images, 100MB for videos).
  - Prevent execution of arbitrary payloads by checking magic numbers of uploaded files.
  - Secure environment variables and database credentials.
- **Scalability & Deployability:**
  - Containerized deployment using Docker and Docker Compose (separate containers for frontend, backend, model worker, and PostgreSQL).
  - Horizontal scaling ready: decoupled API handlers from inference logic via worker queues (planned for video).
- **Reliability:**
  - Robust exception handling in FastAPI with user-friendly error messages and detailed internal structured logging.
  - Comprehensive unit and integration test coverage (`pytest` target > 80%).

---

## 5. Technical Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Dataset Label Quality:** Missing bounding boxes in the PLD/PLQ dataset. | High | Perform a dataset audit; use semi-automated labeling (e.g., Segment Anything Model or Grounding DINO) and manual verification with Label Studio to create high-quality boxes. |
| **Class Imbalance:** Extreme frequency differences between "plastic bottles" and low-frequency items. | Medium | Apply stratified splits, class-weighted loss functions, and focal loss. Implement custom augmentations (e.g., Copy-Paste augmentation) for rare classes. |
| **Illumination & Water Reflectance:** Glare, shadows, and turbidity reduce detection accuracy. | Medium | Experimentally evaluate image enhancement pipelines (e.g., CLAHE, bilateral filters, contrast adjustments) and add heavy illumination augmentations during training. |
| **Model Size/Inference Latency:** High latency during real-time video/webcam processing. | High | Train a lightweight baseline model (e.g., YOLOv8-nano/small), and export the final model to **ONNX Runtime** for optimized CPU inference. |
| **Data Leakage:** Identical water backgrounds appearing in both train and validation splits. | High | Perform video-frame or scene-based grouping when splitting data to ensure validation images come from distinct locations or sequences. |

---

## 6. Assumptions

1. The provided PLD/PLQ dataset contains high-quality JPG/PNG images representing realistic aquatic conditions.
2. Computational resources (at least local GPU, e.g., via CUDA, or standard CPU) are available for model prototyping and training.
3. The environmental analytical metrics (e.g., Pollution Index) are clearly marked as project-specific heuristics.
