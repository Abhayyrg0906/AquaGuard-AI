# Implementation Roadmap: AquaGuard AI

This document establishes the step-by-step phases for developing the AquaGuard AI platform.

---

## Roadmap Overview

```
Phase 1: Dataset Audit & Validation
   └── Phase 2: Pipeline Setup & Annotation Development
         └── Phase 3: ML Baseline Training & MLflow Setup
               └── Phase 4: FastAPI Backend & Postgres Integration
                     └── Phase 5: React Frontend Development
                           └── Phase 6: System Integration, Docker, & CI/CD
                                 └── Phase 7: Verification & Testing
```

---

## Detailed Development Phases

### Phase 1: Dataset Audit & Validation
*Goal: Audit the raw PLD/PLQ dataset to understand the starting point and prevent downstream leaks.*
- [ ] Set up the audit scripts folder inside `ml_pipeline/`.
- [ ] Audit the directory structure, file count, and file formats (validate JPG/PNG headers).
- [ ] Calculate class distribution and analyze class imbalance.
- [ ] Verify image dimensions (uniformity analysis).
- [ ] Scan for corrupted image files, duplicates, and blurry images.
- [ ] Determine the exact format of existing annotations (classification labels vs. missing bounding boxes).
- [ ] Define the data splitting strategy (e.g., stratified split grouped by scene to prevent frame/background leaks).

*Test Gate:* Run audit python script and generate `docs/dataset_audit_report.md` detailing actual file statistics.

---

### Phase 2: Pipeline Setup & Annotation Development
*Goal: Create a clean preprocessing pipeline and convert raw images/annotations to YOLO-compatible inputs.*
- [ ] Implement robust preprocessing utilities (resizing, rgb conversion, normalization).
- [ ] Propose and set up the annotation strategy (e.g., set up Label Studio config or a semi-automated YOLO annotation conversion script).
- [ ] Annotate a sub-split of the dataset for the object detection task (Task B).
- [ ] Implement data augmentation pipeline (geometric adjustments, contrast, CLAHE, and noise injection).
- [ ] Export dataset into YOLO format matching the directory structure:
  ```text
  data/processed/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```

*Test Gate:* Validate exported label syntax and range (ensure float bounding boxes are normalized `[0, 1]` and class IDs are in range).

---

### Phase 3: ML Baseline Training & MLflow Setup
*Goal: Train environmental scene classification (Task A) and object detection (Task B) models, tracking with MLflow.*
- [ ] Configure local MLflow experiment tracking environment.
- [ ] Develop Task A baseline: Train a classification model (e.g., ResNet-18 or MobileNetV3) using transfer learning.
- [ ] Develop Task B baseline: Train a lightweight object detector (e.g., YOLOv8-nano/small) on annotated splits.
- [ ] Log hyperparameters (learning rate, batch size, epochs), metrics (mAP@0.5, precision, recall, validation loss), and model artifacts to MLflow.
- [ ] Conduct offline evaluation (confusion matrix, PR curve, inference latency benchmark).
- [ ] Convert the best-performing models to **ONNX format** for lightweight deployment.

*Test Gate:* Successful execution of evaluation script printing validation mAP scores and latency figures.

---

### Phase 4: FastAPI Backend & Postgres Integration
*Goal: Create the service layer API, database models, and connect the model inference worker.*
- [ ] Set up FastAPI project boilerplate inside `backend/` directory.
- [ ] Define SQLAlchemy database models for `ModelVersion`, `InferenceRun`, `Prediction`, and `Detection`.
- [ ] Implement migrations using Alembic.
- [ ] Build the dynamic Inference Wrapper loading ONNX models.
- [ ] Implement routes:
  - `GET /api/v1/health`
  - `POST /api/v1/predict/image` (returns JSON of bounding boxes and computes Pollution Index)
  - `POST /api/v1/predict/video` (stub for asynchronous processing queue)
  - `GET /api/v1/analytics` (historical counts, type distribution)
- [ ] Implement upload security checks (file size limits, mime-type white-listing, magic-number validations).

*Test Gate:* Run `pytest` covering API routes, database CRUD, upload validation errors, and mock inference.

---

### Phase 5: React Frontend Development
*Goal: Build the responsive dashboard UI to visualize analytics and trigger inferences.*
- [ ] Initialize React + TS project using Vite or Next.js inside `frontend/` directory.
- [ ] Setup Tailwind CSS config and core UI layout (Sidebar, Navbar, Responsive cards).
- [ ] Create the **Image Upload Component** with instant preview and interactive overlay canvas showing bounding boxes.
- **Create the Analytics Dashboard:** Add historical charts (bar chart of plastic types, area chart of pollution level over time).
- [ ] Implement System Health monitor component (fetching backend health status).

*Test Gate:* Local frontend compiles successfully with zero TypeScript compilation errors.

---

### Phase 6: System Integration, Docker, & CI/CD
*Goal: Dockerize the application and set up automated workflows.*
- [ ] Create multi-stage `Dockerfile` configurations for the frontend, backend, and PostgreSQL database.
- [ ] Configure `docker-compose.yml` to spin up local developer environments with persistent database volumes.
- [ ] Write a `Makefile` or scripts for common developer commands (`make install`, `make test`, `make build`, `make run`).
- [ ] Setup GitHub Actions pipeline workflow (`.github/workflows/ci.yml`) to run linting checks, black formatting compliance, and pytest test suites.

*Test Gate:* Run `docker-compose up --build` and ensure all containers successfully start, communicate, and pass tests.

---

### Phase 7: Verification & Testing
*Goal: Conduct exhaustive verification of the integrated platform.*
- [ ] Execute load testing on endpoints to verify latency requirements (< 200ms per image).
- [ ] Validate end-to-end user flows (uploading video, viewing statistics updates).
- [ ] Create a comprehensive `walkthrough.md` with verification results, embedding screenshots/videos.
