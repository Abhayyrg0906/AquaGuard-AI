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
