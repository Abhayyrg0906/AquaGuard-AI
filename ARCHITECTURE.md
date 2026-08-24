# System Architecture: AquaGuard AI

This document details the architectural layout, component relationships, data flow, and database schema for the AquaGuard AI platform.

---

## 1. System Architecture Overview

AquaGuard AI is designed as a modular **monorepo** dividing the frontend presentation, backend REST API, database storage, and machine learning pipeline into decoupled layers.

```mermaid
graph TD
    %% Styling
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef database fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef ml fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff;

    %% Nodes
    subgraph Frontend_Tier ["Presentation Layer"]
        UI["React + TS SPA<br>(Tailwind CSS)"]:::frontend
    end

    subgraph Backend_Tier ["Service Layer"]
        API["FastAPI App<br>(ASGI Web Server)"]:::backend
        Inference["Inference Engine<br>(ONNX / PyTorch Wrapper)"]:::backend
    end

    subgraph Database_Tier ["Data Layer"]
        DB[(PostgreSQL Database)]:::database
    end

    subgraph ML_Lifecycle ["ML Ops & Training"]
        Pipe["ML Pipeline<br>(Data Prep & Training)"]:::ml
        MLflow["MLflow Server<br>(Experiment Registry)"]:::ml
        DataDir["Local Data Directory<br>(Datasets & Weights)"]:::ml
    end

    %% Interactions
    UI <-->|HTTP REST / WebSockets| API
    API -->|Load Model Weights| Inference
    API <-->|SQL / SQLAlchemy| DB
    
    Pipe -->|Read Datasets| DataDir
    Pipe -->|Log Metrics & Artifacts| MLflow
    Pipe -->|Save Exported Model| DataDir
    DataDir -.->|Load Active Weights| Inference
```

---

## 2. Logical Components

### 2.1 Frontend Tier (React, TypeScript, Tailwind CSS)
- **Role:** Interactive UI dashboard.
- **Key Modules:**
  - **Uploader Component:** Handles drag-and-drop file ingestion, file validation (mime-type, size), and upload progress.
  - **Interactive Canvas Component:** Renders the input image or video with custom SVG overlay overlays for bounding boxes, class labels, and confidence metrics.
  - **Analytics Panel:** Renders historical trends using a charting library (e.g., Recharts) showing debris counts, classification distributions, and active pollution index trends.
  - **Configuration Section:** Allows users to dynamically update inference parameters (e.g., confidence threshold, NMS threshold) sent to the API.

### 2.2 Backend Tier (FastAPI, Uvicorn)
- **Role:** High-performance REST API hosting business logic, database transaction orchestration, and inference wrapper.
- **Key Modules:**
  - **Controllers (Routes):** Clean routing definitions with strict Pydantic inputs/outputs.
  - **Services:** Decoupled business logic (e.g., calculation of the Pollution Index, metadata processing).
  - **Inference Wrapper:** Manages model lifetime, loads model files dynamically, performs preprocessing steps, executes forward passes, and handles postprocessing (Non-Maximum Suppression).
  - **Database Manager (SQLAlchemy):** Handles connection pools, sessions, and CRUD operations.

### 2.3 Database Tier (PostgreSQL)
- **Role:** Persistent storage.
- **Schema Design:**
  - Designed to track predictions, individual bounding box detections, loaded model versions, and general processing statistics.

### 2.4 ML Tier (Python Pipeline)
- **Role:** Independent training environment, disconnected from the serving runtime.
- **Key Modules:**
  - **Dataset Auditor:** Standalone auditing scripts verifying data integrity.
  - **Trainer:** Scripts to train classifiers and object detectors.
  - **Exporter:** Converts raw models (e.g., `.pt` files) into lightweight `.onnx` runtimes optimized for CPU execution.

---

## 3. Database Schema

The conceptual PostgreSQL database structure is organized around four key entities:

```mermaid
erDiagram
    models ||--o{ inference_runs : "used in"
    inference_runs ||--o{ predictions : "contains"
    predictions ||--o{ detections : "has"

    models {
        int id PK
        string version "Unique version tag"
        string framework "YOLOv8, ONNX, etc."
        timestamp created_at
        jsonb hyperparameters
        float map_50 "mAP@0.5 validation metric"
        float map_50_95 "mAP@0.5:0.95 validation metric"
    }

    inference_runs {
        uuid id PK
        int model_id FK
        timestamp start_time
        timestamp end_time
        string status "completed, failed"
        string source_type "image, video, webcam"
    }

    predictions {
        uuid id PK
        uuid run_id FK
        string filename
        string filepath "Storage pointer to media"
        int total_objects
        float processing_time_ms
        float pollution_index "Project-specific heuristic index"
        string pollution_severity "low, medium, high"
        timestamp created_at
    }

    detections {
        uuid id PK
        uuid prediction_id FK
        string label_class "bottle, canister, bag, etc."
        float confidence
        float x_min
        float y_min
        float x_max
        float y_max
    }
```

---

## 4. End-to-End Data Flow

The flow of data through the system from an uploaded image to visualization is detailed below:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Frontend
    participant API as FastAPI Backend
    participant ML as Inference Wrapper
    participant DB as PostgreSQL DB

    User->>UI: Uploads image or video file
    UI->>UI: Validates file type and size limit
    UI->>API: HTTP POST /api/v1/predict/image (multipart file)
    API->>API: Validates headers and checks magic numbers
    API->>ML: Pass image buffer
    ML->>ML: Preprocess: Resize, normalization, CLAHE (if enabled)
    ML->>ML: Run model inference (forward pass)
    ML->>ML: Postprocess: Non-Maximum Suppression (NMS)
    ML->>API: Return detections (labels, boxes, confidences, latency)
    API->>API: Calculate Pollution Index & Severity Category
    API->>DB: Write records to predictions & detections tables
    DB-->>API: Confirm database transaction
    API-->>UI: Return JSON Response (bounding boxes, analytical metrics, image id)
    UI->>UI: Render overlays on canvas and update charts
```

---

## 5. Security & Operational Boundary

- **Inference Sandbox:** The inference layer handles only raw numerical arrays in-memory. Under no circumstances will the system dynamically execute code embedded in models or upload packages (e.g., using safe weights loading strategies like YAML loading blocks or avoiding pickle where possible by using ONNX).
- **Network Boundaries:** Frontend communicates strictly via HTTP/HTTPS/WebSockets. Database ports are kept internal to the Docker environment and not exposed to the public internet.
- **Resource Protection:** Enforce video processing rate-limiting or synchronous queues to prevent CPU/GPU memory exhaustion by concurrently processing multiple large video uploads.
