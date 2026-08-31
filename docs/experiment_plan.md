# AquaGuard AI - Model Improvement Experiment Plan

This document establishes a disciplined framework for training and hyperparameter optimization experiments targeting our floating plastic waste detection model.

---

## 1. Baseline Reference

- **Trained Model Checkpoint:** `artifacts/training/baseline-10ep/weights/best.pt`
- **Dataset Configuration:** `artifacts/yolo_dataset/dataset.yaml` (validation split: 341 images, 374 instances)
- **Baseline Hyperparameters:**
  - Model Architecture: YOLOv8-nano (`yolov8n.pt`)
  - Image size: 640px
  - Batch size: 8
  - Device: CPU (deterministic=true, seed=42)
  - Epochs: 10
- **Verified Baseline Metrics:**
  - Precision: `0.8344`
  - Recall: `0.6765`
  - mAP@0.5: `0.7941`
  - mAP@0.5:0.95: `0.5490`
- **Verified CPU Latency & Throughput:**
  - Average inference latency: ~127 ms per frame
  - Sequential video processing throughput: ~7.51 FPS

### Current Limitations:
1. **Low Recall:** The model misses roughly ~32% of plastic objects (Recall `0.6765`), especially partially submerged objects or objects under vegetation/glare.
2. **CPU Throughput:** Throughput on CPU (~7.51 FPS) is insufficient for real-time edge processing (>30 FPS).
3. **Single-Class Scope:** The model only detects generic "plastic" waste.

---

## 2. Proposed Experiment Design

To prevent arbitrary training runs, we design four controlled experiments. To avoid overwriting the baseline run, all output files are organized under `artifacts/training/experiments/<experiment-id>`.

```
  Experiment A (Baseline)
            │
            ▼
  Experiment B (Longer Run) ──► Evaluate epochs influence
            │
            ▼
  Experiment C (Tuned lr0/AdamW/Mosaic) ──► Evaluate augmentation/optimizer
            │
            ▼
  Experiment D (YOLOv8-Small Capacity) ──► Evaluate capacity vs latency
```

### Experiment A: Current Baseline (experiment-A-10ep)
- **Objective:** Reproduce and verify the initial 10-epoch CPU baseline.
- **Variables Changed:** None.
- **Variables Held Constant:** All hyperparameters.
- **Expected Artifacts:** Checkpoints, `metrics.json`, and `training_report.md` inside `artifacts/training/experiments/experiment-A-10ep/`.
- **Reproducibility Command:**
  ```bash
  python -m ml_pipeline.train \
      --data "artifacts/yolo_dataset/dataset.yaml" \
      --model "yolov8n.pt" \
      --epochs 10 \
      --imgsz 640 \
      --batch 8 \
      --device cpu \
      --project "artifacts/training/experiments" \
      --name "experiment-A-10ep" \
      --seed 42
  ```

### Experiment B: Reproducibility Run (experiment-B-10ep)
- **Objective:** Evaluate baseline reproducibility and compile comparison runs.
- **Variables Changed:** Output run directory name only.
- **Variables Held Constant:** All baseline parameters (epochs=10).
- **Expected Artifacts:** Checkpoints and reports under `artifacts/training/experiments/experiment-B-10ep/`.
- **Reproducibility Command:**
  ```bash
  python -m ml_pipeline.train \
      --data "artifacts/yolo_dataset/dataset.yaml" \
      --model "yolov8n.pt" \
      --epochs 10 \
      --imgsz 640 \
      --batch 8 \
      --device cpu \
      --project "artifacts/training/experiments" \
      --name "experiment-B-10ep" \
      --seed 42
  ```

### Experiment C: Tuned Optimizer & Augmentation (exp-03-augmentation-adamw)
- **Objective:** Investigate if the `AdamW` optimizer (better generalization) and lowering the `mosaic` augmentation probability from 1.0 to 0.5 (reducing background context confusion) improves localization precision and recall.
- **Variables Changed:**
  - `epochs` = 30
  - `optimizer` = "AdamW"
  - `lr0` = 0.001
  - `mosaic` = 0.5
  - `patience` = 10 (early stopping safety check)
- **Variables Held Constant:** Model (`yolov8n`), imgsz (640), batch (8), device (cpu), seed (42).
- **Expected Artifacts:** Checkpoints and reports under `artifacts/training/experiments/exp-03-augmentation-adamw/`.
- **Reproducibility Command:**
  ```bash
  python -m ml_pipeline.train \
      --data "artifacts/yolo_dataset/dataset.yaml" \
      --model "yolov8n.pt" \
      --epochs 30 \
      --imgsz 640 \
      --batch 8 \
      --device cpu \
      --project "artifacts/training/experiments" \
      --name "exp-03-augmentation-adamw" \
      --seed 42 \
      --lr0 0.001 \
      --optimizer "AdamW" \
      --patience 10 \
      --mosaic 0.5
  ```

### Experiment D: Alternate Model Capacity (exp-04-yolov8-small)
- **Objective:** Benchmark the slightly larger `yolov8s.pt` model (YOLOv8-Small) to see if the higher parameter capacity translates to a significant accuracy increase, and measure the corresponding CPU frame latency penalty.
- **Variables Changed:** `model` changed from `yolov8n.pt` to `yolov8s.pt`.
- **Variables Held Constant:** Epochs (10), imgsz (640), batch (8), device (cpu), seed (42).
- **Expected Artifacts:** Checkpoints and reports under `artifacts/training/experiments/exp-04-yolov8-small/`.
- **Reproducibility Command:**
  ```bash
  python -m ml_pipeline.train \
      --data "artifacts/yolo_dataset/dataset.yaml" \
      --model "yolov8s.pt" \
      --epochs 10 \
      --imgsz 640 \
      --batch 8 \
      --device cpu \
      --project "artifacts/training/experiments" \
      --name "exp-04-yolov8-small" \
      --seed 42
  ```

---

## 3. Evaluation and Comparison Plan

### Comparison Criteria
Every experiment run automatically compiles a `metrics.json` file. We will compare:
1. **Precision:** Localization reliability.
2. **Recall:** Percentage of floating plastic successfully detected.
3. **mAP50:** Bounding box accuracy at IoU 0.50 threshold.
4. **mAP50-95:** Bounding box regression accuracy over multiple thresholds.
5. **CPU Inference Latency (ms):** Frame-by-frame latency benchmark.
6. **Training Duration:** Time required to converge on CPU.

### Comparison Execution
After running the target experiments, execute the comparison script to automatically generate the Markdown table summary comparing all variables:
```bash
python -m ml_pipeline.compare_experiments
```
The comparison report will be written directly to `artifacts/training/experiments/comparison_report.md`.

---

## 4. Hardware Resource Warnings

> [!WARNING]
> **CPU Training Execution Constraint:**
> Performing YOLO training on CPU is extremely resource-intensive. The 10-epoch baseline training run took approximately **1 hour and 27 minutes** to complete. Running the 30-epoch experiments (Experiment B and Experiment C) on a CPU will take approximately **4.5 to 5 hours** per run.

> [!TIP]
> **GPU Training Recommendation:**
> It is highly recommended to perform these training experiments on a GPU-enabled machine (CUDA). To execute any experiment on GPU (e.g. GPU device 0), change the `--device` flag from `cpu` to `0`. On an NVIDIA GPU, each 10-epoch run is expected to complete in under 5 minutes.
