# Dataset Specification & Audit Plan: AquaGuard AI

This document details the characteristics of the PLD/PLQ dataset and outlines the protocol for auditing the raw files before training.

---

## 1. Dataset Overview

The dataset contains two key sub-components:
- **PLD (Plastic Litter Dataset):** Focuses on scenes containing plastics vs. non-plastic elements (scene classification).
- **PLQ (Plastic Litter Quality):** Contains environmental and plastic material classes, including:
  - Plastic bottles
  - Plastic bag - large / small
  - Polystyrene packaging
  - Styrofoam
  - Plastic bowls / canister / cups / other
  - Non-plastic classes: Water, Vegetation, Sand, Other

---

## 2. Directory Structure

The expected raw dataset format is:

```text
data/raw/
├── PLD/
│   ├── class_plastic/
│   └── class_no_plastic/
└── PLQ/
    ├── class_plastic_bottles/
    ├── class_plastic_bags/
    ├── class_vegetation/
    └── [other class subfolders...]
```

---

## 3. Dataset Audit Protocol

Before running any preprocessing or training loops, we must audit the raw files to uncover structural, content, or label discrepancies. We will run a script to collect the following metrics:

### 3.1 Structural & File Checks
- **Valid Image Headers:** Verify files are readable images (JPEG/PNG) and check for corrupted file endings.
- **Image Resolution:** Calculate minimum, maximum, and average image width and height.
- **File Counts:** Count total samples per class directory.

### 3.2 Integrity & Leakage Checks
- **Duplicate Detection:** Identify exact duplicate files by calculating MD5 hashes of all images.
- **Near-Duplicates (Frame Leaks):** For datasets extracted from video sequences, check for near-identical consecutive frames to avoid splitting similar frames between train and validation sets.

### 3.3 Target Class Distribution
- **Class Imbalance:** Measure ratio of highest-frequency class (e.g., Water/Vegetation) to lowest-frequency class (e.g., Plastic canister).

---

## 4. Database Schema/Audit Script Outline

Below is the python script template (`ml_pipeline/audit_dataset.py`) planned for execution in Phase 1:

```python
import os
import hashlib
from PIL import Image
from collections import Counter

def get_image_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def audit_dataset(root_dir):
    report = {
        "corrupted_files": [],
        "image_resolutions": [],
        "class_counts": Counter(),
        "duplicates": {},
        "file_hashes": {}
    }
    
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(dirpath, f)
            class_name = os.path.basename(dirpath)
            
            # Check for corruption
            try:
                with Image.open(file_path) as img:
                    img.verify()
                    width, height = img.size
                    report["image_resolutions"].append((width, height))
            except Exception as e:
                report["corrupted_files"].append((file_path, str(e)))
                continue
            
            # Hash check for duplicates
            file_hash = get_image_md5(file_path)
            if file_hash in report["file_hashes"]:
                report["duplicates"].setdefault(file_hash, []).append(file_path)
            else:
                report["file_hashes"][file_hash] = file_path
                
            report["class_counts"][class_name] += 1
            
    return report
```

---

## 5. Train / Validation / Test Splitting Strategy

To guarantee the reliability of evaluation metrics, splitting must prevent **data leakage**:

- **No Shared Environments:** If images are extracted from video frames at a specific location, the entire video sequence must be assigned to either the Train OR Validation set, never split between both. This prevents background-based overfitting.
- **Stratified Split:** Split datasets `70% Train / 15% Validation / 15% Test` using stratified sampling based on class distribution.
- **Deterministic Splits:** Set random seeds for all splitting procedures to ensure experiment reproducibility.
