# Air University Islamabad
## Department of Creative Technologies
### Course: AI Project Design and Development (AI-316)
### **Lab 03: Computer Vision Prototyping & Object Detection Pipeline**

---

## Project Overview
This repository contains a modular, production-ready computer vision engineering pipeline developed for **Lab 03 (AI-316)**. It covers the full lifecycle of classical image processing, statistical tensor normalization, frequency and gradient edge detection, deep learning object detection using YOLOv8, perimeter analytics, and real-time edge surveillance logging.

---

## Repository Architecture

```text
Lab03_Computer_Vision/
│
├── data/                       # Downloaded sample benchmark images
│   ├── traffic.jpg             # High-res urban traffic surveillance frame
│   ├── defect.jpg              # Industrial metal specimen with surface rust
│   ├── xray.jpg                # High-resolution clinical chest X-Ray scan
│   ├── street.jpg              # Street scene with vehicles and pedestrians
│   ├── retail.jpg              # Supermarket shelf inventory scene
│   └── person.jpg              # Security camera scene with individuals
│
├── alerts/                     # Generated security breach snapshots (Task 8)
│   ├── 20261004_211319.jpg
│   └── ...
│
├── outputs/                    # Visual figures, plots, and dashboards
│   ├── task1_output.png        # Letterboxed & normalized comparison
│   ├── task2_output.png        # 2x2 defect segmentation grid
│   ├── task3_output.png        # 6-panel medical edge & filtering analysis
│   ├── task4_output.png        # YOLOv8n vs YOLOv8m benchmark comparison
│   ├── task5_output.png        # Class-filtered retail inventory monitoring
│   ├── task6_output.jpg        # ROI perimeter intrusion alert visualization
│   └── webcam_snapshot.jpg     # Live detection snapshot
│
├── task1_preprocessing.py      # Task 1: Letterboxing, tensor normalization & statistics
├── task2_color_segmentation.py # Task 2: HSV/LAB color-space defect segmentation & masking
├── task3_edge_detection.py     # Task 3: Gaussian/Median filtering, Sobel gradients, Canny
├── task4_yolo_benchmark.py     # Task 4: Multi-model YOLO benchmarking (Nano vs Medium)
├── task5_retail_filter.py      # Task 5: Inventory monitoring with COCO class filtering
├── task6_intrusion_alert.py    # Task 6: Smart security intrusion alerting with polygon ROI
├── task7_live_detection.py     # Task 7: Real-time webcam dashboard with FPS calculation
├── task8_surveillance_logger.py# Task 8: Edge surveillance event logger and snapshot engine
│
├── requirements.txt            # System dependencies
├── download_datasets.py        # Automated test image downloader with mirrors
├── events.log                  # Structured audit log of security events
└── README.md                   # Complete documentation
```

---

## Installation & Environment Setup

1. **Activate your Python environment** (Python 3.8+ recommended):
   ```powershell
   python --version
   ```

2. **Install project dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify/Download all benchmark datasets**:
   ```powershell
   python download_datasets.py
   ```

---

## Task-by-Task Execution Guide

### **Task 1: Automated Preprocessing Pipeline for Traffic Camera Surveillance**
* **File:** `task1_preprocessing.py`
* **Features:**
  * Letterboxing algorithm preserving aspect ratio without geometric distortion.
  * Symmetric border padding with neutral gray fill `(114, 114, 114)`.
  * Tensor scaling from integer $[0, 255]$ to floating-point $[0.0, 1.0]$.
  * Prints tensor shape, data type, min value, and max value.
* **Execution:**
  ```powershell
  python task1_preprocessing.py
  ```
* **Output:** Saved to `outputs/task1_output.png`.

---

### **Task 2: Industrial Defect Detection via Color-Space Transformation**
* **File:** `task2_color_segmentation.py`
* **Features:**
  * Transforms BGR image into HSV (Hue, Saturation, Value) and CIE LAB color spaces.
  * Isolates rust and oxidation using tuned HSV bounds ($H \in [5, 25], S \in [50, 255], V \in [50, 255]$).
  * Refines binary mask using binary thresholding and morphological opening/closing.
  * Extracts isolated defect texture using `cv2.bitwise_and`.
* **Execution:**
  ```powershell
  python task2_color_segmentation.py
  ```
* **Output:** $2 \times 2$ analytical grid saved to `outputs/task2_output.png`.

---

### **Task 3: Edge Detection & Noise Reduction for Medical Imaging**
* **File:** `task3_edge_detection.py`
* **Features:**
  * Evaluates Gaussian Blur ($5 \times 5, \sigma=1.0$) and Median Filtering ($k=5$) for noise suppression.
  * Computes directional Sobel derivatives ($I_x, I_y$) using 64-bit float buffers.
  * Calculates Euclidean gradient magnitude:
    $$G = \sqrt{I_x^2 + I_y^2}$$
  * Multi-stage Canny Edge Detection with hysteresis thresholds ($T_{\text{lower}}=50, T_{\text{upper}}=150$).
* **Execution:**
  ```powershell
  python task3_edge_detection.py
  ```
* **Output:** 6-panel analytical visualization saved to `outputs/task3_output.png`.

---

### **Task 4: Multi-Model YOLO Comparative Benchmarking**
* **File:** `task4_yolo_benchmark.py`
* **Features:**
  * Head-to-head comparison between **YOLOv8 Nano** (`yolov8n.pt`) and **YOLOv8 Medium** (`yolov8m.pt`).
  * Measures inference latency in milliseconds (ms) with high-resolution timers.
  * Prints formatted terminal comparison table:
    * `Model Name`
    * `Inference Latency (ms)`
    * `Detected Object Count`
    * `Average Confidence Score`
* **Execution:**
  ```powershell
  python task4_yolo_benchmark.py
  ```
* **Output:** Side-by-side detection plot saved to `outputs/task4_output.png`.

---

### **Task 5: Retail Inventory Monitoring with Class-Filtered Object Detection**
* **File:** `task5_retail_filter.py`
* **Features:**
  * Filters YOLO detections programmatically for specific target class IDs (e.g., COCO ID `47` for apple, `39` for bottle).
  * Enforces confidence cutoff ($> 0.50$).
  * Displays bounding boxes, individual indices, and centroids.
  * Overlays a dark top banner displaying `"Target Item Count: X"`.
* **Execution:**
  ```powershell
  python task5_retail_filter.py
  ```
* **Output:** High-resolution inventory monitor saved to `outputs/task5_output.png`.

---

### **Task 6: Smart Security Intrusion Alerting with ROI Analytics**
* **File:** `task6_intrusion_alert.py`
* **Features:**
  * Defines a 4-point polygon Region of Interest (ROI) security perimeter.
  * Computes bottom-center ground contact points $(x_c, y_2)$ for detected persons (COCO ID `0`).
  * Evaluates perimeter containment using `cv2.pointPolygonTest`.
  * **Intrusion State:** Polygon drawn in **RED**, banner displays `"INTRUSION DETECTED"`.
  * **Secure State:** Polygon drawn in **GREEN**, banner displays `"SYSTEM SECURE"`.
* **Execution:**
  ```powershell
  python task6_intrusion_alert.py
  ```
* **Output:** Security event frame saved to `outputs/task6_output.jpg`.

---

### **Task 7: Real-Time Webcam Multi-Object Detection Dashboard**
* **File:** `task7_live_detection.py`
* **Features:**
  * Real-time stream processing on `cv2.VideoCapture(0)`.
  * Real-time smoothed Frames Per Second (FPS) counter badge on top-left.
  * Interactive key handlers:
    * Press `'s'` to capture snapshot to `outputs/webcam_snapshot.jpg`.
    * Press `'q'` to gracefully release hardware resources and exit.
  * Includes automated `--test` simulation mode for headless validation.
* **Execution:**
  ```powershell
  # Real live webcam feed:
  python task7_live_detection.py

  # Automated test simulation:
  python task7_live_detection.py --test
  ```

---

### **Task 8: Edge-Triggered Webcam Surveillance System with Log Generation**
* **File:** `task8_surveillance_logger.py`
* **Features:**
  * Continuous surveillance feed evaluating high-priority threats (`person`, `cell phone`).
  * Condition gate: Detection confidence $> 0.65$.
  * Automatically saves event snapshots into `alerts/YYYYMMDD_HHMMSS.jpg`.
  * Appends structured audit log entries into `events.log`:
    `Timestamp | Class Name | Confidence | Bounding Box [x1, y1, x2, y2]`
  * Dynamic red recording indicator (`"● REC / LOGGING ACTIVE"`) with blinking dot during active logging.
  * Smooth exit on keypress `'q'`.
* **Execution:**
  ```powershell
  # Real live surveillance:
  python task8_surveillance_logger.py

  # Automated verification simulation:
  python task8_surveillance_logger.py --test
  ```

---

## Summary of Benchmark Results

| Task | Target Image | Processing Technique | Key Output / Metric | Status |
|---|---|---|---|---|
| **Task 1** | `traffic.jpg` | Letterbox 640x640 + [0, 1] Norm | Shape: `(640, 640, 3)`, Range: `[0.0, 0.93]` | Verified |
| **Task 2** | `defect.jpg` | HSV Bound Masking + Bitwise AND | Defect Ratio: `63.75%`, Clusters: `38` | Verified |
| **Task 3** | `xray.jpg` | Gaussian + Sobel Magnitude + Canny | Mean Mag: `11.15`, Canny Density: `0.07%` | Verified |
| **Task 4** | `street.jpg` | YOLOv8n vs YOLOv8m Benchmark | Nano: `145 ms (0.66 conf)`, Med: `862 ms (0.89 conf)` | Verified |
| **Task 5** | `retail.jpg` | Class-Filtered Detection (ID 47) | Target Count: `16 items` (Conf > 0.50) | Verified |
| **Task 6** | `person.jpg` | 4-Point Polygon `pointPolygonTest` | Status: `INTRUSION DETECTED` (Red Banner) | Verified |
| **Task 7** | Live Webcam / Test | Real-Time FPS HUD + Hotkeys | FPS Meter, Key 's' Snapshot, Key 'q' Exit | Verified |
| **Task 8** | Live Stream / Test | Edge Event Audit Logging (> 0.65) | `alerts/*.jpg` + formatted `events.log` | Verified |
