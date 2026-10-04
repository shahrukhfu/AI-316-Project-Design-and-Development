# AI Project Design and Development (AI-316) - Laboratory Repository

**Course:** AI Project Design and Development (AI-316)  
**Department:** Department of Creative Technologies, Faculty of Computing & AI  
**Institution:** Air University, Islamabad  

---

## Overview

This repository serves as the central codebase and documentation hub for laboratory assignments and practical projects completed in the **AI Project Design and Development (AI-316)** course at Air University, Islamabad. 

The contained labs demonstrate computer vision engineering, system architecture design, classical image processing, deep learning object detection, and real-time surveillance analytics.

---

## Repository Structure

```text
.
├── Lab01/                          # Industrial Fruit Defect Detection Mini Project
│   ├── configs/                    # Pipeline configuration files
│   ├── data/                       # Conveyor image datasets and samples
│   ├── inspection_outputs/         # Diagnostic grids and conveyor manifests
│   ├── logs/                       # Execution and telemetry logs
│   ├── models/                     # Deep learning weights and checkpoints
│   ├── notebooks/                  # Interactive experimentation notebooks
│   ├── src/                        # Modular source code
│   ├── requirements.txt            # Lab 01 specific dependencies
│   └── README.md                   # Detailed Lab 01 documentation
│
├── Lab02/                          # AI Surveillance & Analytics Architecture
│   ├── ARCHITECTURE.md             # System requirements and architecture specification
│   └── skeleton.py                 # Core module interfaces and pipeline skeletons
│
├── Lab03/                          # Computer Vision Prototyping & Object Detection
│   ├── alerts/                     # Security breach intrusion captures
│   ├── data/                       # Benchmark image datasets
│   ├── outputs/                    # Visual figures, plots, and benchmark cards
│   ├── download_datasets.py        # Automated test image retrieval script
│   ├── events.log                  # JSON-formatted event telemetry log
│   ├── task1_preprocessing.py      # Letterboxing, normalization, tensor conversion
│   ├── task2_color_segmentation.py # HSV/LAB defect segmentation and masking
│   ├── task3_edge_detection.py     # Filtering, Sobel gradients, Canny hysteresis
│   ├── task4_yolo_benchmark.py     # YOLOv8 nano vs medium multi-model benchmark
│   ├── task5_retail_filter.py      # Inventory counting and class filtering
│   ├── task6_intrusion_alert.py    # Polygon ROI perimeter intrusion monitoring
│   ├── task7_live_detection.py     # Live camera stream object detection dashboard
│   ├── task8_surveillance_logger.py# Automated multi-camera analytics and logger
│   ├── requirements.txt            # Lab 03 specific dependencies
│   └── README.md                   # Comprehensive Lab 03 report and documentation
│
└── README.md                       # Repository-level overview (this file)
```

---

## Laboratory Breakdown

### Lab 01: Industrial Fruit Defect Detection Mini Project
An automated optical inspection (AOI) framework engineered for high-throughput conveyor belts in agricultural sorting and food packaging facilities.
- **Stage 1 (Deterministic CV Filtering):** Frame normalization, noise filtering, Canny structural edge profiling, and bimodal Otsu foreground segmentation.
- **Stage 2 (Deep Learning Detection & Triage):** Ultralytics YOLO instance localization, bounding-box ROI extraction, and surface defect metric grading (edge density and chromatic variance) to classify produce into Grade A Fresh versus Defective / Damaged units.
- **Stage 3 (Diagnostics & Telemetry):** Generates multi-panel visual inspection grids and industrial conveyor manifests.

### Lab 02: AI Surveillance & Analytics System Architecture Specification
A formal architectural specification and abstract interface design for an enterprise-grade AI surveillance and automated attendance system.
- **System Requirements Breakdown:** Formal definition of Functional Requirements (FR-01 to FR-05) and Non-Functional Requirements (NFR-01 to NFR-05) covering latency, throughput, False Acceptance Rate (FAR), biometric data privacy, and encryption standards.
- **Architectural Diagrams:** High-level system architecture, ingestion pipelines, edge processing modules, and database synchronization topology.
- **Core Interfaces (`skeleton.py`):** Type-annotated abstract classes for data ingestion (RTSP stream management), preprocessing, inference engines, face verification, database connectors, and security alert dispatchers.

### Lab 03: Computer Vision Prototyping & Object Detection Pipeline
A production-oriented computer vision pipeline exploring fundamental image processing and state-of-the-art object detection workflows:
- **Task 1: Preprocessing & Statistical Normalization:** Aspect-ratio-preserving letterboxing to 640x640, channel-wise mean and standard deviation computation, float32 scaling, and CHW tensor transposition.
- **Task 2: Color Space Segmentation:** BGR to HSV/LAB conversion, dual-range masking, and morphological filtering (Opening/Closing) for surface rust and defect isolation.
- **Task 3: Spatial Filtering & Edge Detection:** Gaussian versus Median noise attenuation, directional Sobel kernel gradients, and Canny hysteresis edge extraction.
- **Task 4: YOLO Benchmarking:** Comparative latency, memory footprint, parameter count, and detection capability benchmarking between YOLOv8n (nano) and YOLOv8m (medium).
- **Task 5: Retail Inventory Filtering:** Target class filtering for COCO items (bottles, cups, cans) with bounding box annotations and on-screen inventory totals.
- **Task 6: Perimeter Intrusion Monitoring:** Dynamic polygon Region of Interest (ROI) definition, point-in-polygon centroid intersection testing, and visual security violation alerting.
- **Task 7: Real-Time Live Stream Dashboard:** Real-time camera acquisition, moving-average FPS calculation, and live telemetry overlay.
- **Task 8: Production Surveillance Logger:** Multi-camera processing simulation, automated event logging to JSON (`events.log`), and intrusion snapshot archiving.

---

## Environment Setup and Prerequisites

### System Requirements
- Python 3.10+
- CUDA-compatible GPU (optional, for accelerated YOLO deep learning inference)

### Global Dependencies
To install common dependencies across all lab modules:

```bash
pip install numpy opencv-python torch torchvision ultralytics matplotlib pillow
```

Alternatively, navigate into individual lab directories (`Lab01/` or `Lab03/`) and install the pinned laboratory requirements:

```bash
cd Lab01
pip install -r requirements.txt
```

or

```bash
cd Lab03
pip install -r requirements.txt
```

---

## Execution Guidelines

Each lab can be run independently within its dedicated directory:

### Running Lab 01
```bash
cd Lab01
python src/main.py
```

### Running Lab 03 Tasks
```bash
cd Lab03

# Download test datasets
python download_datasets.py

# Execute specific tasks
python task1_preprocessing.py
python task2_color_segmentation.py
python task3_edge_detection.py
python task4_yolo_benchmark.py
python task5_retail_filter.py
python task6_intrusion_alert.py
python task7_live_detection.py
python task8_surveillance_logger.py
```

---

## Academic Integrity and Attribution

This repository is maintained for educational and academic assessment purposes as part of the curriculum for **AI-316: AI Project Design and Development** at **Air University, Islamabad**. All implementations, technical reports, and architectures conform to the course laboratory guidelines and specifications.
