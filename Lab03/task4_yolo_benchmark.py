"""
Task 4: Multi-Model YOLO Comparative Benchmarking
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module executes comparative benchmarking between YOLOv8 Nano (yolov8n.pt) and
YOLOv8 Medium (yolov8m.pt) on urban street scenes. It evaluates detection accuracy,
inference latency (ms), confidence scores, and renders custom bounding boxes and annotations.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task4_YOLOBenchmark")


def generate_distinct_colors(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """Generate deterministic visually distinct colors for bounding boxes."""
    np.random.seed(42)
    colors = []
    for _ in range(num_classes):
        color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
        colors.append(color)
    return colors


CLASS_COLORS = generate_distinct_colors(100)


def run_yolo_inference(
    model_weight: str,
    image_path: str,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run YOLO inference, calculate latency, extract detections, and render custom annotations.

    Args:
        model_weight: Weights filename (e.g., 'yolov8n.pt', 'yolov8m.pt').
        image_path: Absolute path to the test image.
        conf_threshold: Minimum confidence score to filter detections.

    Returns:
        annotated_img: Annotated BGR image with bounding boxes and text.
        summary_stats: Metrics dict (Model Name, Latency ms, Count, Avg Conf).
        detections: List of parsed detection dicts.
    """
    logger.info(f"Loading YOLO model weights: {model_weight}")
    model = YOLO(model_weight)

    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    annotated_img = bgr_img.copy()

    # Warmup pass (optional for hardware stabilization)
    _ = model(bgr_img, verbose=False)

    # Timed inference pass
    start_time = time.perf_counter()
    results = model(bgr_img, conf=conf_threshold, verbose=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    boxes_data = results[0].boxes
    detections: List[Dict[str, Any]] = []
    confidences: List[float] = []

    names_dict = model.names

    for box in boxes_data:
        # Extract coordinates (x1, y1, x2, y2)
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names_dict.get(cls_id, f"cls_{cls_id}")

        confidences.append(conf)
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

        # Draw bounding box and label
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        # Label background pill
        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_y = max(y1, label_h + 10)
        cv2.rectangle(
            annotated_img,
            (x1, top_y - label_h - 6),
            (x1 + label_w + 4, top_y + baseline),
            color,
            -1
        )
        cv2.putText(
            annotated_img,
            label,
            (x1 + 2, top_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    summary_stats = {
        "Model Name": model_weight.replace(".pt", "").upper(),
        "Inference Latency (ms)": round(elapsed_ms, 2),
        "Detected Object Count": len(detections),
        "Average Confidence Score": round(avg_conf, 4)
    }

    return annotated_img, summary_stats, detections


def run_benchmark(
    input_filename: str = "street.jpg",
    output_filename: str = "task4_output.png"
):
    """
    Execute comparative benchmarking between YOLOv8 Nano and Medium models.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    # 1. Benchmark YOLOv8n (Nano)
    img_nano_bgr, stats_nano, _ = run_yolo_inference("yolov8n.pt", image_path)
    img_nano_rgb = cv2.cvtColor(img_nano_bgr, cv2.COLOR_BGR2RGB)

    # 2. Benchmark YOLOv8m (Medium)
    img_med_bgr, stats_med, _ = run_yolo_inference("yolov8m.pt", image_path)
    img_med_rgb = cv2.cvtColor(img_med_bgr, cv2.COLOR_BGR2RGB)

    # Print Formatted Comparative Terminal Table
    table_header = f"{'Model Name':<12} | {'Latency (ms)':<16} | {'Detected Count':<16} | {'Avg Confidence':<16}"
    divider = "-" * len(table_header)
    row_nano = f"{stats_nano['Model Name']:<12} | {stats_nano['Inference Latency (ms)']:<16.2f} | {stats_nano['Detected Object Count']:<16} | {stats_nano['Average Confidence Score']:<16.4f}"
    row_med = f"{stats_med['Model Name']:<12} | {stats_med['Inference Latency (ms)']:<16.2f} | {stats_med['Detected Object Count']:<16} | {stats_med['Average Confidence Score']:<16.4f}"

    print("\n" + "=" * len(table_header))
    print(" TASK 4: MULTI-MODEL YOLO COMPARATIVE BENCHMARK ")
    print("=" * len(table_header))
    print(table_header)
    print(divider)
    print(row_nano)
    print(row_med)
    print("=" * len(table_header) + "\n")

    # Matplotlib Side-by-Side Visual Comparison
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    # Nano Plot
    axes[0].imshow(img_nano_rgb)
    axes[0].set_title(
        f"YOLOv8 Nano (yolov8n)\nLatency: {stats_nano['Inference Latency (ms)']} ms | Objects: {stats_nano['Detected Object Count']} | Avg Conf: {stats_nano['Average Confidence Score']:.2f}",
        fontsize=12, fontweight="bold", color="#61afef", pad=12
    )
    axes[0].axis("off")

    # Medium Plot
    axes[1].imshow(img_med_rgb)
    axes[1].set_title(
        f"YOLOv8 Medium (yolov8m)\nLatency: {stats_med['Inference Latency (ms)']} ms | Objects: {stats_med['Detected Object Count']} | Avg Conf: {stats_med['Average Confidence Score']:.2f}",
        fontsize=12, fontweight="bold", color="#98c379", pad=12
    )
    axes[1].axis("off")

    plt.suptitle(
        "Lab 03 - Task 4: Multi-Model YOLO Comparative Benchmarking (Nano vs. Medium)",
        fontsize=14, fontweight="bold", y=0.98, color="#e5c07b"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Successfully saved Task 4 benchmark plot to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
