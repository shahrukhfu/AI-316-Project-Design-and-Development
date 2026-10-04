"""
Task 5: Retail Inventory Monitoring with Class-Filtered Object Detection
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module performs automated inventory counting on retail supermarket shelves.
It filters YOLOv8 model detections programmatically for specific target class IDs
(e.g., COCO ID 47 for 'apple' or ID 39 for 'bottle'), enforces confidence thresholds (> 0.50),
overlays high-visibility bounding boxes, and renders a status banner.
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task5_RetailFilter")


def filter_inventory_detections(
    results,
    model_names: Dict[int, str],
    target_class_ids: List[int] = [47],  # 47 is 'apple', 39 is 'bottle' in COCO
    conf_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    """
    Extract and filter detection predictions for specified target class IDs and confidence threshold.

    Args:
        results: Ultralytics YOLO Results object.
        model_names: Mapping of class index to class label string.
        target_class_ids: List of integer class IDs to retain.
        conf_threshold: Confidence score cutoff (exclusive or inclusive > 0.50).

    Returns:
        List of target detection dictionaries.
    """
    filtered_items: List[Dict[str, Any]] = []
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())

        # Check if detected class is in target classes and passes confidence gate
        if cls_id in target_class_ids and conf > conf_threshold:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            filtered_items.append({
                "class_id": cls_id,
                "class_name": model_names.get(cls_id, f"Class_{cls_id}"),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "centroid": ((x1 + x2) // 2, (y1 + y2) // 2)
            })

    return filtered_items


def render_retail_overlay(
    image: np.ndarray,
    target_detections: List[Dict[str, Any]],
    target_names_str: str,
    box_color: Tuple[int, int, int] = (46, 204, 113)  # Emerald Green in BGR
) -> np.ndarray:
    """
    Draw bounding boxes, item indices, centroids, and a dark top status banner.

    Args:
        image: Original BGR image.
        target_detections: Filtered detection list.
        target_names_str: Text label of target class(es).
        box_color: BGR color for bounding boxes.

    Returns:
        Annotated BGR image with top banner.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Draw individual detected item boxes and labels
    for idx, item in enumerate(target_detections, start=1):
        x1, y1, x2, y2 = item["bbox"]
        conf = item["confidence"]
        cx, cy = item["centroid"]

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        # Center marker
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Label tag
        tag = f"#{idx} {item['class_name']}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_top = max(y1, th + 8)
        cv2.rectangle(annotated, (x1, tag_top - th - 4), (x1 + tw + 4, tag_top + baseline), box_color, -1)
        cv2.putText(
            annotated, tag, (x1 + 2, tag_top - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Render dark top banner across the entire frame width
    banner_height = 65
    banner_overlay = annotated.copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_height), (18, 18, 18), -1)
    # Blend overlay with 85% opacity
    cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)

    # Accent indicator border
    cv2.rectangle(annotated, (0, banner_height - 3), (w, banner_height), (46, 204, 113), -1)

    # Banner Text
    item_count = len(target_detections)
    main_banner_text = f"Target Item Count: {item_count}"
    sub_banner_text = f"Target Filter: [{target_names_str.upper()}] | Confidence Gate: > 0.50 | Shelf Status: MONITORED"

    cv2.putText(
        annotated, main_banner_text, (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        annotated, sub_banner_text, (22, 53),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 230, 180), 1, cv2.LINE_AA
    )

    return annotated


def run_retail_monitoring(
    input_filename: str = "retail.jpg",
    output_filename: str = "task5_output.png",
    target_classes: Optional[List[int]] = None,
    conf_threshold: float = 0.50
):
    """
    Execute Task 5 Retail Inventory Monitoring pipeline:
    1. Load image and ensure dataset
    2. Run YOLOv8 model inference
    3. Filter for target class IDs (default: 47 for 'apple', or 39 for 'bottle' if bottles present)
    4. Render top banner and bounding boxes
    5. Save visual report to outputs/task5_output.png
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    logger.info(f"Loading retail scene image: {image_path}")
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Use YOLOv8 Medium or Nano for high fidelity detection
    logger.info("Initializing YOLOv8 model...")
    model = YOLO("yolov8m.pt")

    # Run inference
    raw_results = model(bgr_img, conf=0.25, verbose=False)
    names = model.names

    # Check detected classes to support bottle (39) or apple (47) intelligently
    detected_class_ids = set([int(b.cls[0].cpu().numpy()) for b in raw_results[0].boxes])
    if target_classes is None:
        # Default priority: if apples detected, filter apples (47); else if bottles detected, filter bottles (39)
        if 47 in detected_class_ids:
            target_classes = [47]
        elif 39 in detected_class_ids:
            target_classes = [39]
        else:
            # Fallback to both bottle and apple
            target_classes = [47, 39]

    target_labels = [names.get(cid, str(cid)) for cid in target_classes]
    target_label_str = ", ".join(target_labels)

    logger.info(f"Filtering for target class IDs: {target_classes} ({target_label_str}) with conf > {conf_threshold}")

    # Filter detections
    target_detections = filter_inventory_detections(
        raw_results,
        model_names=names,
        target_class_ids=target_classes,
        conf_threshold=conf_threshold
    )

    # Terminal summary table
    print("\n" + "=" * 65)
    print(f" TASK 5: RETAIL INVENTORY MONITORING - ITEM COUNT ")
    print("=" * 65)
    print(f" • Target Class(es)           : {target_label_str.title()} (IDs: {target_classes})")
    print(f" • Confidence Threshold       : > {conf_threshold:.2f}")
    print(f" • TOTAL ISOLATED ITEMS COUNT : {len(target_detections)}")
    print("-" * 65)
    print(f"{'Item #':<8} | {'Class Name':<12} | {'Confidence':<12} | {'Bounding Box [x1, y1, x2, y2]'}")
    print("-" * 65)
    for i, itm in enumerate(target_detections, start=1):
        print(f"#{i:<7} | {itm['class_name']:<12} | {itm['confidence']:<12.3f} | {itm['bbox']}")
    print("=" * 65 + "\n")

    # Render visual overlay
    annotated_bgr = render_retail_overlay(
        bgr_img,
        target_detections,
        target_names_str=target_label_str,
        box_color=(46, 204, 113)
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Save via Matplotlib for high DPI formatting
    plt.style.use("dark_background")
    plt.figure(figsize=(13, 9), dpi=150)
    plt.imshow(annotated_rgb)
    plt.title(
        f"Lab 03 - Task 5: Retail Inventory Monitoring | Target: {target_label_str.title()} | Count = {len(target_detections)}",
        fontsize=13, fontweight="bold", pad=12, color="#98c379"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Successfully saved Task 5 retail inventory report to: {output_path}")


if __name__ == "__main__":
    run_retail_monitoring()
