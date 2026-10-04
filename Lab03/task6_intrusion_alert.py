"""
Task 6: Smart Security Intrusion Alerting with ROI Bounding Box Analytics
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module performs automated perimeter surveillance. It defines a 4-point polygon
Region of Interest (ROI) security zone, runs YOLOv8 to detect humans (COCO class ID 0),
computes their ground contact point (bottom-center or centroid), and utilizes cv2.pointPolygonTest
to classify whether a security breach has occurred. The visual feed displays status-adaptive
banners and perimeter coloring (Red for Intrusion, Green for Secure).
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task6_IntrusionAlert")


def default_security_roi(width: int, height: int, mode: str = "intrusion") -> np.ndarray:
    """
    Generate 4-point polygon Region of Interest (ROI) scaled to image dimensions.

    Args:
        width: Frame width.
        height: Frame height.
        mode: 'intrusion' (encompasses active person ground points) or 'secure' (perimeter offset).

    Returns:
        np.ndarray of shape (4, 1, 2) with int32 polygon coordinates.
    """
    if mode == "intrusion":
        # ROI zone covering central and right security perimeter including floor contact
        pts = np.array([
            [int(width * 0.25), int(height * 0.35)],  # Top-left
            [int(width * 0.98), int(height * 0.35)],  # Top-right
            [int(width * 0.99), height - 1],          # Bottom-right
            [int(width * 0.20), height - 1]           # Bottom-left
        ], dtype=np.int32)
    else:
        # Restricted safe zone far off in the top-left corner
        pts = np.array([
            [int(width * 0.05), int(height * 0.05)],
            [int(width * 0.30), int(height * 0.05)],
            [int(width * 0.30), int(height * 0.25)],
            [int(width * 0.05), int(height * 0.25)]
        ], dtype=np.int32)

    return pts.reshape((-1, 1, 2))


def check_person_intrusion(
    person_box: List[int],
    roi_polygon: np.ndarray,
    use_bottom_center: bool = True
) -> Tuple[bool, Tuple[int, int], float]:
    """
    Determine if a detected person violates the polygon security zone using cv2.pointPolygonTest.

    Args:
        person_box: [x1, y1, x2, y2] bounding box coordinates.
        roi_polygon: Polygon contour array.
        use_bottom_center: If True, uses ground contact point (x_c, y2).
                           If False, uses geometric centroid (x_c, y_c).

    Returns:
        is_inside: True if point is strictly inside or on the boundary.
        anchor_point: (x, y) coordinates of the evaluated anchor.
        distance: Signed distance from point to polygon border (> 0 inside, < 0 outside, 0 on edge).
    """
    x1, y1, x2, y2 = person_box
    xc = (x1 + x2) // 2

    if use_bottom_center:
        yc = y2  # Ground / foot level
    else:
        yc = (y1 + y2) // 2  # Centroid

    anchor_point = (int(xc), int(yc))

    # cv2.pointPolygonTest: measureDist=True returns signed Euclidean distance
    dist = cv2.pointPolygonTest(roi_polygon, (float(xc), float(yc)), measureDist=True)
    is_inside = (dist >= 0.0)

    return is_inside, anchor_point, float(dist)


def annotate_security_frame(
    image: np.ndarray,
    roi_polygon: np.ndarray,
    person_detections: List[Dict[str, Any]],
    intrusion_detected: bool
) -> np.ndarray:
    """
    Annotate frame with colored ROI polygon, anchor crosshairs, bounding boxes,
    and a top status banner.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Theme colors (BGR)
    if intrusion_detected:
        theme_color = (0, 0, 235)      # Vivid Crimson Red
        banner_title = "INTRUSION DETECTED"
        sub_text = "SECURITY PERIMETER BREACHED - ZONE COMPROMISED"
    else:
        theme_color = (46, 204, 113)   # Vivid Emerald Green
        banner_title = "SYSTEM SECURE"
        sub_text = "ALL PERSON TRACKS OUTSIDE RESTRICTED SECURITY PERIMETER"

    # 1. Draw Semi-transparent ROI zone fill
    roi_overlay = annotated.copy()
    cv2.fillPoly(roi_overlay, [roi_polygon], theme_color)
    cv2.addWeighted(roi_overlay, 0.22, annotated, 0.78, 0, annotated)

    # 2. Draw Solid ROI perimeter border
    cv2.polylines(annotated, [roi_polygon], isClosed=True, color=theme_color, thickness=3, lineType=cv2.LINE_AA)

    # Label ROI corner
    roi_top_left = roi_polygon[0][0]
    cv2.putText(
        annotated, "[ RESTRICTED ZONE ]", (roi_top_left[0] + 5, roi_top_left[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2, cv2.LINE_AA
    )

    # 3. Draw Detected Persons and Anchor Points
    for det in person_detections:
        x1, y1, x2, y2 = det["bbox"]
        inside = det["is_inside"]
        pt = det["anchor_point"]
        conf = det["confidence"]

        box_color = (0, 0, 255) if inside else (0, 220, 0)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Ground anchor crosshair and circle
        cv2.circle(annotated, pt, 6, box_color, -1)
        cv2.circle(annotated, pt, 11, (255, 255, 255), 2)
        cv2.drawMarker(annotated, pt, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        # Label tag
        status_label = "INTRUDER" if inside else "AUTHORIZED"
        label_text = f"{status_label} ({conf:.2f})"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y = max(y1, th + 8)
        cv2.rectangle(annotated, (x1, tag_y - th - 6), (x1 + tw + 6, tag_y + baseline), box_color, -1)
        cv2.putText(
            annotated, label_text, (x1 + 3, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    # 4. Render Dark Top Status Banner
    banner_h = 70
    banner_overlay = annotated.copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(banner_overlay, 0.88, annotated, 0.12, 0, annotated)

    # Banner accent stripe
    cv2.rectangle(annotated, (0, banner_h - 4), (w, banner_h), theme_color, -1)

    # Banner Text
    cv2.putText(
        annotated, banner_title, (25, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, theme_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        annotated, sub_text, (27, 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA
    )

    return annotated


def run_intrusion_pipeline(
    input_filename: str = "person.jpg",
    output_filename: str = "task6_output.jpg",
    conf_threshold: float = 0.50
):
    """
    Execute Task 6 Perimeter Intrusion Detection Pipeline:
    1. Load image and ensure dataset
    2. Define 4-point polygon ROI security perimeter
    3. Run YOLOv8 detection for person (class ID 0)
    4. Compute bottom-center ground coordinates
    5. Perform point-in-polygon tests
    6. Render status-adaptive banners and colored overlays
    7. Save final output to outputs/task6_output.jpg
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    logger.info(f"Loading surveillance feed: {image_path}")
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    h, w = bgr_img.shape[:2]

    # Step 1: Define 4-point polygon ROI
    roi_polygon = default_security_roi(w, h, mode="intrusion")

    # Step 2: YOLOv8 person detection (Class 0)
    logger.info("Running YOLOv8 person detection model...")
    model = YOLO("yolov8n.pt")
    results = model(bgr_img, conf=conf_threshold, verbose=False)

    person_detections: List[Dict[str, Any]] = []
    overall_intrusion = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        if cls_id == 0:  # Class ID 0 is 'person' in COCO
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Step 3 & 4: Calculate bottom-center coordinate & test polygon inclusion
            is_inside, anchor_point, dist = check_person_intrusion([x1, y1, x2, y2], roi_polygon)

            if is_inside:
                overall_intrusion = True

            person_detections.append({
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "anchor_point": anchor_point,
                "is_inside": is_inside,
                "signed_distance": dist
            })

    # Terminal Analytics Log
    status_str = "INTRUSION DETECTED" if overall_intrusion else "SYSTEM SECURE"
    print("\n" + "=" * 70)
    print(" TASK 6: SMART SECURITY INTRUSION ALERT REPORT ")
    print("=" * 70)
    print(f" • Perimeter Status           : {status_str}")
    print(f" • Frame Resolution           : {w} x {h} px")
    print(f" • Total Persons Detected     : {len(person_detections)}")
    print(f" • Polygon ROI Vertices (4-pts):")
    for i, pt in enumerate(roi_polygon.reshape(-1, 2), start=1):
        print(f"     Point #{i}: ({pt[0]}, {pt[1]})")
    print("-" * 70)
    print(f"{'Person #':<10} | {'Conf':<6} | {'Anchor (x,y)':<16} | {'Dist to ROI':<14} | {'Status'}")
    print("-" * 70)
    for idx, p in enumerate(person_detections, start=1):
        stat = "INTRUSION" if p["is_inside"] else "SECURE"
        print(f"#{idx:<9} | {p['confidence']:<6.2f} | {str(p['anchor_point']):<16} | {p['signed_distance']:<14.2f} | {stat}")
    print("=" * 70 + "\n")

    # Step 5 & 6: Annotate frame with adaptive colors and banners
    annotated_frame = annotate_security_frame(
        bgr_img,
        roi_polygon=roi_polygon,
        person_detections=person_detections,
        intrusion_detected=overall_intrusion
    )

    # Step 7: Save annotated result to outputs/task6_output.jpg
    cv2.imwrite(output_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    logger.info(f"Successfully saved Task 6 security snapshot to: {output_path}")

    # Also save complementary PNG visualization
    png_path = output_path.replace(".jpg", ".png")
    cv2.imwrite(png_path, annotated_frame)


if __name__ == "__main__":
    run_intrusion_pipeline()
