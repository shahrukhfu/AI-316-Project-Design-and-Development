"""
Task 8: Edge-Triggered Webcam Surveillance System with Log Generation
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module implements an autonomous edge security surveillance system.
It monitors live video feed via YOLOv8, detects high-priority target classes
(e.g., 'person' and 'cell phone') exceeding confidence threshold > 0.65, captures
timestamped event snapshots to the alerts/ directory, appends audit events to events.log,
and renders a visual red recording indicator ('● REC / LOGGING ACTIVE') on the video stream.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task8_SurveillanceLogger")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
LOG_FILE_PATH = os.path.join(BASE_DIR, "events.log")


def initialize_event_logger():
    """Ensure alerts/ directory and events.log with header exist."""
    os.makedirs(ALERTS_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE_PATH) or os.path.getsize(LOG_FILE_PATH) == 0:
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            header = f"{'Timestamp':<23} | {'Class Name':<15} | {'Confidence':<10} | {'Bounding Box [x1, y1, x2, y2]'}\n"
            divider = "-" * 85 + "\n"
            f.write(header)
            f.write(divider)
        logger.info(f"Initialized security events audit log at: {LOG_FILE_PATH}")


def log_surveillance_event(
    class_name: str,
    confidence: float,
    bbox: List[int],
    timestamp_str: Optional[str] = None
):
    """
    Append an alert entry to events.log.

    Format: Timestamp | Class Name | Confidence | Bounding Box [x1, y1, x2, y2]
    """
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"{timestamp_str:<23} | {class_name:<15} | {confidence:<10.2f} | {str(bbox)}\n"
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(entry)

    logger.info(f"SECURITY EVENT LOGGED: {class_name} ({confidence:.2f}) at {bbox}")


def render_surveillance_hud(
    frame: np.ndarray,
    is_logging_active: bool,
    active_threats: List[str],
    alert_count: int,
    fps: float = 0.0
) -> np.ndarray:
    """
    Render professional surveillance camera HUD with blinking recording banner.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Timestamp overlay in top-right
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (tw, th), _ = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        annotated, current_time_str, (w - tw - 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Status Banner in top-left
    if is_logging_active:
        # Red Alert Banner
        banner_w = 340
        banner_h = 75
        cv2.rectangle(annotated, (15, 15), (15 + banner_w, 15 + banner_h), (10, 10, 180), -1)
        cv2.rectangle(annotated, (15, 15), (15 + banner_w, 15 + banner_h), (0, 0, 255), 2)

        # Blinking REC dot (toggles every 500ms)
        blink_on = (int(time.time() * 2) % 2) == 0
        dot_color = (0, 0, 255) if blink_on else (50, 50, 200)
        cv2.circle(annotated, (35, 42), 8, dot_color, -1)
        cv2.circle(annotated, (35, 42), 11, (255, 255, 255), 1)

        cv2.putText(
            annotated, "REC / LOGGING ACTIVE", (55, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
        sub_text = f"Threats: {', '.join(set(active_threats))} | Total Alerts: {alert_count}"
        cv2.putText(
            annotated, sub_text, (25, 76),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 255), 1, cv2.LINE_AA
        )
    else:
        # Green Standby Banner
        banner_w = 280
        banner_h = 55
        cv2.rectangle(annotated, (15, 15), (15 + banner_w, 15 + banner_h), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, 15), (15 + banner_w, 15 + banner_h), (46, 204, 113), 1)

        cv2.circle(annotated, (35, 42), 6, (46, 204, 113), -1)
        cv2.putText(
            annotated, "STANDBY / MONITORING", (50, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (46, 204, 113), 1, cv2.LINE_AA
        )

    # Bottom controls guide
    guide_text = "Press 'q' to stop surveillance | Audit: events.log | Snapshots: alerts/"
    cv2.putText(
        annotated, guide_text, (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )

    return annotated


def run_surveillance_system(
    source: str = "0",
    model_weight: str = "yolov8n.pt",
    priority_classes: List[str] = ["person", "cell phone"],
    conf_threshold: float = 0.65,
    cooldown_seconds: float = 2.0,
    max_frames: Optional[int] = None
):
    """
    Run automated edge surveillance and event logging system.

    Args:
        source: Video capture source ("0" for webcam, or video/image path).
        model_weight: Weights for YOLO model.
        priority_classes: List of high-priority target class names to trigger alerts.
        conf_threshold: Minimum detection confidence threshold (> 0.65).
        cooldown_seconds: Minimum duration between snapshot captures to prevent disk flood.
        max_frames: Optional limit for automated testing.
    """
    initialize_event_logger()

    logger.info(f"Loading surveillance detection model ({model_weight})...")
    model = YOLO(model_weight)

    is_simulation = False
    cap = None

    if source.isdigit():
        cam_idx = int(source)
        logger.info(f"Accessing surveillance camera feed on index {cam_idx}...")
        try:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)
        except Exception as e:
            logger.warning(f"Failed to initialize camera {cam_idx}: {e}")
            cap = None

        if cap is None or not cap.isOpened():
            logger.warning(
                f"Camera {cam_idx} not detected. Falling back to data/person.jpg simulation."
            )
            is_simulation = True
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        logger.info(f"Loading video/image feed from source: {source}")
        if os.path.exists(source) and any(source.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            is_simulation = True
            fallback_path = source
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                is_simulation = True

    if is_simulation:
        fallback_path = os.path.join(BASE_DIR, "data", "person.jpg")
        if not os.path.exists(fallback_path):
            ensure_dataset("person.jpg")
        fallback_frame = cv2.imread(fallback_path)
        logger.info(f"Surveillance simulation feed active on {fallback_path}")

    last_snapshot_time = 0.0
    total_alerts_logged = 0
    frame_idx = 0

    print("\n" + "=" * 70)
    print(" TASK 8: EDGE-TRIGGERED WEBCAM SURVEILLANCE & LOGGING SYSTEM ")
    print("=" * 70)
    print(f" • Priority Alert Classes     : {priority_classes}")
    print(f" • Confidence Gate Threshold   : > {conf_threshold:.2f}")
    print(f" • Snapshots Directory        : {ALERTS_DIR}")
    print(f" • Audit Log File             : {LOG_FILE_PATH}")
    print(" • Press 'q' to stop surveillance.")
    print("=" * 70 + "\n")

    try:
        while True:
            if is_simulation:
                frame = fallback_frame.copy()
                time.sleep(0.03)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended or failed to read frame.")
                    break

            frame_idx += 1

            # Run YOLO inference
            results = model(frame, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            names = model.names

            triggered_events = []
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].cpu().numpy())

                # Check if detected object is a high-priority threat with conf > 0.65
                if cls_name.lower() in [c.lower() for c in priority_classes] and conf > conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    triggered_events.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": bbox
                    })

            is_logging_active = len(triggered_events) > 0
            current_time = time.time()
            now_dt = datetime.now()

            # If threats detected and cooldown expired: save snapshot and log events
            if is_logging_active:
                if current_time - last_snapshot_time >= cooldown_seconds:
                    # Timestamp format: YYYYMMDD_HHMMSS.jpg
                    ts_filename = now_dt.strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"{ts_filename}.jpg"
                    snapshot_full_path = os.path.join(ALERTS_DIR, snapshot_filename)

                    # Ensure unique filename if multiple snapshots occur in same second
                    counter = 1
                    while os.path.exists(snapshot_full_path):
                        snapshot_filename = f"{ts_filename}_{counter}.jpg"
                        snapshot_full_path = os.path.join(ALERTS_DIR, snapshot_filename)
                        counter += 1

                    # Save unannotated or HUD-annotated alert snapshot
                    cv2.imwrite(snapshot_full_path, frame)
                    last_snapshot_time = current_time
                    total_alerts_logged += 1
                    logger.info(f"Saved alert snapshot to: {snapshot_full_path}")

                    # Append entries to events.log
                    log_ts_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                    for ev in triggered_events:
                        log_surveillance_event(
                            class_name=ev["class_name"],
                            confidence=ev["confidence"],
                            bbox=ev["bbox"],
                            timestamp_str=log_ts_str
                        )

            # Draw detection boxes
            annotated_frame = frame.copy()
            for ev in triggered_events:
                x1, y1, x2, y2 = ev["bbox"]
                cls_name = ev["class_name"]
                conf = ev["confidence"]
                # Alert bounding box in Crimson Red
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 240), 2)
                tag = f"{cls_name.upper()}: {conf:.2f}"
                cv2.putText(
                    annotated_frame, tag, (x1, max(y1 - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 240), 2, cv2.LINE_AA
                )

            # Render Surveillance HUD with REC indicator
            threat_names = [ev["class_name"] for ev in triggered_events]
            dashboard_frame = render_surveillance_hud(
                annotated_frame,
                is_logging_active=is_logging_active,
                active_threats=threat_names,
                alert_count=total_alerts_logged
            )

            # Display feed safely
            try:
                cv2.imshow("Air University - Lab 03: Surveillance Logger System", dashboard_frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 0xFF

            # Exit cleanly on 'q'
            if key == ord('q') or key == 27:
                logger.info("Termination signal received. Exiting surveillance loop...")
                break

            # Automated test termination
            if max_frames is not None and frame_idx >= max_frames:
                logger.info(f"Test frame limit reached ({max_frames}). Completed successfully.")
                break

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Released camera hardware resources.")
        cv2.destroyAllWindows()
        logger.info(f"Closed all display windows. Total event alerts recorded: {total_alerts_logged}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 8: Edge-Triggered Webcam Surveillance System")
    parser.add_argument("--source", type=str, default="0", help="Camera index ('0') or video/image path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold (> 0.65)")
    parser.add_argument("--frames", type=int, default=None, help="Max frames for testing")
    parser.add_argument("--test", action="store_true", help="Run automated test simulation")

    args = parser.parse_args()

    if args.test:
        run_surveillance_system(source="data/person.jpg", model_weight=args.model, conf_threshold=args.conf, max_frames=15)
    else:
        run_surveillance_system(source=args.source, model_weight=args.model, conf_threshold=args.conf, max_frames=args.frames)
