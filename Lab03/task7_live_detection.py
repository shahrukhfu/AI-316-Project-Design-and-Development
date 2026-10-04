"""
Task 7: Real-Time Webcam Multi-Object Detection Dashboard
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module implements a real-time computer vision dashboard using a live webcam feed.
It continuously runs YOLOv8 multi-object inference, calculates and renders real-time
Frames Per Second (FPS), provides keyboard event listeners ('q' to quit, 's' to capture
snapshots to outputs/webcam_snapshot.jpg), and ensures robust hardware resource cleanup.
"""

import os
import sys
import time
import logging
from typing import Tuple, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task7_LiveDetection")


class FPSCalculator:
    """Calculates smoothed real-time Frames Per Second (FPS)."""

    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.prev_time = time.perf_counter()
        self.smoothed_fps = 0.0

    def update(self) -> float:
        current_time = time.perf_counter()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0:
            instant_fps = 1.0 / delta
            if self.smoothed_fps == 0.0:
                self.smoothed_fps = instant_fps
            else:
                self.smoothed_fps = (self.smoothing_factor * self.smoothed_fps) + ((1.0 - self.smoothing_factor) * instant_fps)

        return self.smoothed_fps


def render_dashboard_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    detected_count: int,
    snapshot_notice_time: float
) -> np.ndarray:
    """
    Render high-contrast top-left Heads-Up Display (HUD) showing FPS, latency, and controls.

    Args:
        frame: Current video frame (BGR).
        fps: Current real-time FPS value.
        inference_ms: Inference duration in milliseconds.
        detected_count: Number of objects detected in current frame.
        snapshot_notice_time: Timestamp of last snapshot saved (for visual confirmation).

    Returns:
        Annotated BGR frame.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Draw semi-transparent HUD panel in top-left
    hud_w = 320
    hud_h = 105
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Accent left border
    cv2.rectangle(annotated, (10, 10), (15, 10 + hud_h), (0, 200, 255), -1)

    # Render FPS
    fps_color = (0, 255, 100) if fps >= 20.0 else ((0, 200, 255) if fps >= 10.0 else (0, 100, 255))
    cv2.putText(
        annotated, f"FPS: {fps:.1f}", (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_color, 2, cv2.LINE_AA
    )

    # Render Latency & Object Count
    stats_text = f"Latency: {inference_ms:.1f} ms | Objects: {detected_count}"
    cv2.putText(
        annotated, stats_text, (25, 66),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
    )

    # Render Keyboard Controls Guide
    controls_text = "Controls: [Q] Exit  |  [S] Save Snapshot"
    cv2.putText(
        annotated, controls_text, (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA
    )

    # Display temporary "SNAPSHOT SAVED!" notification if triggered recently
    if time.time() - snapshot_notice_time < 1.5:
        notice_box_w = 260
        notice_box_h = 45
        nx = (w - notice_box_w) // 2
        ny = 25
        cv2.rectangle(annotated, (nx, ny), (nx + notice_box_w, ny + notice_box_h), (0, 180, 0), -1)
        cv2.putText(
            annotated, "SNAPSHOT SAVED!", (nx + 28, ny + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

    return annotated


def run_live_detection(
    source: str = "0",
    model_weight: str = "yolov8n.pt",
    conf_threshold: float = 0.45,
    max_frames: Optional[int] = None
):
    """
    Run real-time webcam detection loop.

    Args:
        source: Camera device index ("0") or video/image filepath.
        model_weight: YOLO model weights to load (default 'yolov8n.pt').
        conf_threshold: Minimum detection confidence threshold.
        max_frames: Optional frame limit (useful for non-interactive automated testing).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    snapshot_path = os.path.join(outputs_dir, "webcam_snapshot.jpg")

    logger.info(f"Loading YOLO model weights: {model_weight}...")
    model = YOLO(model_weight)

    # Determine whether input is hardware webcam or simulation file
    is_simulation = False
    cap = None

    if source.isdigit():
        cam_idx = int(source)
        logger.info(f"Initializing live video capture on camera index {cam_idx}...")
        try:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)
        except Exception as e:
            logger.warning(f"Could not open camera index {cam_idx}: {e}")
            cap = None

        if cap is None or not cap.isOpened():
            logger.warning(
                f"Webcam index {cam_idx} is unavailable. "
                "Switching to synthetic stream on data/street.jpg for demonstration."
            )
            is_simulation = True
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        logger.info(f"Loading input video/image source: {source}")
        if os.path.exists(source) and any(source.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            is_simulation = True
            fallback_img_path = source
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                is_simulation = True

    if is_simulation:
        fallback_img_path = os.path.join(base_dir, "data", "street.jpg")
        if not os.path.exists(fallback_img_path):
            ensure_dataset("street.jpg")
        fallback_frame = cv2.imread(fallback_img_path)
        logger.info(f"Stream simulation initialized using {fallback_img_path}")

    fps_meter = FPSCalculator()
    snapshot_notice_time = 0.0
    frame_counter = 0

    print("\n" + "=" * 65)
    print(" TASK 7: REAL-TIME WEBCAM DETECTION DASHBOARD ")
    print("=" * 65)
    print(" • Press 'q' to gracefully release camera and exit.")
    print(f" • Press 's' to capture snapshot to: {snapshot_path}")
    print("=" * 65 + "\n")

    try:
        while True:
            if is_simulation:
                frame = fallback_frame.copy()
                time.sleep(0.02)  # Simulate frame timing
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("End of video stream or failed to grab frame.")
                    break

            frame_counter += 1

            # Execute YOLO inference
            t_start = time.perf_counter()
            results = model(frame, conf=conf_threshold, verbose=False)
            t_infer = (time.perf_counter() - t_start) * 1000.0

            # Render YOLO detection bounding boxes
            annotated_frame = results[0].plot()

            # Update FPS
            current_fps = fps_meter.update()
            detected_count = len(results[0].boxes)

            # Render Heads-Up Display
            dashboard_frame = render_dashboard_hud(
                annotated_frame,
                fps=current_fps,
                inference_ms=t_infer,
                detected_count=detected_count,
                snapshot_notice_time=snapshot_notice_time
            )

            # Display window (handled safely if headless environment)
            try:
                cv2.imshow("Air University - Lab 03: Real-Time YOLO Dashboard", dashboard_frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 0xFF

            # Handle Keyboard Input: 'q' -> Exit
            if key == ord('q') or key == 27:
                logger.info("Exit key pressed. Terminating feed gracefully...")
                break

            # Handle Keyboard Input: 's' -> Save Snapshot
            if key == ord('s'):
                cv2.imwrite(snapshot_path, dashboard_frame)
                snapshot_notice_time = time.time()
                logger.info(f"Snapshot saved to: {snapshot_path}")

            # Verification frame limit
            if max_frames is not None and frame_counter >= max_frames:
                cv2.imwrite(snapshot_path, dashboard_frame)
                logger.info(f"Completed {max_frames} verification frames. Snapshot saved to {snapshot_path}")
                break

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Released camera hardware resources.")
        cv2.destroyAllWindows()
        logger.info("Closed all OpenCV display windows.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 7: Real-Time Webcam Detection Dashboard")
    parser.add_argument("--source", type=str, default="0", help="Camera index ('0') or video/image path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process before exiting")
    parser.add_argument("--test", action="store_true", help="Run automated test simulation")

    args = parser.parse_args()

    if args.test:
        run_live_detection(source="data/street.jpg", model_weight=args.model, conf_threshold=args.conf, max_frames=15)
    else:
        run_live_detection(source=args.source, model_weight=args.model, conf_threshold=args.conf, max_frames=args.frames)
