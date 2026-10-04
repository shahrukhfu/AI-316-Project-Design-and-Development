"""
Task 2: Industrial Defect Detection via Color-Space Transformation & Thresholding
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module performs surface defect and rust detection on industrial materials.
It transforms the image into HSV and LAB color spaces, isolates rust and corrosion
using tuned chromatic bounds, refines the binary mask using morphological operations
and thresholding, and extracts the defect via bitwise masking.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_datasets import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Task2_ColorSegmentation")


def convert_color_spaces(bgr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert standard BGR image into RGB, HSV, and LAB color spaces.

    Args:
        bgr_img: Input OpenCV image in BGR format.

    Returns:
        Tuple containing (rgb_img, hsv_img, lab_img).
    """
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    return rgb_img, hsv_img, lab_img


def segment_defects(
    hsv_img: np.ndarray,
    bgr_orig: np.ndarray,
    lower_hsv: Tuple[int, int, int] = (5, 50, 50),
    upper_hsv: Tuple[int, int, int] = (25, 255, 255),
    kernel_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Segment defects using HSV color bounds, thresholding, and morphological filtering.

    Args:
        hsv_img: HSV color space image.
        bgr_orig: Original BGR image for bitwise extraction.
        lower_hsv: Lower bound (Hue [0-179], Saturation [0-255], Value [0-255]).
        upper_hsv: Upper bound (Hue [0-179], Saturation [0-255], Value [0-255]).
        kernel_size: Kernel dimension for morphological noise cleanup.

    Returns:
        refined_mask: Binary mask of detected defects (255 for defect, 0 for clean).
        isolated_defects_rgb: RGB image of isolated defective areas.
        metrics: Statistical dictionary of detected defect parameters.
    """
    lower_bound = np.array(lower_hsv, dtype=np.uint8)
    upper_bound = np.array(upper_hsv, dtype=np.uint8)

    # Generate primary binary mask via inRange
    raw_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # Refine mask using binary thresholding
    _, thresh_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological filtering: Opening (removes noise) followed by Closing (fills small voids)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    # Bitwise-AND with original image to isolate defective regions
    isolated_bgr = cv2.bitwise_and(bgr_orig, bgr_orig, mask=refined_mask)
    isolated_rgb = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2RGB)

    # Calculate metrics
    total_pixels = refined_mask.size
    defect_pixels = int(np.count_nonzero(refined_mask))
    defect_percentage = (defect_pixels / total_pixels) * 100.0

    # Count defect contours
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_defects = [c for c in contours if cv2.contourArea(c) > 100]

    metrics = {
        "Total Surface Pixels": total_pixels,
        "Defective Pixels": defect_pixels,
        "Defect Area Ratio (%)": round(defect_percentage, 2),
        "Significant Defect Clusters": len(significant_defects),
        "Lower HSV Bound": list(lower_hsv),
        "Upper HSV Bound": list(upper_hsv)
    }

    return refined_mask, isolated_rgb, metrics


def run_defect_detection(
    input_filename: str = "defect.jpg",
    output_filename: str = "task2_output.png"
):
    """
    Execute complete Task 2 defect detection pipeline and generate 2x2 visualization grid.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    logger.info(f"Loading industrial inspection sample: {image_path}")
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Color space transformations
    rgb_img, hsv_img, lab_img = convert_color_spaces(bgr_img)

    # Segment defects using specified HSV thresholds for rust / oxidation
    lower_bound = (5, 50, 50)
    upper_bound = (25, 255, 255)
    binary_mask, isolated_defect_rgb, metrics = segment_defects(
        hsv_img, bgr_img, lower_hsv=lower_bound, upper_hsv=upper_bound
    )

    print("\n" + "=" * 65)
    print(" TASK 2: INDUSTRIAL DEFECT & RUST SEGMENTATION METRICS ")
    print("=" * 65)
    for k, v in metrics.items():
        print(f" • {k:<30}: {v}")
    print("=" * 65 + "\n")

    # Matplotlib 2x2 Grid Visualization
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)

    # 1. Original Image (RGB)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("(1) Original Inspection Specimen (RGB)", fontsize=12, fontweight="bold", color="#61afef")
    axes[0, 0].set_xlabel(f"Resolution: {rgb_img.shape[1]} x {rgb_img.shape[0]} px", fontsize=10)
    axes[0, 0].tick_params(colors="gray")

    # 2. HSV Representation
    # Display HSV mapped as RGB representation for visual contrast of chromatic channels
    axes[0, 1].imshow(hsv_img)
    axes[0, 1].set_title("(2) HSV Color Space Representation", fontsize=12, fontweight="bold", color="#d19a66")
    axes[0, 1].set_xlabel("Channels: Hue [0-179], Sat [0-255], Val [0-255]", fontsize=10)
    axes[0, 1].tick_params(colors="gray")

    # 3. Binary Defect Mask
    axes[1, 0].imshow(binary_mask, cmap="gray")
    axes[1, 0].set_title(
        f"(3) Refined Binary Mask (Defect Area: {metrics['Defect Area Ratio (%)']}%)",
        fontsize=12, fontweight="bold", color="#e5c07b"
    )
    axes[1, 0].set_xlabel(f"Defect Pixels: {metrics['Defective Pixels']:,} / {metrics['Total Surface Pixels']:,}", fontsize=10)
    axes[1, 0].tick_params(colors="gray")

    # 4. Isolated Defect Output
    axes[1, 1].imshow(isolated_defect_rgb)
    axes[1, 1].set_title(
        f"(4) Isolated Defect Extraction (Bitwise-AND)\nClusters Identified: {metrics['Significant Defect Clusters']}",
        fontsize=12, fontweight="bold", color="#e06c75"
    )
    axes[1, 1].set_xlabel("Defects Isolated on Neutral Background", fontsize=10)
    axes[1, 1].tick_params(colors="gray")

    plt.suptitle(
        "Lab 03 - Task 2: Industrial Defect Detection via HSV Color Segmentation & Thresholding",
        fontsize=14, fontweight="bold", y=0.98, color="#98c379"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Successfully generated Task 2 visualization: {output_path}")


if __name__ == "__main__":
    run_defect_detection()
