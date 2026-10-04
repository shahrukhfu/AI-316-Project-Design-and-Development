"""
Task 3: Edge Detection & Noise Reduction for Medical Imaging
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module performs edge detection and noise suppression on medical grayscale scans (Chest X-Ray).
It evaluates Gaussian Blurring and Median Filtering, computes directional Sobel derivatives
(Sobel-X and Sobel-Y), computes the Euclidean gradient magnitude, and applies multi-stage
Canny Edge Detection with dual hysteresis thresholding.
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
logger = logging.getLogger("Task3_EdgeDetection")


def apply_noise_reduction(
    gray_img: np.ndarray,
    gaussian_ksize: Tuple[int, int] = (5, 5),
    median_ksize: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian and Median filtering for noise suppression.

    Args:
        gray_img: 2D grayscale image array.
        gaussian_ksize: Kernel dimensions for Gaussian Blur.
        median_ksize: Kernel aperture size for Median Filter (must be odd).

    Returns:
        Tuple containing (gaussian_blurred, median_filtered).
    """
    gaussian_blurred = cv2.GaussianBlur(gray_img, gaussian_ksize, sigmaX=1.0)
    median_filtered = cv2.medianBlur(gray_img, median_ksize)
    return gaussian_blurred, median_filtered


def compute_sobel_gradients(filtered_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute directional Sobel gradients (Ix, Iy) and gradient magnitude.

    Gradient magnitude is computed as:
        G = sqrt(Ix^2 + Iy^2)

    Args:
        filtered_img: 2D smoothed grayscale image.

    Returns:
        Tuple containing:
            sobel_x_abs: Normalized absolute horizontal gradient (8-bit).
            sobel_y_abs: Normalized absolute vertical gradient (8-bit).
            sobel_mag_abs: Euclidean gradient magnitude (8-bit).
    """
    # Use 64-bit float to preserve negative slopes and prevent underflow
    sobel_x = cv2.Sobel(filtered_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(filtered_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Compute Euclidean gradient magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scale back to 8-bit [0, 255]
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_mag_abs = cv2.convertScaleAbs(sobel_magnitude)

    return sobel_x_abs, sobel_y_abs, sobel_mag_abs


def compute_canny_edges(
    filtered_img: np.ndarray,
    t_lower: int = 50,
    t_upper: int = 150
) -> np.ndarray:
    """
    Apply Canny Edge Detection with hysteresis thresholds.

    Args:
        filtered_img: Smoothed grayscale image.
        t_lower: Lower hysteresis threshold (weak edges).
        t_upper: Upper hysteresis threshold (strong edges).

    Returns:
        Binary Canny edge map (255 for edge, 0 for background).
    """
    canny_edges = cv2.Canny(filtered_img, threshold1=t_lower, threshold2=t_upper, L2gradient=True)
    return canny_edges


def run_pipeline(
    input_filename: str = "xray.jpg",
    output_filename: str = "task3_output.png"
):
    """
    Execute complete Task 3 medical edge detection pipeline:
    1. Load in grayscale mode
    2. Compare Gaussian vs Median filtering
    3. Calculate Sobel X, Y, and Magnitude
    4. Compute Canny Edge Detection (50, 150)
    5. Save 6-panel comparison figure
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    logger.info(f"Loading medical grayscale scan: {image_path}")
    raw_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Downsample if ultra-high res for display efficiency while maintaining fine details
    if raw_gray.shape[0] > 1200 or raw_gray.shape[1] > 1200:
        scale = 1000.0 / max(raw_gray.shape)
        gray = cv2.resize(raw_gray, (int(raw_gray.shape[1] * scale), int(raw_gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray = raw_gray

    # Step 1: Compare noise filters
    gaussian_img, median_img = apply_noise_reduction(gray, gaussian_ksize=(5, 5), median_ksize=5)

    # Step 2: Sobel Derivatives and Euclidean Gradient Magnitude
    sobel_x, sobel_y, sobel_mag = compute_sobel_gradients(gaussian_img)

    # Step 3: Canny Edge Detection with Hysteresis (50, 150)
    canny_edges = compute_canny_edges(gaussian_img, t_lower=50, t_upper=150)

    # Compute quantitative metrics
    total_pixels = gray.size
    canny_edge_pixels = int(np.count_nonzero(canny_edges))
    canny_edge_density = (canny_edge_pixels / total_pixels) * 100.0
    mean_gradient = float(np.mean(sobel_mag))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    print("\n" + "=" * 65)
    print(" TASK 3: MEDICAL EDGE DETECTION & FILTERING METRICS ")
    print("=" * 65)
    print(f" • Input Image Shape             : {gray.shape[1]} x {gray.shape[0]} px")
    print(f" • Image Focus Measure (VarLap)   : {laplacian_variance:.2f}")
    print(f" • Mean Gradient Magnitude       : {mean_gradient:.2f}")
    print(f" • Canny Edge Pixel Count        : {canny_edge_pixels:,}")
    print(f" • Canny Edge Density            : {canny_edge_density:.2f}%")
    print(f" • Canny Hysteresis Bounds       : Lower=50, Upper=150")
    print("=" * 65 + "\n")

    # Plot 6-panel Matplotlib comparison (2 rows x 3 columns)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    # 1. Original Grayscale Scan
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title(
        f"(1) Original Chest X-Ray\nRes: {gray.shape[1]}x{gray.shape[0]} | Range: [{np.min(gray)}, {np.max(gray)}]",
        fontsize=11, fontweight="bold", color="#61afef"
    )
    axes[0, 0].axis("off")

    # 2. Filtering Comparison (Gaussian vs Median difference or Gaussian Denoised)
    # We display Gaussian Blurred which is passed downstream to edge detectors
    axes[0, 1].imshow(gaussian_img, cmap="gray")
    axes[0, 1].set_title(
        "(2) Gaussian Filtered (5x5, \u03c3=1.0)\nNoise Suppressed & Anatomical Borders Preserved",
        fontsize=11, fontweight="bold", color="#98c379"
    )
    axes[0, 1].axis("off")

    # 3. Sobel Horizontal Derivative (Ix)
    axes[0, 2].imshow(sobel_x, cmap="gray")
    axes[0, 2].set_title(
        r"(3) Sobel Horizontal Gradient ($I_x$)" + "\nVertical Structures (Rib Borders / Spine)",
        fontsize=11, fontweight="bold", color="#e5c07b"
    )
    axes[0, 2].axis("off")

    # 4. Sobel Vertical Derivative (Iy)
    axes[1, 0].imshow(sobel_y, cmap="gray")
    axes[1, 0].set_title(
        r"(4) Sobel Vertical Gradient ($I_y$)" + "\nHorizontal Structures (Clavicle / Diaphragm)",
        fontsize=11, fontweight="bold", color="#e5c07b"
    )
    axes[1, 0].axis("off")

    # 5. Sobel Euclidean Gradient Magnitude
    axes[1, 1].imshow(sobel_mag, cmap="inferno")
    axes[1, 1].set_title(
        r"(5) Sobel Gradient Magnitude ($\sqrt{I_x^2 + I_y^2}$)" + f"\nMean Magnitude: {mean_gradient:.2f}",
        fontsize=11, fontweight="bold", color="#d19a66"
    )
    axes[1, 1].axis("off")

    # 6. Canny Edge Detection
    axes[1, 2].imshow(canny_edges, cmap="gray")
    axes[1, 2].set_title(
        f"(6) Canny Edge Detection\nDual Thresholds: [50, 150] | Edge Density: {canny_edge_density:.2f}%",
        fontsize=11, fontweight="bold", color="#e06c75"
    )
    axes[1, 2].axis("off")

    plt.suptitle(
        "Lab 03 - Task 3: Edge Detection & Noise Reduction for Medical Imaging (Chest X-Ray)",
        fontsize=14, fontweight="bold", y=0.98, color="#56b6c2"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Successfully saved Task 3 6-panel visualization to: {output_path}")


if __name__ == "__main__":
    run_pipeline()
