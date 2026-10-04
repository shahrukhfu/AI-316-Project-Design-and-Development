"""
Task 1: Automated Preprocessing Pipeline for Traffic Camera Surveillance
Course: AI Project Design and Development (AI-316) - Air University Islamabad
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module implements an automated preprocessing pipeline tailored for deep
learning vision models (such as YOLOv8). It includes aspect-ratio preserving
letterbox resizing, border padding, channel color conversion, and tensor normalization.
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
logger = logging.getLogger("Task1_Preprocessing")


def letterbox_image(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target shape maintaining aspect ratio with symmetric border padding (letterbox).

    Args:
        image: Original BGR or RGB image (H, W, C).
        target_shape: Desired output dimensions (height, width). Default (640, 640).
        color: Padding fill color in BGR/RGB tuple. Default (114, 114, 114) for YOLO neutral gray.

    Returns:
        padded_image: Letterboxed image of shape (target_h, target_w, C).
        ratio: Scaling ratio applied to the original image.
        (pad_w, pad_h): Padding added to width and height (half per side).
    """
    h_orig, w_orig = image.shape[:2]
    target_h, target_w = target_shape

    # Calculate scaling ratio (fit within target boundaries)
    ratio = min(target_w / w_orig, target_h / h_orig)
    new_unpad_w = int(round(w_orig * ratio))
    new_unpad_h = int(round(h_orig * ratio))

    # Resize using area interpolation for downsampling or linear for upsampling
    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=interp)

    # Compute total padding required to reach target dimensions
    dw = target_w - new_unpad_w
    dh = target_h - new_unpad_h

    # Divide padding equally between sides
    pad_left = dw // 2
    pad_right = dw - pad_left
    pad_top = dh // 2
    pad_bottom = dh - pad_top

    # Apply constant border padding
    padded_image = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded_image, ratio, (pad_left, pad_top)


def normalize_tensor(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] uint8 to [0.0, 1.0] float32.

    Args:
        image: uint8 numpy image array.

    Returns:
        float32 normalized image array in range [0.0, 1.0].
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def compute_tensor_statistics(tensor: np.ndarray) -> Dict[str, Any]:
    """
    Compute key statistical parameters of the preprocessed tensor.

    Args:
        tensor: Image numpy array.

    Returns:
        Dictionary containing shape, dtype, min, max, mean, and std.
    """
    stats = {
        "Shape": tensor.shape,
        "Data Type": str(tensor.dtype),
        "Min Value": float(np.min(tensor)),
        "Max Value": float(np.max(tensor)),
        "Mean Value": float(np.mean(tensor)),
        "Standard Deviation": float(np.std(tensor))
    }
    return stats


def run_pipeline(
    input_filename: str = "traffic.jpg",
    output_filename: str = "task1_output.png"
):
    """
    Execute complete Task 1 preprocessing workflow:
    1. Load image
    2. Letterbox resize to 640x640 with padding
    3. Normalize to [0.0, 1.0]
    4. Print statistical summary
    5. Save visual comparison plot
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "data", input_filename)
    output_path = os.path.join(base_dir, "outputs", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure dataset is available
    if not os.path.exists(image_path):
        ensure_dataset(input_filename)

    logger.info(f"Loading input traffic image: {image_path}")
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Failed to read image from {image_path}")

    # Convert to RGB for proper color display
    rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Apply Letterboxing to standard 640x640 deep learning input size
    letterboxed_rgb, scale_ratio, (pad_w, pad_h) = letterbox_image(
        rgb_orig, target_shape=(640, 640), color=(114, 114, 114)
    )

    # Normalize pixel intensity to [0.0, 1.0] float32
    normalized_tensor = normalize_tensor(letterboxed_rgb)

    # Compute and display pixel tensor statistics
    stats = compute_tensor_statistics(normalized_tensor)

    print("\n" + "=" * 60)
    print(" TASK 1: PREPROCESSED TENSOR STATISTICS ")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f" • {k:<20}: {v:.6f}")
        else:
            print(f" • {k:<20}: {v}")
    print(f" • Scale Ratio         : {scale_ratio:.4f}")
    print(f" • Padding (W, H)      : ({pad_w}, {pad_h}) px")
    print("=" * 60 + "\n")

    # Matplotlib visualization: Side-by-side comparison
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Original Image Plot
    axes[0].imshow(rgb_orig)
    axes[0].set_title(
        f"Original Traffic Image\nDimensions: {rgb_orig.shape[1]}x{rgb_orig.shape[0]} | Dtype: {rgb_orig.dtype}",
        fontsize=12, fontweight="bold", pad=10, color="#61afef"
    )
    axes[0].set_xlabel("Width (pixels)", fontsize=10)
    axes[0].set_ylabel("Height (pixels)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.3)

    # Preprocessed Letterboxed & Normalized Plot
    axes[1].imshow(normalized_tensor)
    axes[1].set_title(
        f"Preprocessed Output (Letterboxed & Normalized)\nDimensions: {normalized_tensor.shape[1]}x{normalized_tensor.shape[0]} | Range: [{stats['Min Value']:.2f}, {stats['Max Value']:.2f}]",
        fontsize=12, fontweight="bold", pad=10, color="#98c379"
    )
    axes[1].set_xlabel("Width (pixels)", fontsize=10)
    axes[1].set_ylabel("Height (pixels)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.3)

    plt.suptitle(
        "Lab 03 - Task 1: Traffic Camera Preprocessing Pipeline (Letterbox 640x640 & [0,1] Norm)",
        fontsize=14, fontweight="bold", y=0.98, color="#e5c07b"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Successfully saved Task 1 visual output to: {output_path}")


if __name__ == "__main__":
    run_pipeline()
