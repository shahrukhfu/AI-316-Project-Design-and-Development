"""
Automated Dataset Downloader
Air University Islamabad - Department of Creative Technologies
Course: AI Project Design and Development (AI-316)
Lab 03: Computer Vision Prototyping & Object Detection Pipeline

This module fetches high-quality test images required for all tasks in Lab 03.
It provides automatic verification, fallbacks, and directory structure creation.
"""

import os
import sys
import logging
import urllib.request
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DatasetDownloader")

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")

# Primary and mirror URLs for robust fetching
DATASET_URLS = {
    "traffic.jpg": [
        "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=1280&q=80",
        "https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg?auto=compress&cs=tinysrgb&w=1280",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    ],
    "defect.jpg": [
        "https://images.unsplash.com/photo-1589939705384-5185137a7f0f?auto=format&fit=crop&w=1280&q=80",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Rust_on_iron.jpg/1280px-Rust_on_iron.jpg",
        "https://images.unsplash.com/photo-1533035353720-f1c6a75cd8ab?auto=format&fit=crop&w=1280&q=80"
    ],
    "xray.jpg": [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFBC-5BA8230AE942.jpeg",
        "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=1280&q=80"
    ],
    "street.jpg": [
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "https://images.unsplash.com/photo-1477959858617-67f30bc75b82?auto=format&fit=crop&w=1280&q=80"
    ],
    "retail.jpg": [
        "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?auto=format&fit=crop&w=1280&q=80",
        "https://images.unsplash.com/photo-1610832958506-aa56368176cf?auto=format&fit=crop&w=1280&q=80"
    ],
    "person.jpg": [
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=1280&q=80"
    ]
}


def create_required_directories():
    """Ensure data/, outputs/, and alerts/ directories exist."""
    for folder in [DATA_DIR, OUTPUTS_DIR, ALERTS_DIR]:
        os.makedirs(folder, exist_ok=True)
    logger.info("Checked/created directories: data/, outputs/, alerts/")


def _download_file_with_headers(url: str, dest_path: str, timeout: int = 15) -> bool:
    """Download a file with user-agent headers and verify it is a readable image."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            with open(dest_path, "wb") as f:
                f.write(content)

        # Verify readability with OpenCV
        test_img = cv2.imread(dest_path)
        if test_img is not None and test_img.size > 0:
            return True
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    except Exception as e:
        logger.debug(f"Attempt failed for URL {url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def _generate_synthetic_image(filename: str, dest_path: str):
    """
    Fallback generator if network is completely offline.
    Synthesizes valid images with appropriate features (edges, color patches, shapes).
    """
    logger.warning(f"Generating synthetic fallback image for {filename}...")
    np.random.seed(42)
    h, w = 720, 1280

    if filename == "traffic.jpg":
        img = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
        # Road markings
        cv2.line(img, (w // 2, h), (w // 2, h // 2), (255, 255, 255), 8)
        # Vehicles
        cv2.rectangle(img, (200, 450), (450, 600), (0, 0, 200), -1)  # Red car
        cv2.rectangle(img, (750, 350), (1050, 520), (220, 150, 50), -1)  # Blue SUV
        cv2.putText(img, "TRAFFIC CAMERA SIMULATION", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    elif filename == "defect.jpg":
        # Metallic gray background with rust spots (orange/brown HSV: H ~ 10-25)
        img = np.full((h, w, 3), (170, 170, 175), dtype=np.uint8)
        # Add texture noise
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Add prominent rust / corrosion patches (BGR: high Red, moderate Green, low Blue)
        cv2.ellipse(img, (450, 320), (180, 90), 25, 0, 360, (25, 75, 185), -1)
        cv2.ellipse(img, (820, 480), (120, 160), -30, 0, 360, (20, 65, 170), -1)
        cv2.circle(img, (300, 520), 65, (30, 85, 195), -1)
        # Defect scratch
        cv2.line(img, (580, 200), (700, 380), (15, 45, 140), 6)
    elif filename == "xray.jpg":
        # Grayscale medical scan simulation
        gray = np.zeros((h, w), dtype=np.uint8)
        # Lung lobes
        cv2.ellipse(gray, (w // 2 - 250, h // 2), (180, 280), 0, 0, 360, 180, -1)
        cv2.ellipse(gray, (w // 2 + 250, h // 2), (180, 280), 0, 0, 360, 180, -1)
        # Ribs structure
        for y in range(160, 560, 55):
            cv2.ellipse(gray, (w // 2 - 240, y), (170, 25), -10, 0, 360, 230, 4)
            cv2.ellipse(gray, (w // 2 + 240, y), (170, 25), 10, 0, 360, 230, 4)
        # Spine
        cv2.rectangle(gray, (w // 2 - 35, 80), (w // 2 + 35, 660), 220, -1)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # Generic test scene with shapes
        img = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)
        cv2.putText(img, filename, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    cv2.imwrite(dest_path, img)
    logger.info(f"Generated synthetic image for {filename} -> {dest_path}")


def download_single_image(filename: str, force_download: bool = False) -> str:
    """Download or verify a single image in data/."""
    create_required_directories()
    dest_path = os.path.join(DATA_DIR, filename)

    if not force_download and os.path.exists(dest_path):
        # Validate readability
        img = cv2.imread(dest_path)
        if img is not None and img.size > 0:
            logger.info(f"Image '{filename}' already exists and is valid at {dest_path}")
            return dest_path
        else:
            logger.warning(f"Existing file {dest_path} is corrupt. Re-downloading...")
            os.remove(dest_path)

    urls = DATASET_URLS.get(filename, [])
    success = False
    for url in urls:
        logger.info(f"Downloading '{filename}' from: {url}")
        if _download_file_with_headers(url, dest_path):
            logger.info(f"Successfully saved valid image: {dest_path}")
            success = True
            break

    if not success:
        logger.warning(f"Could not download {filename} from remote URLs. Using fallback...")
        _generate_synthetic_image(filename, dest_path)

    return dest_path


def download_all_datasets(force_download: bool = False):
    """Download all required images for Lab 03."""
    logger.info("=== Starting Dataset Verification & Download for Lab 03 ===")
    create_required_directories()
    for filename in DATASET_URLS:
        download_single_image(filename, force_download=force_download)
    logger.info("=== All Datasets Are Ready in data/ ===")


def ensure_dataset(filename: str) -> str:
    """Helper for task scripts to guarantee image availability."""
    return download_single_image(filename, force_download=False)


if __name__ == "__main__":
    download_all_datasets()
