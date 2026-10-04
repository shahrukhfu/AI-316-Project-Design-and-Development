"""
Quick Test Utility: Single Frame Conveyor Defect Inspection
============================================================
Allows testing the defect detection pipeline on any specific image frame
with custom confidence thresholds and automated diagnostic reporting.

Usage:
    # Run on default damaged strawberry sample:
    python test_single_frame.py

    # Run on any custom image path:
    python test_single_frame.py path/to/image.jpg

    # Run with custom confidence threshold (e.g. 0.30):
    python test_single_frame.py path/to/image.jpg --conf 0.30
"""

import sys
import argparse
from pathlib import Path

# Ensure src directory is in sys.path when executed directly
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import modular components from main pipeline script
from conveyor_fruit_defect_detection import (
    load_yolo_model,
    process_single_frame,
    find_dataset_samples
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Fruit Defect Inspection on a specific conveyor frame."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the conveyor frame image. If omitted, uses default sample."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inspection_outputs",
        help="Output directory to save the 2x4 visual diagnostic grid."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="YOLO model weights path or name (default: yolo26n.pt)."
    )

    args = parser.parse_args()

    # Determine image path: user argument or fall back to default damaged strawberry
    if args.image_path:
        test_path = Path(args.image_path)
    else:
        # Check default damaged sample first (check data/raw/archive, archive, etc.)
        candidates = [
            Path("data/raw/archive/training/Fresa_danada/IMG_20251103_091112.jpg"),
            Path("archive/training/Fresa_danada/IMG_20251103_091112.jpg"),
            Path(__file__).parent.parent / "data/raw/archive/training/Fresa_danada/IMG_20251103_091112.jpg"
        ]
        test_path = None
        for cand in candidates:
            if cand.exists():
                test_path = cand
                break

        if test_path is None:
            # Fall back to first discovered defective sample
            samples = find_dataset_samples(".")
            if samples["defective"]:
                test_path = samples["defective"][0]
            elif samples["healthy"]:
                test_path = samples["healthy"][0]
            else:
                print("[ERROR] No image path provided and no dataset samples found in workspace.")
                sys.exit(1)

    print("=" * 75)
    print(" QUICK INSPECTION TEST RUNNER")
    print("=" * 75)
    print(f" Target Image       : {test_path}")
    print(f" Confidence Thresh  : {args.conf}")
    print(f" Model Weights      : {args.model}")
    print(f" Output Directory   : {args.output_dir}")
    print("=" * 75)

    if not test_path.is_file():
        print(f"[ERROR] Specified image file not found: {test_path.resolve()}")
        sys.exit(1)

    # 1. Load YOLO Model
    model = load_yolo_model(model_name=args.model)

    # 2. Execute Inspection Pipeline on the target frame
    result = process_single_frame(
        image_path=test_path,
        model=model,
        conf=args.conf,
        save_plot=True,
        output_dir=args.output_dir
    )

    print("\n[SUCCESS] Inspection completed successfully.")
    print(f"Items detected      : {len(result['detections'])}")
    for d, h in zip(result['detections'], result['health_evaluations']):
        print(f"  * Item #{d['item_id']}: [{h['status']}] (Confidence: {d['confidence']*100:.1f}%)")
        print(f"    Rationale       : {h['reason']}")
        print(f"    Bounding Box    : {d['bbox']}")


if __name__ == "__main__":
    main()
