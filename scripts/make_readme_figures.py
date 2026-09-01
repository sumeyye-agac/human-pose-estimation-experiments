#!/usr/bin/env python3
# ruff: noqa: E402
"""Build the downscaled figures that README.md embeds.

Notebook runs write full-resolution artifacts into `assets/generated/`, which stays
untracked. This script derives small, committable copies in `assets/readme/` so the
README shows real output without carrying multi-megabyte files in git history.

Overlays are cropped to the detected subject (the demo photo is mostly sky) using the
MediaPipe demo export as the reference bounding box, so every tool is cropped
identically and the three overlays stay directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from posebench.keypoints_schema import CANONICAL_KEYPOINTS

GENERATED_DIR = REPO_ROOT / "assets" / "generated"
README_DIR = REPO_ROOT / "assets" / "readme"
REFERENCE_CSV = REPO_ROOT / "results" / "mediapipe_demo_keypoints.csv"

# Overlays are cropped to the subject; charts are only downscaled.
OVERLAYS = [
    ("mediapipe_demo_overlay.jpg", "mediapipe_overlay.jpg"),
    ("detectron2_demo_overlay.jpg", "detectron2_overlay.jpg"),
    ("openpose_demo_overlay.jpg", "openpose_overlay.jpg"),
]
CHARTS = [
    ("benchmark_fps.png", "benchmark_fps.png"),
    ("quality_metrics_overview.png", "quality_metrics_overview.png"),
]

OVERLAY_WIDTH = 420
CHART_WIDTH = 900
CROP_MARGIN = 0.18
CROP_ASPECT = 0.62  # width / height


def load_reference_bbox() -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) of the resolved MediaPipe keypoints."""
    with REFERENCE_CSV.open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    xs = [float(row[f"{name}_x"]) for name in CANONICAL_KEYPOINTS if row[f"{name}_x"]]
    ys = [float(row[f"{name}_y"]) for name in CANONICAL_KEYPOINTS if row[f"{name}_y"]]
    if not xs or not ys:
        raise SystemExit(f"No resolved keypoints in {REFERENCE_CSV}")
    return min(xs), min(ys), max(xs), max(ys)


def crop_box(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Expand the keypoint bbox by a margin, then fit it to the target aspect ratio."""
    x_min, y_min, x_max, y_max = load_reference_bbox()
    margin_x = (x_max - x_min) * CROP_MARGIN
    margin_y = (y_max - y_min) * CROP_MARGIN
    x_min, x_max = x_min - margin_x, x_max + margin_x
    y_min, y_max = y_min - margin_y, y_max + margin_y

    height = y_max - y_min
    width = max(x_max - x_min, height * CROP_ASPECT)
    center_x = (x_min + x_max) / 2

    left = int(max(0, round(center_x - width / 2)))
    right = int(min(image_width, round(center_x + width / 2)))
    top = int(max(0, round(y_min)))
    bottom = int(min(image_height, round(y_max)))
    return left, top, right, bottom


def resize_to_width(image, target_width: int):
    height, width = image.shape[:2]
    if width <= target_width:
        return image
    target_height = round(height * target_width / width)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def write_overlay(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Missing overlay: {source}. Run the per-tool notebooks first.")

    height, width = image.shape[:2]
    left, top, right, bottom = crop_box(width, height)
    cropped = resize_to_width(image[top:bottom, left:right], OVERLAY_WIDTH)
    cv2.imwrite(str(destination), cropped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Wrote {destination.relative_to(REPO_ROOT)} ({cropped.shape[1]}x{cropped.shape[0]})")


def write_chart(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Missing chart: {source}. Run the analysis notebooks first.")

    height, width = image.shape[:2]
    if width <= CHART_WIDTH:
        # Already small enough; copy the bytes instead of re-encoding the PNG,
        # which would only inflate it.
        destination.write_bytes(source.read_bytes())
        print(f"Copied {destination.relative_to(REPO_ROOT)} ({width}x{height})")
        return

    resized = resize_to_width(image, CHART_WIDTH)
    cv2.imwrite(str(destination), resized)
    print(f"Wrote {destination.relative_to(REPO_ROOT)} ({resized.shape[1]}x{resized.shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate README figures.")
    parser.parse_args()

    README_DIR.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in OVERLAYS:
        write_overlay(GENERATED_DIR / source_name, README_DIR / destination_name)
    for source_name, destination_name in CHARTS:
        write_chart(GENERATED_DIR / source_name, README_DIR / destination_name)


if __name__ == "__main__":
    main()
