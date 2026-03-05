"""Visualization helpers for canonical pose keypoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from .keypoints_schema import CANONICAL_EDGES


def draw_skeleton(
    image: np.ndarray,
    keypoints: Mapping[str, Mapping[str, float | None]],
    *,
    min_confidence: float = 0.3,
    point_color: tuple[int, int, int] = (0, 220, 0),
    edge_color: tuple[int, int, int] = (255, 140, 0),
    radius: int = 4,
) -> np.ndarray:
    canvas = image.copy()

    def valid(name: str) -> tuple[int, int] | None:
        point = keypoints.get(name)
        if point is None:
            return None
        confidence = point.get("confidence", 0.0)
        x = point.get("x")
        y = point.get("y")
        if x is None or y is None or float(confidence) < min_confidence:
            return None
        return int(round(float(x))), int(round(float(y)))

    for start, end in CANONICAL_EDGES:
        p0 = valid(start)
        p1 = valid(end)
        if p0 is not None and p1 is not None:
            cv2.line(canvas, p0, p1, edge_color, 2)

    for name, _point in keypoints.items():
        coords = valid(name)
        if coords is not None:
            cv2.circle(canvas, coords, radius, point_color, -1)

    return canvas


def overlay_and_save(
    image_path: str | Path,
    keypoints: Mapping[str, Mapping[str, float | None]],
    output_path: str | Path,
    *,
    min_confidence: float = 0.3,
) -> Path:
    source = cv2.imread(str(image_path))
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rendered = draw_skeleton(source, keypoints, min_confidence=min_confidence)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rendered)
    return out_path
