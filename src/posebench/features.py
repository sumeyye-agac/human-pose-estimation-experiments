"""Feature extraction helpers built on canonical keypoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, savgol_filter
except Exception:  # pragma: no cover - scipy optional fallback
    find_peaks = None
    savgol_filter = None


def joint_angle_degrees(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
    point_c: tuple[float, float] | None,
) -> float:
    """Compute angle ABC in degrees, returning NaN for invalid geometry."""

    if point_a is None or point_b is None or point_c is None:
        return float("nan")

    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)

    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product == 0:
        return float("nan")

    cosine = np.clip(np.dot(ba, bc) / norm_product, -1.0, 1.0)
    return float(math.degrees(math.acos(cosine)))


def _point_from_frame(
    frame: Mapping[str, object],
    name: str,
    min_confidence: float,
) -> tuple[float, float] | None:
    keypoints = frame.get("keypoints", {})
    point = keypoints.get(name) if isinstance(keypoints, Mapping) else None
    if not isinstance(point, Mapping):
        return None

    confidence = point.get("confidence", point.get("score", 0.0))
    if confidence is None or float(confidence) < min_confidence:
        return None

    x = point.get("x")
    y = point.get("y")
    if x is None or y is None:
        return None

    return float(x), float(y)


def extract_joint_angles(
    frames: Sequence[Mapping[str, object]],
    joint_triplets: Mapping[str, tuple[str, str, str]],
    *,
    min_confidence: float = 0.3,
) -> pd.DataFrame:
    """Extract named joint angles for each frame as a tidy DataFrame."""

    records: list[dict[str, float | int]] = []
    for frame in frames:
        record: dict[str, float | int] = {
            "frame_index": int(frame.get("frame_index", len(records))),
            "timestamp_ms": float(frame.get("timestamp_ms", 0.0)),
        }
        for feature_name, (a, b, c) in joint_triplets.items():
            point_a = _point_from_frame(frame, a, min_confidence)
            point_b = _point_from_frame(frame, b, min_confidence)
            point_c = _point_from_frame(frame, c, min_confidence)
            record[feature_name] = joint_angle_degrees(point_a, point_b, point_c)
        records.append(record)

    return pd.DataFrame.from_records(records)


def smooth_series(values: Sequence[float], *, method: str = "ema", alpha: float = 0.2) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    if method == "savgol" and savgol_filter is not None and arr.size >= 7:
        window = 7 if arr.size >= 7 else arr.size - 1
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            return savgol_filter(arr, window_length=window, polyorder=2, mode="interp")

    smoothed = arr.copy()
    for idx in range(1, len(smoothed)):
        if np.isnan(smoothed[idx]):
            smoothed[idx] = smoothed[idx - 1]
        elif np.isnan(smoothed[idx - 1]):
            continue
        else:
            smoothed[idx] = alpha * smoothed[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def compute_angular_velocity(values: Sequence[float], *, fps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    dt = 1.0 / fps
    return np.gradient(arr, dt)


def summarize_series_features(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "peak_count": 0,
        }

    if find_peaks is not None:
        peaks, _ = find_peaks(arr)
        peak_count = int(len(peaks))
    else:
        peak_count = int(np.sum((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))) if arr.size > 2 else 0

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "peak_count": peak_count,
    }
