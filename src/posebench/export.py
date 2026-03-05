"""Stable CSV/JSON export format for canonical keypoints."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .keypoints_schema import CANONICAL_KEYPOINTS


def canonical_csv_columns() -> list[str]:
    columns = ["frame_index", "timestamp_ms", "person_id", "tool", "schema"]
    for name in CANONICAL_KEYPOINTS:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_confidence"])
    return columns


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_to_row(frame: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": frame.get("frame_index"),
        "timestamp_ms": frame.get("timestamp_ms"),
        "person_id": frame.get("person_id", 0),
        "tool": frame.get("tool", "unknown"),
        "schema": frame.get("schema", "coco17"),
    }

    keypoints = frame.get("keypoints", {})
    for name in CANONICAL_KEYPOINTS:
        point = keypoints.get(name, {}) if isinstance(keypoints, Mapping) else {}
        row[f"{name}_x"] = _as_float(point.get("x"))
        row[f"{name}_y"] = _as_float(point.get("y"))
        row[f"{name}_confidence"] = _as_float(point.get("confidence", point.get("score")))

    return row


def export_frames_to_csv(frames: Iterable[Mapping[str, object]], csv_path: str | Path) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = canonical_csv_columns()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for frame in frames:
            writer.writerow(frame_to_row(frame))

    return path


def export_frames_to_json(frames: Iterable[Mapping[str, object]], json_path: str | Path) -> Path:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = list(frames)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return path
