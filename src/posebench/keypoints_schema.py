"""Canonical keypoint schema and cross-framework mapping utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence

CANONICAL_SCHEMA_NAME = "coco17"
CANONICAL_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
CANONICAL_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

_TOOL_ALIASES = {
    "mp": "mediapipe",
    "blazepose": "mediapipe",
    "body_25": "openpose",
    "body25": "openpose",
    "coco": "alphapose",
    "mmpose": "alphapose",
    "d2": "detectron2",
}

SUPPORTED_TOOLS = ("mediapipe", "openpose", "alphapose", "detectron2")

# Canonical index -> tool index
_TOOL_INDEX_MAPS: dict[str, dict[int, int]] = {
    "mediapipe": {
        0: 0,
        1: 2,
        2: 5,
        3: 7,
        4: 8,
        5: 11,
        6: 12,
        7: 13,
        8: 14,
        9: 15,
        10: 16,
        11: 23,
        12: 24,
        13: 25,
        14: 26,
        15: 27,
        16: 28,
    },
    "openpose": {
        0: 0,
        1: 16,
        2: 15,
        3: 18,
        4: 17,
        5: 5,
        6: 2,
        7: 6,
        8: 3,
        9: 7,
        10: 4,
        11: 12,
        12: 9,
        13: 13,
        14: 10,
        15: 14,
        16: 11,
    },
    "alphapose": {idx: idx for idx in range(17)},
    "detectron2": {idx: idx for idx in range(17)},
}


def normalize_tool_name(tool_name: str) -> str:
    name = tool_name.strip().lower()
    return _TOOL_ALIASES.get(name, name)


def _xyc_from_point(point: object) -> tuple[float | None, float | None, float | None]:
    if isinstance(point, Mapping):
        x = point.get("x")
        y = point.get("y")
        c = point.get("confidence", point.get("score", point.get("visibility")))
        return _to_float(x), _to_float(y), _to_float(c)

    if isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
        x = _to_float(point[0]) if len(point) > 0 else None
        y = _to_float(point[1]) if len(point) > 1 else None
        c = _to_float(point[2]) if len(point) > 2 else None
        return x, y, c

    return None, None, None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_lookup(tool_keypoints: Sequence[object] | Mapping[object, object]) -> MutableMapping[int, object]:
    lookup: MutableMapping[int, object] = {}
    if isinstance(tool_keypoints, Mapping):
        for key, value in tool_keypoints.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            lookup[idx] = value
        return lookup

    for idx, value in enumerate(tool_keypoints):
        lookup[idx] = value
    return lookup


def empty_canonical_pose(default_confidence: float = 0.0) -> dict[str, dict[str, float | None]]:
    return {
        name: {"x": None, "y": None, "confidence": default_confidence}
        for name in CANONICAL_KEYPOINTS
    }


def map_tool_keypoints_to_canonical(
    tool_name: str,
    tool_keypoints: Sequence[object] | Mapping[object, object],
    *,
    min_confidence: float = 0.0,
) -> dict[str, dict[str, float | None]]:
    """Map framework-specific keypoints into the canonical COCO-17 subset."""

    normalized_tool = normalize_tool_name(tool_name)
    if normalized_tool not in _TOOL_INDEX_MAPS:
        raise ValueError(f"Unsupported tool '{tool_name}'. Expected one of: {SUPPORTED_TOOLS}")

    lookup = _build_lookup(tool_keypoints)
    mapped = empty_canonical_pose(default_confidence=0.0)
    index_map = _TOOL_INDEX_MAPS[normalized_tool]

    for canonical_idx, canonical_name in enumerate(CANONICAL_KEYPOINTS):
        tool_idx = index_map.get(canonical_idx)
        point = lookup.get(tool_idx)
        x, y, confidence = _xyc_from_point(point)
        confidence = 0.0 if confidence is None else confidence

        if confidence < min_confidence or x is None or y is None:
            mapped[canonical_name] = {"x": None, "y": None, "confidence": confidence}
            continue

        mapped[canonical_name] = {"x": x, "y": y, "confidence": confidence}

    return mapped


def list_schema_rows() -> Iterable[tuple[int, str]]:
    yield from enumerate(CANONICAL_KEYPOINTS)
