from pathlib import Path

from posebench.export import canonical_csv_columns, export_frames_to_csv
from posebench.features import joint_angle_degrees, summarize_series_features


def test_export_writes_canonical_columns(tmp_path: Path) -> None:
    frames = [
        {
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "person_id": 0,
            "tool": "mediapipe",
            "schema": "coco17",
            "keypoints": {
                "nose": {"x": 10.0, "y": 20.0, "confidence": 0.9},
            },
        }
    ]

    out_path = export_frames_to_csv(frames, tmp_path / "sample.csv")
    text = out_path.read_text(encoding="utf-8")

    for column in canonical_csv_columns()[:8]:
        assert column in text


def test_angle_and_features_are_computable() -> None:
    angle = joint_angle_degrees((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    features = summarize_series_features([10.0, 12.0, 15.0, 12.0, 11.0])

    assert round(angle, 3) == 90.0
    assert features["peak_count"] >= 1
