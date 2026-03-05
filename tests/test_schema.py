from posebench.keypoints_schema import CANONICAL_KEYPOINTS, map_tool_keypoints_to_canonical


def test_mediapipe_mapping_assigns_expected_points() -> None:
    points = [{"x": float(i), "y": float(i + 1), "confidence": 0.99} for i in range(33)]
    mapped = map_tool_keypoints_to_canonical("mediapipe", points)

    assert set(mapped.keys()) == set(CANONICAL_KEYPOINTS)
    assert mapped["left_shoulder"]["x"] == 11.0
    assert mapped["right_ankle"]["x"] == 28.0


def test_openpose_mapping_handles_missing_confidence() -> None:
    points = [(0.0, 1.0)] * 25
    mapped = map_tool_keypoints_to_canonical("openpose", points)

    assert mapped["nose"]["x"] == 0.0
    assert mapped["nose"]["confidence"] == 0.0
