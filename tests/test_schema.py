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


def test_openpose_coco_mapping_uses_18_point_ordering() -> None:
    # OpenPose COCO model output order: index 8 is right_hip, 11 is left_hip,
    # 14/15 are the eyes and 16/17 the ears.
    points = [{"x": float(i), "y": float(i + 1), "confidence": 0.9} for i in range(18)]
    mapped = map_tool_keypoints_to_canonical("openpose_coco", points)

    assert mapped["right_hip"]["x"] == 8.0
    assert mapped["left_hip"]["x"] == 11.0
    assert mapped["right_ankle"]["x"] == 10.0
    assert mapped["left_ankle"]["x"] == 13.0
    assert mapped["left_eye"]["x"] == 15.0
    assert mapped["right_ear"]["x"] == 16.0


def test_openpose_body25_and_coco_maps_are_distinct() -> None:
    points = [{"x": float(i), "y": 0.0, "confidence": 0.9} for i in range(25)]
    body25 = map_tool_keypoints_to_canonical("openpose", points)
    coco18 = map_tool_keypoints_to_canonical("openpose_coco", points)

    assert body25["left_hip"]["x"] != coco18["left_hip"]["x"]
