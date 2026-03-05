import posebench


def test_smoke_import() -> None:
    assert hasattr(posebench, "map_tool_keypoints_to_canonical")
