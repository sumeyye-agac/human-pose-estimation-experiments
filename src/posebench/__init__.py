"""Shared utilities for cross-framework human pose experiments."""

from .benchmark import BenchmarkConfig, benchmark_backend, collect_environment
from .export import canonical_csv_columns, export_frames_to_csv, export_frames_to_json
from .features import (
    compute_angular_velocity,
    extract_joint_angles,
    joint_angle_degrees,
    smooth_series,
    summarize_series_features,
)
from .keypoints_schema import (
    CANONICAL_EDGES,
    CANONICAL_KEYPOINTS,
    CANONICAL_SCHEMA_NAME,
    SUPPORTED_TOOLS,
    map_tool_keypoints_to_canonical,
)

__all__ = [
    "BenchmarkConfig",
    "CANONICAL_EDGES",
    "CANONICAL_KEYPOINTS",
    "CANONICAL_SCHEMA_NAME",
    "SUPPORTED_TOOLS",
    "benchmark_backend",
    "canonical_csv_columns",
    "collect_environment",
    "compute_angular_velocity",
    "export_frames_to_csv",
    "export_frames_to_json",
    "extract_joint_angles",
    "joint_angle_degrees",
    "map_tool_keypoints_to_canonical",
    "smooth_series",
    "summarize_series_features",
]
