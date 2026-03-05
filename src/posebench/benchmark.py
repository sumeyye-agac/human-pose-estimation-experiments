"""Benchmarking helpers for reproducible pose inference measurements."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class InferenceBackend(Protocol):
    name: str

    def infer(self, frame: np.ndarray) -> Any:
        ...


@dataclasses.dataclass(slots=True)
class BenchmarkConfig:
    warmup_frames: int = 20
    measured_frames: int = 120
    repeat: int = 3
    include_decode: bool = False


def benchmark_backend(
    backend: InferenceBackend,
    frames: Iterable[np.ndarray],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Benchmark requires at least one frame")

    for idx in range(config.warmup_frames):
        backend.infer(frame_list[idx % len(frame_list)])

    run_means: list[float] = []
    all_measurements_ms: list[float] = []
    for _ in range(config.repeat):
        measurements_ms: list[float] = []
        for idx in range(config.measured_frames):
            frame = frame_list[idx % len(frame_list)]
            started = time.perf_counter()
            backend.infer(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            measurements_ms.append(elapsed_ms)
            all_measurements_ms.append(elapsed_ms)

        run_means.append(float(np.mean(measurements_ms)))

    mean_ms = float(np.mean(all_measurements_ms))
    std_ms = float(np.std(all_measurements_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "tool": backend.name,
        "status": "measured",
        "avg_ms_per_frame": mean_ms,
        "std_ms_per_frame": std_ms,
        "fps": fps,
        "measured_frames": config.measured_frames,
        "warmup_frames": config.warmup_frames,
        "repeat": config.repeat,
        "include_decode": config.include_decode,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_gpu_name() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    first_line = output.strip().splitlines()
    return first_line[0].strip() if first_line else None


def collect_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu_name": _probe_gpu_name(),
        "library_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "opencv-python": _package_version("opencv-python"),
            "mediapipe": _package_version("mediapipe"),
            "torch": _package_version("torch"),
            "detectron2": _package_version("detectron2"),
        },
    }


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return out_path
