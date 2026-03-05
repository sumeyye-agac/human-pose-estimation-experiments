#!/usr/bin/env python3
# ruff: noqa: E402
"""Run benchmark experiments and generate reproducible artifacts in results/."""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from posebench.benchmark import BenchmarkConfig, benchmark_backend, collect_environment, write_json

TOOLS = ("mediapipe", "detectron2", "openpose", "alphapose")
CSV_COLUMNS = [
    "tool",
    "status",
    "avg_ms_per_frame",
    "std_ms_per_frame",
    "fps",
    "measured_frames",
    "warmup_frames",
    "repeat",
    "include_decode",
    "notes",
    "raw_json",
]
README_SNAPSHOT_START = "<!-- BENCHMARK_SNAPSHOT_START -->"
README_SNAPSHOT_END = "<!-- BENCHMARK_SNAPSHOT_END -->"
MODEL_CACHE_DIR = REPO_ROOT / ".cache" / "models"


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self) -> None:
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, frame: np.ndarray) -> Any:
        return self._pose.process(frame)


class Detectron2Backend:
    name = "detectron2"

    def __init__(self) -> None:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        from detectron2 import model_zoo

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self._predictor = DefaultPredictor(cfg)

    def infer(self, frame: np.ndarray) -> Any:
        return self._predictor(frame)


class OpenPoseBackend:
    name = "openpose"

    def __init__(self, prototxt_path: Path, caffemodel_path: Path) -> None:
        import cv2

        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
        self._input_size = (368, 368)

    def infer(self, frame: np.ndarray) -> Any:
        blob = self._cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=self._input_size,
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        return self._net.forward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pose benchmark experiments.")
    parser.add_argument("--tool", choices=[*TOOLS, "all"], default="all")
    parser.add_argument("--frames", type=int, default=90, help="Measured frames per repeat")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up frames")
    parser.add_argument("--repeat", type=int, default=3, help="Benchmark repeats")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def _synthetic_frame(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return frame


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as file_obj:
        file_obj.write(response.read())


def _ensure_openpose_model_files() -> tuple[Path, Path]:
    prototxt = MODEL_CACHE_DIR / "openpose" / "pose_deploy_linevec.prototxt"
    caffemodel = MODEL_CACHE_DIR / "openpose" / "pose_iter_440000.caffemodel"

    sources = {
        prototxt: [
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
            "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_deploy_linevec.prototxt",
        ],
        caffemodel: [
            "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
        ],
    }

    for target_path, urls in sources.items():
        if target_path.exists():
            continue

        last_error: Exception | None = None
        for url in urls:
            try:
                _download_file(url, target_path)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not download {target_path.name}: {last_error}")

    return prototxt, caffemodel


def _run_mediapipe(config: BenchmarkConfig, frame: np.ndarray, output_dir: Path) -> dict[str, Any]:
    try:
        backend = MediaPipeBackend()
    except Exception as exc:
        return {
            "tool": "mediapipe",
            "status": "not_measured",
            "notes": f"mediapipe import/setup failed: {exc}",
        }

    result = benchmark_backend(backend=backend, frames=[frame], config=config)
    raw_path = output_dir / "benchmark_raw_mediapipe.json"
    write_json(result, raw_path)
    result["raw_json"] = str(raw_path)
    result["notes"] = "Synthetic single-frame benchmark. Inference only."
    return result


def _run_detectron2(config: BenchmarkConfig, frame: np.ndarray, output_dir: Path) -> dict[str, Any]:
    try:
        backend = Detectron2Backend()
    except Exception as exc:
        return {
            "tool": "detectron2",
            "status": "not_measured",
            "notes": f"detectron2 import/setup failed: {exc}",
        }

    result = benchmark_backend(backend=backend, frames=[frame], config=config)
    raw_path = output_dir / "benchmark_raw_detectron2.json"
    write_json(result, raw_path)
    result["raw_json"] = str(raw_path)
    result["notes"] = "Synthetic single-frame benchmark. Inference only."
    return result


def _run_openpose(config: BenchmarkConfig, frame: np.ndarray, output_dir: Path) -> dict[str, Any]:
    try:
        prototxt_path, caffemodel_path = _ensure_openpose_model_files()
        backend = OpenPoseBackend(prototxt_path=prototxt_path, caffemodel_path=caffemodel_path)
    except Exception as exc:
        return {
            "tool": "openpose",
            "status": "not_measured",
            "notes": f"openpose setup failed: {exc}",
        }

    result = benchmark_backend(backend=backend, frames=[frame], config=config)
    raw_path = output_dir / "benchmark_raw_openpose.json"
    write_json(result, raw_path)
    result["raw_json"] = str(raw_path)
    result["notes"] = (
        "Synthetic single-frame benchmark. Inference only. "
        "OpenPose COCO model executed via OpenCV DNN."
    )
    return result


def _not_measured(tool: str, reason: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": "not_measured",
        "notes": reason,
    }


def _generate_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Benchmark Results",
        "",
        "Preliminary artifact generated by `scripts/run_benchmarks.py`.",
        "Only rows with `status=measured` contain numeric latency/FPS values.",
        "",
        "| Tool | Status | Avg ms/frame | Std ms/frame | FPS | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        avg_ms = row.get("avg_ms_per_frame")
        std_ms = row.get("std_ms_per_frame")
        fps = row.get("fps")
        avg_ms_str = f"{avg_ms:.2f}" if isinstance(avg_ms, (float, int)) else "-"
        std_ms_str = f"{std_ms:.2f}" if isinstance(std_ms, (float, int)) else "-"
        fps_str = f"{fps:.2f}" if isinstance(fps, (float, int)) else "-"
        notes = row.get("notes", "")
        lines.append(
            f"| {row['tool']} | {row['status']} | {avg_ms_str} | {std_ms_str} | {fps_str} | {notes} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Tool | Status | Avg ms/frame | Std ms/frame | FPS |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        avg_ms = row.get("avg_ms_per_frame")
        std_ms = row.get("std_ms_per_frame")
        fps = row.get("fps")
        avg_ms_str = f"{avg_ms:.2f}" if isinstance(avg_ms, (float, int)) else "-"
        std_ms_str = f"{std_ms:.2f}" if isinstance(std_ms, (float, int)) else "-"
        fps_str = f"{fps:.2f}" if isinstance(fps, (float, int)) else "-"
        lines.append(f"| {row['tool']} | {row['status']} | {avg_ms_str} | {std_ms_str} | {fps_str} |")
    return lines


def _update_readme_snapshot(rows: list[dict[str, Any]], readme_path: Path) -> None:
    if not readme_path.exists():
        return

    text = readme_path.read_text(encoding="utf-8")
    if README_SNAPSHOT_START not in text or README_SNAPSHOT_END not in text:
        return

    start_idx = text.index(README_SNAPSHOT_START) + len(README_SNAPSHOT_START)
    end_idx = text.index(README_SNAPSHOT_END)
    snapshot = "\n" + "\n".join(_snapshot_lines(rows)) + "\n"
    updated = text[:start_idx] + snapshot + text[end_idx:]
    readme_path.write_text(updated, encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], out_path: Path, config: BenchmarkConfig) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            enriched = {
                "tool": row.get("tool"),
                "status": row.get("status", "not_measured"),
                "avg_ms_per_frame": row.get("avg_ms_per_frame"),
                "std_ms_per_frame": row.get("std_ms_per_frame"),
                "fps": row.get("fps"),
                "measured_frames": row.get("measured_frames", config.measured_frames),
                "warmup_frames": row.get("warmup_frames", config.warmup_frames),
                "repeat": row.get("repeat", config.repeat),
                "include_decode": row.get("include_decode", config.include_decode),
                "notes": row.get("notes", ""),
                "raw_json": row.get("raw_json", ""),
            }
            writer.writerow(enriched)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        warmup_frames=args.warmup,
        measured_frames=args.frames,
        repeat=args.repeat,
        include_decode=False,
    )

    frame = _synthetic_frame(width=args.width, height=args.height, seed=args.seed)

    selected_tools = TOOLS if args.tool == "all" else (args.tool,)
    rows: list[dict[str, Any]] = []

    for tool in selected_tools:
        if tool == "mediapipe":
            rows.append(_run_mediapipe(config=config, frame=frame, output_dir=output_dir))
        elif tool == "detectron2":
            rows.append(_run_detectron2(config=config, frame=frame, output_dir=output_dir))
        elif tool == "openpose":
            rows.append(_run_openpose(config=config, frame=frame, output_dir=output_dir))
        elif tool == "alphapose":
            rows.append(
                _not_measured(
                    tool="alphapose",
                    reason=(
                        "AlphaPose official install requires CUDA custom ops (CUDA_HOME). "
                        "This macOS arm64 CPU environment is unsupported."
                    ),
                )
            )

    if args.tool == "all":
        present = {row["tool"] for row in rows}
        for tool in TOOLS:
            if tool not in present:
                rows.append(_not_measured(tool, "No benchmark adapter configured."))

    environment = collect_environment()
    environment["benchmark_config"] = asdict(config)
    environment["input"] = {
        "kind": "synthetic_random_frame",
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
    }

    _write_csv(rows=rows, out_path=output_dir / "benchmark.csv", config=config)
    _generate_markdown(rows=rows, out_path=output_dir / "benchmark.md")
    write_json(environment, output_dir / "environment.json")
    _update_readme_snapshot(rows=rows, readme_path=REPO_ROOT / "README.md")

    measured = [row for row in rows if row.get("status") == "measured"]
    print(f"Wrote {output_dir / 'benchmark.csv'}")
    print(f"Wrote {output_dir / 'benchmark.md'}")
    print(f"Wrote {output_dir / 'environment.json'}")
    print(f"Measured tools: {', '.join(row['tool'] for row in measured) if measured else 'none'}")


if __name__ == "__main__":
    main()
