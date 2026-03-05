# Human Pose Estimation Experiments

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Colab-Open%20Notebooks-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![CI](https://img.shields.io/badge/CI-ruff%20%2B%20pytest-1f6feb)](./.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

This repository is a practical benchmark and analysis workspace for **MediaPipe**, **OpenPose**, **AlphaPose**, and **Detectron2**.
It focuses on one clean pipeline: run inference, map all outputs to one canonical schema, export stable artifacts, and analyze motion signals.

![Pipeline overview](./assets/pipeline_overview.svg)

## Why this repo

Human pose estimation tool outputs are not directly comparable without shared schema and reproducible metrics.
This repo provides a small but production-minded scaffold for:

- Cross-tool canonical keypoint mapping (COCO-17 subset)
- Stable CSV/JSON frame exports
- Angle, angular velocity, and smoothing features
- Reproducible benchmark artifacts and environment capture

## Repository layout

```text
.
├── MediaPipe/
├── OpenPose/
├── AlphaPose/
├── Detectron2/
├── notebooks/
├── src/posebench/
├── scripts/
├── results/
├── assets/
├── docs/
└── tests/
```

## Notebook index

| Notebook | Scope | Open in Colab |
| --- | --- | --- |
| `MediaPipe/01_mediapipe_pose_demo.ipynb` | MediaPipe inference, canonical mapping, export, mini benchmark | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/MediaPipe/01_mediapipe_pose_demo.ipynb) |
| `MediaPipe/02_mediapipe_export_and_features.ipynb` | Frame sequence export, angles, smoothing, velocity | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/MediaPipe/02_mediapipe_export_and_features.ipynb) |
| `Detectron2/01_detectron2_keypoints_demo.ipynb` | Detectron2 keypoint demo and export | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/Detectron2/01_detectron2_keypoints_demo.ipynb) |
| `OpenPose/01_openpose_install_and_run.ipynb` | OpenPose recommended/fallback setup, export path | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/OpenPose/01_openpose_install_and_run.ipynb) |
| `AlphaPose/01_alphapose_colab_inference.ipynb` | AlphaPose recommended/fallback setup, export path | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/AlphaPose/01_alphapose_colab_inference.ipynb) |
| `notebooks/01_benchmark_all_tools.ipynb` | Generate and inspect benchmark artifacts | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/notebooks/01_benchmark_all_tools.ipynb) |
| `notebooks/02_keypoints_timeseries_analysis.ipynb` | Canonical CSV time-series analysis and feature extraction | [Open](https://colab.research.google.com/github/sumeyye-agac/human-pose-estimation-experiments/blob/main/notebooks/02_keypoints_timeseries_analysis.ipynb) |

## Results snapshot

Current snapshot is generated from `results/benchmark.csv` and rendered in [`results/benchmark.md`](./results/benchmark.md).
Numbers are shown only when a tool is actually measured in the current runtime.

<!-- RESULTS_SNAPSHOT_START -->
| Tool | Status | Avg ms/frame | FPS |
| --- | --- | --- | --- |
| mediapipe | measured | 7.70 | 129.88 |
| detectron2 | measured | 1018.91 | 0.98 |
| openpose | measured | 429.43 | 2.33 |
| alphapose | not_measured | - | - |
<!-- RESULTS_SNAPSHOT_END -->

Notes for this snapshot:

- `openpose` is measured with the official COCO OpenPose Caffe model executed through OpenCV DNN.
- `alphapose` remains `not_measured` on this machine because the official build path requires CUDA custom ops.

## Quick start

### Colab

- Open any notebook from the table above.
- The first setup cells clone the repo when needed and install runtime dependencies idempotently.

### Local

```bash
git clone https://github.com/sumeyye-agac/human-pose-estimation-experiments.git
cd human-pose-estimation-experiments
python -m pip install -r requirements.txt
```

## Output format

All tools are mapped to a shared canonical schema.
The schema and CSV contract are documented in [`docs/schema.md`](./docs/schema.md).

## Reproducibility

Benchmark artifacts are generated through one command:

```bash
python scripts/run_benchmarks.py --tool all
```

Generated files:

- `results/benchmark.csv`
- `results/benchmark.md`
- `results/environment.json`
- `results/benchmark_raw_<tool>.json` for measured tools

Method details are in [`docs/benchmark_methodology.md`](./docs/benchmark_methodology.md).

For the current measured snapshot on macOS arm64, the command was executed in a `conda` environment with `python=3.10`, `detectron2=0.6`, `mediapipe=0.10.14`, and CPU inference.

## OpenPose and AlphaPose setup reality

OpenPose and AlphaPose can break on clean Colab or local sessions due build/CUDA constraints.
Their notebooks include:

- A recommended path for ready environments
- A fallback flow that keeps export and schema validation runnable

## Limitations and next steps

- Run complete cross-tool benchmarks on one fixed Colab runtime and publish measured rows.
- Add multi-person tracking and identity consistency across frames.
- Add standardized sample clips and per-tool decode policy toggles.
- Add downstream gesture/posture classifiers on top of current feature exports.
- Add optional 3D keypoint schema support.

## References

- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [OpenPose (official)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [AlphaPose (official)](https://github.com/MVIG-SJTU/AlphaPose)
- [Detectron2 (official)](https://github.com/facebookresearch/detectron2)

## Author

- [GitHub](https://github.com/sumeyye-agac)
- [LinkedIn](https://www.linkedin.com/in/sumeyye-agac)
