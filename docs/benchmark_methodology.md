# Benchmark Methodology

This project measures latency and FPS using `scripts/run_benchmarks.py`.

## Measurement protocol

- Warm-up runs execute first to avoid first-call overhead bias.
- Measured runs record per-frame latency with `time.perf_counter()`.
- Reported metrics include `avg_ms_per_frame`, `std_ms_per_frame`, and `fps`.
- Inference timing excludes video decode by default (`include_decode=false`).

## Input and runtime settings

- Default input is a deterministic synthetic RGB frame.
- Defaults: `warmup=20`, `frames=90`, `repeat=3`.
- Environment metadata is saved to `results/environment.json`.

## Reproducing results

```bash
PYTHONPATH=src python scripts/run_benchmarks.py --tool all
```

Generated artifacts:

- `results/benchmark.csv`
- `results/benchmark.md`
- `results/environment.json`
- `results/benchmark_raw_<tool>.json` for measured tools

## Notes on tool readiness

- Detectron2 can be measured on macOS arm64 via conda-forge `detectron2` CPU builds.
- OpenPose can be measured via OpenCV DNN using the official COCO Caffe model weights.
- AlphaPose currently requires CUDA-dependent custom ops in the official install path; on non-CUDA environments it is expected to remain `not_measured`.

## Exact environment used for current snapshot (March 6, 2026)

```bash
conda create -y -n posebench-d2 -c conda-forge python=3.10 detectron2
conda run -n posebench-d2 python -m pip install mediapipe==0.10.14 opencv-python-headless "setuptools<81"
conda run -n posebench-d2 python scripts/run_benchmarks.py --tool all --frames 30 --warmup 5 --repeat 1
```
