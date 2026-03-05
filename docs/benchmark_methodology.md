# Benchmark Methodology

This repository measures pose inference latency with `scripts/run_benchmarks.py` and writes artifacts under `results/`.

## Measurement protocol

- Warm-up runs execute first to reduce first-call overhead bias.
- Measured runs use `time.perf_counter()` per frame.
- Reported metrics include `avg_ms_per_frame`, `std_ms_per_frame`, and `fps`.
- Timing is inference-only by default (`include_decode=false`).

## Input and runtime defaults

- Input is a deterministic synthetic RGB frame.
- Default benchmark settings:
  - `warmup=20`
  - `frames=90`
  - `repeat=3`
- Environment metadata is captured in `results/environment.json`.

## Reproducing results

```bash
python scripts/run_benchmarks.py --tool all
```

Generated artifacts:

- `results/benchmark.csv`
- `results/benchmark.md`
- `results/environment.json`
- `results/benchmark_raw_<tool>.json` for measured tools

The README snapshot block is auto-updated from `results/benchmark.csv` by the same script.

## Framework-specific notes

- MediaPipe is typically the easiest measured path with `pip`.
- Detectron2 support depends on runtime and install channel. This repo uses best-effort adapter loading.
- OpenPose measurement uses the official COCO Caffe model executed via OpenCV DNN.
- AlphaPose official path requires CUDA custom ops; unsupported runtimes remain `not_measured`.
