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

OpenPose, AlphaPose, and Detectron2 often require model weights or custom build steps in Colab. The benchmark runner marks these as `not_measured` until their notebook setup path is completed.
