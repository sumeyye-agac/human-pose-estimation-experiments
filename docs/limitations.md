# Current Limitations

- Cross-tool benchmarks are not directly comparable unless all tools run on the same input pipeline and runtime.
- Multi-person tracking and identity association are not implemented in the shared exporter yet.
- 3D pose outputs are not included in the canonical schema.
- AlphaPose official setup requires CUDA custom ops and is not currently runnable on macOS arm64 CPU-only environments.
