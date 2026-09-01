# Limitations

This repository favors transparent and reproducible comparisons, but several constraints still apply.

## Current constraints

- Full parity across frameworks is hard because installation maturity differs by runtime.
- AlphaPose is not currently measurable on macOS arm64 CPU due CUDA-dependent custom ops,
  so its demo export stays a labeled synthetic sample rather than model output.
- OpenPose benchmark and demo paths use OpenCV DNN instead of `pyopenpose` native runtime,
  so their latency is not comparable to published OpenPose numbers.
- OpenPose keypoints are decoded with a per-part argmax and no PAF grouping, which is valid
  only for single-person frames such as the demo image.
- Confidence values are not comparable across tools: BlazePose visibility, Detectron2 heatmap
  scores, and OpenPose heatmap peak probabilities use different scales.
- Shared export contract is frame-level and does not include multi-person identity tracking.
- Canonical schema is 2D COCO-17 subset only; 3D landmarks are out of scope.

## Planned next steps

- Add fixed real-video benchmark clips with explicit licensing. Single-frame demos already
  run on a licensed real photo (`assets/pose_demo_full_body.jpg`, see `assets/SOURCES.md`),
  but the latency benchmark still uses a synthetic frame.
- Add decode-included and decode-excluded benchmark modes.
- Add lightweight multi-person association fields in exports.
- Add optional 3D schema extension while preserving 2D compatibility.
