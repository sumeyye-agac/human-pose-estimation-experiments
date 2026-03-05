# Limitations

This repository favors transparent and reproducible comparisons, but several constraints still apply.

## Current constraints

- Full parity across frameworks is hard because installation maturity differs by runtime.
- AlphaPose is not currently measurable on macOS arm64 CPU due CUDA-dependent custom ops.
- OpenPose benchmark path uses OpenCV DNN instead of `pyopenpose` native runtime.
- Shared export contract is frame-level and does not include multi-person identity tracking.
- Canonical schema is 2D COCO-17 subset only; 3D landmarks are out of scope.

## Planned next steps

- Add fixed real-video benchmark clips with explicit licensing.
- Add decode-included and decode-excluded benchmark modes.
- Add lightweight multi-person association fields in exports.
- Add optional 3D schema extension while preserving 2D compatibility.
