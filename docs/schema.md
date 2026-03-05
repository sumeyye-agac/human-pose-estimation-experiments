# Canonical Keypoint Schema

This repository standardizes all framework outputs to the **COCO-17** canonical schema. The mapping lives in `src/posebench/keypoints_schema.py`.

## Canonical keypoint order

| Index | Name |
| --- | --- |
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

## CSV export format

Frame-level exports are produced by `posebench.export` with stable columns:

- `frame_index`
- `timestamp_ms`
- `person_id`
- `tool`
- `schema`
- For each keypoint, three columns: `{name}_x`, `{name}_y`, `{name}_confidence`

## Mapping coverage

- MediaPipe BlazePose (33 landmarks) -> canonical subset via index mapping.
- OpenPose BODY_25 -> canonical subset via index mapping.
- AlphaPose COCO-17 -> one-to-one mapping.
- Detectron2 COCO keypoints -> one-to-one mapping.

