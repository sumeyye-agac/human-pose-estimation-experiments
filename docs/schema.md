# Canonical Schema

This project maps all framework outputs to a canonical **COCO-17 subset** so that exports and features are comparable across tools.

The mapping logic is implemented in `src/posebench/keypoints_schema.py`.

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

## Skeleton topology

Canonical edges are stored in `CANONICAL_EDGES` and used by `posebench.viz` for overlay rendering.

Representative links:

- upper body: shoulders, elbows, wrists
- lower body: hips, knees, ankles
- torso connectors: shoulder-hip and hip-hip links
- face anchors: nose-eye-ear links

## Framework mapping coverage

- MediaPipe BlazePose (33 landmarks) to COCO-17 subset via index mapping.
- OpenPose BODY_25 to COCO-17 subset via index mapping.
- AlphaPose COCO-17 to canonical one-to-one mapping.
- Detectron2 COCO keypoints to canonical one-to-one mapping.

## CSV contract

`posebench.export.canonical_csv_columns()` defines a stable export contract:

- metadata columns
  - `frame_index`
  - `timestamp_ms`
  - `person_id`
  - `tool`
  - `schema`
- per-keypoint columns
  - `{name}_x`
  - `{name}_y`
  - `{name}_confidence`

Every row is frame-level and uses the canonical schema, regardless of source framework.
