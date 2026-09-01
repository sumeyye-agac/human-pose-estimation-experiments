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

- MediaPipe BlazePose (33 landmarks) to COCO-17 subset via index mapping (`mediapipe`).
- OpenPose BODY_25 (25 parts) to COCO-17 subset via index mapping (`openpose`).
- OpenPose COCO model (18 parts) to COCO-17 subset via index mapping (`openpose_coco`).
- AlphaPose COCO-17 to canonical one-to-one mapping (`alphapose`).
- Detectron2 COCO keypoints to canonical one-to-one mapping (`detectron2`).

### OpenPose has two output layouts

`pose_deploy_linevec.prototxt` (the COCO model used by the OpenCV DNN path in this
repository) emits **18** parts, while the native `pyopenpose` default emits **25**.
The two orderings differ: BODY_25 inserts `mid_hip` at index 8 and shifts the whole
leg and face block, so reusing the BODY_25 map on COCO output silently swaps hips,
knees, ankles, eyes, and ears. Pass `openpose_coco` for 18-part output and
`openpose` for BODY_25 output.

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
