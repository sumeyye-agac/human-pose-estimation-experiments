# Asset Sources

Media files under `assets/` are **not committed**. They are fetched on demand by the
notebooks (see `.gitignore`), so this file is the provenance record for every input
the experiments depend on.

## Demo input image

- `pose_demo_full_body.jpg` — Photo by Valerie Elash on Unsplash
  https://unsplash.com/photos/woman-standing-near-outdoor-during-daytime-aHdkrxbQhZo
  License: Unsplash License (free for commercial/non-commercial use, no attribution required)

Stable download URL used by the notebooks (1080 px wide re-encode of the original
3907x5861 file):

```text
https://images.unsplash.com/photo-1560362614-89027598847b?auto=format&fit=max&w=1080&q=85
```

Fetch it manually with:

```bash
curl -L -o assets/pose_demo_full_body.jpg \
  "https://images.unsplash.com/photo-1560362614-89027598847b?auto=format&fit=max&w=1080&q=85"
```

Why this image: it is a single, fully visible standing person with unoccluded limbs,
so all four tools exercise the complete canonical COCO-17 mapping instead of the
head-only crop used previously.

## Other inputs

- `sample_input_motion.mp4` — short clip used by the sequence/time-series notebooks.
  Downloaded at runtime from `opencv_extra` test data; a synthetic clip is generated
  locally when the download fails.
- `generated/` — overlays and figures written by notebook runs. Always reproducible,
  never committed.
