# human-pose-estimation-experiments

This repository contains a collection of hands-on experiments with human pose estimation tools, including **MediaPipe**, **OpenPose**, **AlphaPose**, and **Detectron2**. The goal is to explore real-time and offline pose estimation techniques, visualize keypoints, and extract motion features for analysis. All notebooks are compatible with **Google Colab** and organized by tool for clarity and modularity.

---

## Folder Structure
```
./
├── README.md
├── requirements.txt
├── MediaPipe/
│   ├── demo.ipynb
│   └── keypoints_to_csv.ipynb
├── OpenPose/
│   └── install_and_run.ipynb
├── AlphaPose/
│   └── colab_inference.ipynb
└── Detectron2/
    └── pose_estimation.ipynb
```

---

## Tools Covered

- **MediaPipe** – Lightweight and fast, suitable for real-time applications.
- **OpenPose** – Pioneering multi-person 2D/3D pose estimation framework.
- **AlphaPose** – High-accuracy bottom-up pose detection framework.
- **Detectron2** – Flexible object detection and keypoint estimation by Facebook AI.

---

## Use Cases Explored

- Real-time pose visualization with webcam
- Keypoint extraction and CSV export for motion analysis
- Simple gesture or posture recognition
- Body angle and joint movement calculation *(coming soon)*
- Framework speed and accuracy comparison *(coming soon)*

---

## Notes

- All notebooks are built for **Google Colab**, eliminating the need for local installation.
- You can clone this repo and open any `.ipynb` notebook directly in Colab.
- MediaPipe operates on CPU; others may require GPU runtimes in Colab.

---

