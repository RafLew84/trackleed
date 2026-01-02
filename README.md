# STAL - Spot Tracking & Analysis for LEED

Trackleed is a PyQt6 desktop application for loading LEED image stacks, detecting diffraction spots, tracking them over time, and fitting spot parameters for quantitative analysis.

## Overview

- Load LEED frame sequences (TIFF stacks or folders).
- Detect diffraction spots (classic image processing or YOLO).
- Track spots across frames (Hungarian linking or optical flow).
- Fit 2D Gaussians per spot and derive radii relative to the pattern center.
- Export YOLO labels and images for dataset building, and save/load sessions.

## Methods & Models

**Spot detection**
- **Classic**: white-top-hat + peak finding with tunable blur, WTH radius, and ROI size.
- **YOLO**: Ultralytics YOLO detector using `.pt` checkpoints in `segmentation_models/`.

**Tracking**
- **Detection Linking (Hungarian)**: assignment by distance between detections.
- **Detection Linking (Directional)**: radial-aware Hungarian with lateral penalties and ring-bin gating.
- **Optical Flow**: sparse Lucas–Kanade tracking from a reference frame.
- **Optical Flow (skip, anchor)**: LK with skip-anchor logic, optional forward-backward and NCC checks.

**Analysis**
- 2D Gaussian fitting per ROI and pattern center estimation using a robust symmetry-based method (RANSAC-style).

Note: a U-Net segmentation backend exists in code and can be re-enabled later if needed.

## Quick Start (New Users)

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run the app:
```bash
python -m trackleed
```

3) Load data:
- File -> Open Directory... (select a folder with LEED frames)
- Optional: File -> Load Background File... to enable background subtraction

4) Detect spots:
- Analysis Parameters -> Find Spots / Find All Spots
- Choose **Classic** or **YOLO** in "Spot detector"

5) Track:
- Tracking -> choose an algorithm:
  - Detection Linking (Hungarian) for simple linking
  - Detection Linking (Directional) for radial in/out motion
  - Optical Flow for tracking from a selected reference frame

6) Review and edit:
- Visualize Tracking -> edit, delete, add, or merge tracks

7) Export:
- File -> Export Spots to YOLO Labels... or Export All (images + YOLO labels)

8) Save sessions:
- File -> Save Session... / Load Session...

## UI Tips and Shortcuts

- Modes (main toolbar): `P` Pan, `A` Add ROI, `M` Move ROI, `D` Delete ROI.
- Cycle modes: `L` cycles Pan -> Add -> Delete.
- BBox transparency:
  - `Ctrl+0` = 0%, `Ctrl+2` = 20%, `Ctrl+5` = 50%, `Ctrl+8` = 80%.
  - `O` cycles 0% -> 20% -> 50% -> 80% -> 0%.
- Quick Save/Load:
  - `Ctrl+S` Quick Save to `save/quick_save.lts.json`
  - `Ctrl+L` Quick Load from `save/quick_save.lts.json`

## Notes

- YOLO models: place `.pt` checkpoints in `segmentation_models/` and click **Refresh**.
- The quick-save file is independent from standard session save/load dialogs.
