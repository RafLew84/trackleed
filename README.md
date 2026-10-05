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
- **YOLO**: Ultralytics YOLO detector using `.pt` checkpoints in `tracking_models/yolo/`.

### Current YOLO Checkpoints (Training History)

- `tracking_models/yolo/yolo11s_pro_v1.pt`  
  Base YOLO model trained on synthetic LEED-like data only.

- `tracking_models/yolo/yolo11s_pro_v1_fine_tuning.pt`  
  First fine-tuning stage on real experimental data (starting from `yolo11s_pro_v1.pt`).

- `tracking_models/yolo/yolo11s_dimmed_full_from_smoke_20260225_154251_best.pt`  
  Second fine-tuning stage for difficult dimmed/blurred spots and mixed backgrounds (starting from the first fine-tuned model).

Notes:
- The latest dimmed model training config is stored in:  
  `tracking_models/yolo/yolo11s_dimmed_full_from_smoke_20260225_154251_train_config.json`
- In the app, choose the checkpoint in the YOLO model dropdown (Find Spots panel).

**Tracking**
- **Detection Linking (Hungarian)**: assignment by distance between detections.
- **Detection Linking (Directional)**: radial-aware Hungarian with lateral penalties and ring-bin gating.
- **Optical Flow**: sparse Lucas–Kanade tracking from a reference frame.
- **Optical Flow (skip, anchor)**: LK with skip-anchor logic, optional forward-backward and NCC checks.

**Analysis**
- 2D Gaussian fitting per ROI and pattern center estimation using a robust symmetry-based method (RANSAC-style).
- Per-frame radii and errors are computed from the fitted spot centers relative to the pattern center.

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

7) Analyze:
- Analyze Tracks -> choose Pattern Center Mode (per-frame RANSAC, per-frame circle fit, or global median) and run analysis.
- Use **Save Session** in the analysis window to store results without leaving the dialog.

8) Export:
- File -> Export Spots to YOLO Labels... or Export All (images + YOLO labels)

9) Save sessions:
- File -> Save Session... / Load Session... (full sessions, including analysis results).

## UI Tips and Shortcuts

- Modes (main toolbar): `P` Pan, `A` Add ROI, `M` Move ROI, `D` Delete ROI.
- Cycle modes: `L` cycles Pan -> Add -> Delete.
- BBox transparency:
  - `Ctrl+0` = 0%, `Ctrl+2` = 20%, `Ctrl+5` = 50%, `Ctrl+8` = 80%.
  - `O` cycles 0% -> 20% -> 50% -> 80% -> 0%.
- Quick Save/Load:
  - `Ctrl+S` Quick Save to `save/quick_save.lts.json`
  - `Ctrl+L` Quick Load from `save/quick_save.lts.json`

## Analysis Details

### What is computed

For each tracked spot on each frame:
- A 2D **elliptical Gaussian** is fitted to the ROI.
- The fitted **global center** is stored (ROI position + fit center).
- Derived metrics are computed: **volume**, **FWHM X/Y**, **asymmetry**, **R-squared**, and parameter errors.

Pattern center:
- **Per-frame (RANSAC)**: RANSAC is run on the fitted spot centers of that frame.
- **Per-frame (circle fit)**: least-squares circle fit to spot centers (stable for symmetric rings).
- **Global median**: per-frame centers are computed, then a single median center is applied to all frames.

Radius:
- For each spot and frame, `radius = distance(global_center, pattern_center)`.
- Error propagation uses spot center errors and pattern center errors.

Frame-level summary:
- **Average radius per frame** and its standard error are computed across tracks.

### Analysis UI: plots and tables

**ROI 3D Preview**
- Left: 3D surface of the raw ROI data.
- Right: 3D surface of the fitted Gaussian model.

**Dependency Plots**
- Plots any chosen metric (e.g., Radius, Global Center X/Y, Volume, Amplitude, FWHM X/Y, Asymmetry, R-squared)
  against Frame ID, Temperature, or Energy.

**Position Plot**
- 2D plot of spot positions (global centers) over time.
- Optional overlay of the pattern center and track traces.

**Comparative Plots**
- Compare multiple tracks on the same axes for a selected metric.

**Results Table**
- Tabular view of per-frame/per-track metrics with filters (frame and parameter selections).

### Exported analysis outputs

When you click **Export Results**, the following are written:
- `full_analysis_per_spot.csv`: per-track, per-frame fit results and derived metrics.
- `pattern_center_data.csv`: pattern center per frame.
- `average_radius_per_frame.csv`: average radius per frame with error.
- `track_legend.csv`: track IDs with their starting positions.
- `track_visualizations/`: PNG overlays with track traces (every 5th frame).

### Exported positions-analysis outputs

The **Positions Analysis -> Export Results** action writes one directory per configured
group. Each group directory contains `positions_analysis_results_group_XX.csv` and a
`positions_overlay_images/` directory.

The positions CSV preserves the position columns (`frame_id`, `group_id`, `point_role`,
`track_id`, `distance_from_reference`, `x`, and `y`) and also exports the stored Gaussian
fit metadata:

- `position_source`: `gaussian_fit` when `x`/`y` come from an accepted fit, otherwise
  `tracked_center` when the tracked ROI center is used as a fallback.
- `fit_ok`, `quality_status`, and `quality_flags`: fit validation outcome and diagnostics.
- `sigma_x`, `sigma_y`, `fwhm_x`, `fwhm_y`, `fwhm_x_error`, and `fwhm_y_error`: Gaussian
  widths in pixels. The X/Y names refer to the principal axes of the rotated Gaussian,
  not fixed image axes.
- `theta`: rotation of the Gaussian principal axes in radians.
- `r_squared`: goodness-of-fit diagnostic when available.

Fits with `quality_status=warning` retain their width values and warning flags. Rows with
`fit_ok=false` retain their tracked position and diagnostics, but their sigma, FWHM,
FWHM-error, and theta fields are empty. Export uses the fit results already stored in the
positions-analysis session; it does not reload TIFF files or rerun Gaussian fitting.

### Non-destructive publication export

Use the batch exporter to create a separate, self-contained copy of saved exports and
analysis sessions with FWHM fields added where stored fit data permits:

```bash
python scripts/export_publication_data.py \
  --source data \
  --destination publication_exports
```

The destination must be outside the source tree. Dedicated export directories such as
`eksport`, `results`, `wew`, `zew`, and `group_*` are copied in full, including unrelated
sidecar files and visualization directories. Standalone result CSV files and saved
`.positions.json`/`.lts.json` sessions are copied without copying neighboring raw TIFF
images.

Copied per-spot CSV files retain every original row and column. The exporter uses existing
`sigma_x` and `sigma_y` values when needed without adding new sigma columns, and completes
`fwhm_x`, `fwhm_y`, their FWHM errors,
`fwhm_mean`, `fwhm_geometric`, and `fwhm_source`. The provenance value is `stored`,
`calculated_from_sigma`, or `missing`; failed fits retain their rows but publish no width
values. Each copied CSV receives `fwhm_per_frame.csv`, and the output root receives
`export_manifest.json` with source/output hashes, row counts, provenance counts, and the
input fingerprint.

The command never reads TIFF pixel data or reruns Gaussian fitting. It reuses an existing
destination only when its manifest fingerprint exactly matches the current inputs;
otherwise it fails instead of overwriting files.

## Notes

- YOLO models: place `.pt` checkpoints in `tracking_models/yolo/` and click **Refresh**.
- Legacy fallback: the app also scans `segmentation_models/` for backwards compatibility.
- The quick-save file is independent from standard session save/load dialogs.

## Developer Notes

- Tracker integration guide: [docs/tracker_integration.md](docs/tracker_integration.md)

## Synthetic tracking benchmark

For complete stationary H6 sequences with known positions and persistent IDs,
see [the benchmark guide](docs/synthetic-benchmark.md). The separate generator
writes float32 TIFF, fixed-scale PNG8, versioned reference tables and a verified
manifest. It preserves the existing segmentation-training workflow.
