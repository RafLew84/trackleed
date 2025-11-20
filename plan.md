# Deep-Learning Extension Plan for LEEDTrack

This document captures the multi-phase plan to add deep-learning based spot segmentation and optical-flow backends to LEEDTrack, while keeping the existing "classic" pipeline working and selectable.

## Phase 0 – Environment and Working Branch

- Create branch: `feature/dl-flow-unet` (develop all DL work there).
- Target environment: Colab Pro (A100 40/80 GB), Python 3.10+, PyTorch with CUDA.
- Baseline libraries: `numpy`, `scikit-image`, `opencv-python`, `albumentations`, `matplotlib`, `pandas`.
- Data layout:
  - `data/segmentation/{train,val,test}/images/*.png`
  - `data/segmentation/{train,val,test}/masks/*.png` – binary 0/1 masks.
  - `data/flow/{train,val,test}/I1/*.png`, `I2/*.png`, `flow/*.(npz|flo)` – optional for RAFT fine-tune.
- Add a config file (YAML or JSON), e.g. `configs/leedtrack_dl.yaml`, with switches:
  - `preproc_backend`: `"classic" | "unet"`
  - `detector_backend`: `"classic" | "unet"`
  - `flow_backend`: `"lk" | "deep"`
  - Thresholds/params: `seg_threshold`, `min_area_px`, `max_area_px`, `bbox_margin_px`, `flow_conf_threshold`.
- Output: config skeleton + a Colab notebook for training and experiments.
- Acceptance: repo still builds and the current pipeline runs with all backends set to `"classic"`.

## Phase 1 – Synthetic Data for Segmentation (and Flow, Optional)

Goal: Build a rich synthetic dataset for spot segmentation, and optionally for optical-flow ground truth.

- Domain characteristics:
  - Spots: Gaussian-like blobs, varying sigma, ellipticity, anisotropy, satellites, blinking.
  - Background: nonlinear gradients, moiré/grid artifacts, vignetting, global drift, ring artifacts, mixed Poisson+Gaussian noise, hot/dead pixels.
  - Motion: typical 0–6 px, occasional 10–15 px, both translations and small local deformations.
- Volumes (segmentation, required):
  - 50k–100k train frames (1-channel, e.g. 512×512 or 640×640).
  - 5k val, 5k test.
  - Labels: binary masks (1 = spot pixels).
- Volumes (flow, optional for RAFT fine-tune):
  - 20k–50k `(I1, I2)` pairs + dense flow GT + occlusion masks.
  - 2k–5k val/test pairs.
- Implementation:
  - Synthetic generator with strong domain randomization of all above factors.
  - For segmentation: save `(image, mask)`.
  - For flow: save `(I1, I2, flow, occlusion)`.
  - Control difficulty mix: ~70% typical, ~20% medium, ~10% extreme.
- Acceptance: dataset statistics validated (distribution of spot counts, sigma, SNR, motion); visual inspection of random samples.

## Phase 2 – Baselines and Metrics

Goal: Quantify current performance of the classic pipeline (spot finding + LK tracking).

- Metrics:
  - Segmentation/detection: IoU, Dice; component-level Precision/Recall/F1 (IoU ≥ 0.3 or centroid distance ≤ 2 px for matches).
  - Tracking: MOTA, IDF1, number of ID switches, fraction of lost tracks.
  - Flow (synthetic): EPE (Endpoint Error), AAE (Average Angular Error).
  - Spot statistics stability: coefficient of variation for intensity/area of stationary spots over time.
- Procedure:
  - Run current classic pipeline on synthetic test sets + a small set of real sequences.
  - Log all metrics as baseline (tables + short text notes).
- Acceptance: baseline metrics computed and stored as reference for later comparison.

## Phase 3 – U-Net for Spot Segmentation

Goal: Train a compact segmentation model that predicts spot masks.

- Model:
  - Small U-Net (3–4 down/upsampling levels, 16–32–64… channels).
  - Input: single-channel image; output: single-channel probability map (sigmoid).
  - Loss: BCE + Dice, with class balancing for the positive class.
- Augmentations:
  - Flips/rotations, gamma/contrast jitter.
  - Additive noise, slight blur, mild affine/perspective.
- Data: same as Phase 1 (segmentation dataset).
- Training:
  - Use early-stopping on validation IoU.
  - Save best checkpoint `unet_spots.pt` by validation IoU.
- Targets (approximate):
  - Synthetic test IoU ≥ 0.85; detection F1 ≥ 0.95.
  - Visual inspection on real LEED frames (overlay masks) looks sensible.
- Acceptance: these quality targets are met or clearly approached.

## Phase 4 – Integrating Segmentation into the Pipeline

Goal: Use the U-Net outputs for pre-processing and detection without rewriting the rest of the pipeline.

- Pre-processing (`preproc_backend="unet"`):
  - Option A: `image_preproc = image * mask` (hard or soft thresholded mask).
  - Option B: background suppression: `image_preproc = image - (1 - mask) * bg_estimate`.
- Detection (`detector_backend="unet"`):
  - Threshold probabilities: `P > seg_threshold`.
  - Connected components → filter by `min_area_px` / `max_area_px`.
  - For each component: centroid, bounding box (with `bbox_margin_px`), local binary mask.
  - Derive spot statistics: area, sum/mean/max intensity, optional moments.
  - Map these back into `SpotROI` or a compatible structure.
- Config additions:
  - `seg_threshold`, `min_area_px`, `max_area_px`, `bbox_margin_px`.
- Acceptance:
  - Synthetic: detection F1 ≥ classic baseline, no regression on key metrics.
  - Real: less background clutter, clean spot ROIs.

## Phase 5 – RAFT as Optional Optical Flow Backend

Goal: Add a deep optical-flow backend with fallback to LK.

- Operation:
  - `flow_backend="deep"` selects RAFT instead of `calcOpticalFlowPyrLK`.
  - Run RAFT on a resized frame (short side 512–768), rescale flow back to original resolution.
  - Sample dense flow at spot centroids to update positions.
  - Forward–backward RAFT consistency: compute flow `t→t+1` and `t+1→t`; if `|fwd + bwd| > tau`, mark vector as unreliable.
  - For unreliable vectors, fall back to LK for that spot.
  - Integrate with skip-anchor logic by running RAFT on (anchor, frame_t) pairs.
- Config parameters:
  - `raft_resize_short_side`, `flow_conf_threshold`, `fwd_bwd_tau`.
- Acceptance:
  - Synthetic flow: EPE not worse than LK, ideally better in challenging cases.
  - Real tracking: fewer ID switches or improved track retention at similar detection thresholds.

## Phase 6 – Optional RAFT Fine-Tuning

Goal: Fine-tune RAFT on the synthetic flow dataset if the pre-trained model is not sufficient.

- Data: 30k–50k training pairs, 3k–5k validation pairs, with strong domain randomization.
- Training:
  - Low learning rate, partial freezing of early layers.
  - Monitor validation EPE and especially performance on small motion (0–6 px).
  - Save `raft_finetuned.pt` at minimum validation EPE.
- Acceptance:
  - EPE improvement vs. pre-trained weights.
  - Measurable gains in tracking metrics when plugged into the pipeline.

## Phase 7 – End-to-End Validation and Ablations

Goal: Understand the contribution of each DL component.

- Compare variants on the same datasets:
  - A) classic preproc + classic detector + LK.
  - B) U-Net preproc + U-Net detector + LK.
  - C) classic preproc + classic detector + RAFT.
  - D) U-Net preproc + U-Net detector + RAFT (with LK fallback).
- Metrics:
  - Detection: Precision/Recall/F1 (synthetic and real).
  - Tracking: MOTA, IDF1, ID switches, track retention.
  - Spot stability: CV for intensity and area vs. time.
- Output:
  - Ablation table + short interpretation.
  - Selection of a recommended default profile.

## Phase 8 – Configuration and Documentation

Goal: Make backend switching and threshold tuning explicit and user friendly.

- Centralize parameters in config files, with clear keys:
  - `preproc_backend`, `detector_backend`, `flow_backend`.
  - `seg_threshold`, `min_area_px`, `max_area_px`, `bbox_margin_px`.
  - `raft_resize_short_side`, `fwd_bwd_tau`, `flow_conf_threshold`.
- Presets:
  - `classic_fast`: original pipeline, fast, no DL.
  - `unet_lk`: U-Net segmentation, LK flow.
  - `unet_raft`: full DL stack with RAFT and LK fallback.
- Documentation:
  - Update README with quick-start for configs and training.
  - Brief guide for Colab training notebooks.

## Plan Assessment

Strengths:
- Clear phase separation: you can stop after segmentation (Phases 3–4) and still get value.
- Synthetic data first makes it easy to iterate on architectures and hyperparameters.
- Metrics and ablations are built-in, so each change can be justified quantitatively.

Risks and Mitigations:
- **Scope**: Full plan (0–8) is large. Mitigation: treat RAFT fine-tune (Phase 6) as optional and prioritize Phases 3–4.
- **Domain gap** between synthetic and real LEED: Mitigation: keep real validation sets from the start, adjust noise/artefacts in the generator to match real data.
- **Compute and storage**: Synthetic volumes are big. Mitigation: patch-based training, on-the-fly generation, and limiting resolution when needed.

Recommended first milestones:
1. Implement synthetic generator (Phase 1) and baseline metrics (Phase 2).
2. Train and integrate U-Net segmentation (Phases 3–4) as the first production-ready DL feature.
3. Only then evaluate whether RAFT integration (Phases 5–7) is worth the added complexity for your LEED datasets.

