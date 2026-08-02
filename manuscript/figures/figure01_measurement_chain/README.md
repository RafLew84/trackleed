# Figure 1 — Measurement chain and evidence ladder

Status: generated and validated.

This directory freezes the real LEED image, one canonical spot, the five
archived proposal coordinates, and the five common Gaussian measurements that
will anchor the reader-facing measurement-chain schematic.

## Final artifacts

- [PNG](output/figure01_measurement_chain.png)
- [PDF](output/pdf/figure01_measurement_chain.pdf)
- [render manifest](render_manifest.json)

The source-case integrity record remains in `manifest.json`; the final
PNG/PDF, renderer, environment, and deterministic-rebuild hashes are recorded
separately in `render_manifest.json`.

## Caption

**Figure 1. Tracking proposal, common position measurement, and evidence
ladder.** **a,** Native outputs of five strategies for the same Cu(111) spot:
LocoTrack point, OSTrack box, DAM4SAM mask, Lucas–Kanade optical-flow point,
and EasyLEED Kalman position. The view is the final image of the 701-image
fixed-energy thermal cycle at 56 eV and 92 °C. The metadata-selected case used
one sequence, spot, and seed without correction or reinitialization and was
selected without inspecting tracker behavior; the empirical OSTrack box and
unsmoothed DAM4SAM mask are retained. **b,** Each native output is reduced to
a proposal coordinate centering the common 48 × 48-pixel ROI; one elliptical
Gaussian fit to the raw pixels supplies the reported position. Proposal
centers differ by at most 5.60 px and refined centers by 0.0023 px in this
explanatory case, which was not selected from tracker outcomes and does not
rank methods. **c,** Two refined positions define pair distance \(D_t\) and
\(\Delta D/D\); common translation cancels, whereas non-rigid distortion does
not. **d,** Controlled translations, Cu(111), independent Ag(111) scans, and
the controlled SAM2 family test equivariance, endpoint agreement, motion plus
natural \(I(V)\) gaps, and a memory mechanism, respectively. They retain
separate denominators and do not form a composite ranking.

## Selection

The case was selected without inspecting tracker outputs:

1. take the predeclared diagnostic probe whose reason is
   `evaluation_range_end`;
2. take the lowest canonical core spot ID.

This gives the final image of the 701-image Cu(111) sequence, acquired at
92 °C and 56 eV, and spot 1. The frame is already a metadata-selected
diagnostic anchor, and spot 1 is selected by identity order rather than by
tracker behavior.

All five methods process the same 701-image sequence and the same selected
spot. They start at image 0 from the same Gaussian-refined seed center
(`x = 140.7950 px`, `y = 712.6157 px`). Point methods receive that center;
box- and mask-based methods receive the shared `48 x 48 px` box centered on
it. The seed is an initialization prompt, not ground truth, and correction or
reinitialization is forbidden.

After selection, the completeness check found:

- five of five native proposals available;
- five of five accepted common Gaussian fits;
- no manual annotation or reinitialization;
- maximum pairwise separation of the reduced native proposal coordinates:
  `5.599510 px`;
- maximum pairwise separation of the common Gaussian centers: `0.002306 px`.

These separations describe the selected case; they were not used to select it
and do not constitute a qualitative method ranking.

## Native-geometry re-export

The initial publication run retained the native point for LocoTrack, the
accepted Lucas-Kanade point, and the EasyLEED Kalman position, but reduced the
OSTrack bbox to its center and the DAM4SAM mask to its centroid. This gap has
now been resolved by a targeted replay of the frozen configurations.

The replay processed all 701 images chronologically from the same image-0
prompt, for spot 1 only and without correction or reinitialization. Restricting
the replay to one query does not change its result because both frozen adapters
process queries independently. The re-exported reduced trajectories reproduce
the archived run over all 701 images:

- OSTrack maximum center difference: `0 cache px`;
- DAM4SAM maximum centroid difference: `0 cache px`;
- proposal availability: identical on all 701 images for both methods.

The selected OSTrack bbox is `40.3547 x 41.5865 px` on the original image. The
selected DAM4SAM mask contains 98 true pixels on the native `256 x 256`
tracking grid. Its mask pixels are stored losslessly, not reconstructed from
the centroid.

The final renderer must show these empirical geometries wherever it depicts the
OSTrack bounding-box output or the DAM4SAM mask output. It must use the frozen
bbox coordinates and binary mask without smoothing, redrawing, or substituting
schematic geometry.

## Files

- `contract.json` - frozen selection, completeness, descriptive geometry, claim
  limit, and native-geometry replay contract.
- `export_native_geometry.py` - targeted full-sequence GPU replay and native
  geometry validator.
- `freeze_case.py` - deterministic source-case extractor and validator.
- `test_export_native_geometry.py` - bbox/mask reduction contract tests.
- `test_freeze_case.py` - selection, completeness, and package tests.
- `source_data/representative_native_geometry.json` - empirical OSTrack bbox,
  DAM4SAM mask geometry, replay consistency, and source hashes.
- `source_data/representative_native_geometry_replay.csv` - both re-exported
  reduced trajectories on all 701 images, used to recompute replay consistency.
- `source_data/representative_dam4sam_mask_cache.png` - lossless binary mask on
  the native 256 x 256 tracking grid.
- `source_data/representative_initialization.json` - shared sequence, spot,
  seed center, prompt box, and no-reinitialization contract.
- `source_data/representative_method_outputs.csv` - one row per method with
  proposal coordinate, ROI, fit center, and fit diagnostics.
- `source_data/representative_spot_crop.png` - lossless 128 x 128 crop from the
  raw TIFF.
- `source_data/representative_spot_crop.json` - crop coordinates and image
  provenance.
- `source_data/source_provenance.json` - hashes of all upstream artifacts.
- `manifest.json` - code, contract, and committed source-data hashes.
- `figure_contract.py` - reader-facing rendering contract and cross-checks.
- `figure_render.py` - deterministic four-panel renderer.
- `build_figure.py` - PNG/PDF build and deterministic validation.
- `test_build_figure.py` - final-figure contract tests.
- `render_manifest.json` - final code, source, output, and environment hashes.

## Rebuild

When the frozen local publication artifacts are available:

```bash
python freeze_case.py
```

Validate the committed case without rewriting it:

```bash
python freeze_case.py --validate-only
```

This full validation checks recorded upstream hashes and compares the committed
files byte-for-byte with a fresh deterministic rebuild. When the upstream
artifacts are unavailable, only the integrity of the committed figure package
can be checked:

```bash
python freeze_case.py --offline-integrity-only
```

Run the contract tests:

```bash
python test_export_native_geometry.py
python test_freeze_case.py
python test_build_figure.py
```

The native geometry does not need to be regenerated during an ordinary figure
build. To reproduce the expensive GPU replay explicitly:

```bash
python export_native_geometry.py
```

Build and validate the final figure:

```bash
python build_figure.py
python build_figure.py --validate-only
```
