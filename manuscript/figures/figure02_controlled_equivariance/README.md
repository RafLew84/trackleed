# Figure 2 — Controlled estimator behavior

Status: generated and validated.

This directory is the publication artifact for the controlled response of the
complete `proposal -> ROI -> Gaussian position` chain. Energy-scan geometry and
measurement-contract perturbations are intentionally absent from Figure 2.

## Panels

- **a.** The frozen six-step closed translation path over a crop from the initial
  10 °C Cu(111) image. The dashed square is the 48 x 48 measurement ROI at the
  seed position.
- **b.** Median native-proposal and Gaussian-refined displacement-equivariance
  errors. The vertical scale is logarithmic; the dashed line is the oracle-ROI
  median.
- **c.** Gaussian-measurement coverage with the common denominator of 80
  opportunities per method.
- **d.** Maximum absolute pairwise `Delta D/D` under the same translations. The
  valid-pair count is printed against the common denominator of 40 opportunities
  per method. The oracle-ROI result is shown as a reference rather than as a
  sixth tracker.
## Caption draft

**Figure 2. Controlled equivariance of the measurement chain.** (a) Frozen
closed translation path applied to
four independently initialized Cu(111) states; the representative initial
10 °C image is shown. (b) Median displacement-equivariance error before and
after the common two-dimensional Gaussian estimator. The oracle ROI estimates
the interpolation-plus-estimator floor without a tracker. (c) Valid Gaussian
measurements out of 80 predefined spot opportunities per method. (d) Maximum
absolute pairwise relative-distance change under common image translation.
Valid-pair counts out of 40 predefined opportunities are printed beside each
method. Known translations assess measurement-chain equivariance and common-
estimator behavior, not absolute tracking accuracy or identity ground truth.

## Files

- `build_figure.py` - compact build, manifest, and validation entry point.
- `figure_contract.py` - frozen data contract and source-data preparation.
- `figure_render.py` - deterministic four-panel renderer.
- `test_build_figure.py` - tests for the frozen path and denominators.
- `source_data/` - committed figure-level tables, representative crop, and
  provenance hashes.
- `output/figure02_controlled_equivariance.png` - 400 dpi raster output.
- `output/pdf/figure02_controlled_equivariance.pdf` - vector output.
- `manifest.json` - source, source-data, builder, environment, palette, and output
  hashes.

## Rebuild

The default command rebuilds the figure from the committed source data and
validates byte-identical PNG and PDF outputs:

```bash
python build_figure.py
```

When the upstream frozen artifacts are available, refresh the figure-level source
data before rebuilding:

```bash
python build_figure.py --refresh-source-data
```

Validate without rewriting the outputs:

```bash
python build_figure.py --validate-only
```
