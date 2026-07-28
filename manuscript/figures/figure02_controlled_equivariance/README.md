# Figure 2 - controlled equivariance and energy-scan geometry

Status: generated and validated.

This directory is the self-contained publication artifact for Figure 2. It
separates the controlled response of the complete
`proposal -> ROI -> Gaussian position` chain from the energy-dependent geometry
used in the Ag(111) stress tests.

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
- **e.** Pair separation `D(E)` and common midpoint drift for the frozen primary
  pair of the memory-analysis energy scan. Missing pair measurements are shown as
  gray marks and are not connected across gaps.

## Caption draft

**Figure 2. Controlled equivariance of the measurement chain and
energy-dependent pair geometry.** (a) Frozen closed translation path applied to
four independently initialized Cu(111) states; the representative initial
10 °C image is shown. (b) Median displacement-equivariance error before and
after the common two-dimensional Gaussian estimator. The oracle ROI estimates
the interpolation-plus-estimator floor without a tracker. (c) Valid Gaussian
measurements out of 80 predefined spot opportunities per method. (d) Maximum
absolute pairwise relative-distance change under common image translation.
Valid-pair counts out of 40 predefined opportunities are printed beside each
method.
(e) Measured pair distance compared with the expected `1/sqrt(E)` contraction,
with the pair midpoint drift retained as a separate apparatus/projection
diagnostic. Gray marks denote unavailable pair measurements. Controlled
translations test measurement-chain equivariance, not experimental dynamics or
absolute spot identity.

## Files

- `build_figure.py` - compact build, manifest, and validation entry point.
- `figure_contract.py` - frozen data contract and source-data preparation.
- `figure_render.py` - deterministic five-panel renderer.
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
