# Figure S2 — measurement-window and capture-contract sensitivity

Figure S2 is the sole sensitivity figure. Panels a–d show the nine-element
ROI × proposal-to-fit matrix and six one-factor-at-a-time perturbations for
coverage, recovery, geometry inconsistency, and conditional pair-distance
residual. Panel e shows the signed fifth-percentile central-contract margin
for every acceptance gate, with negative counts and finite-margin n; its color
uses the frozen gate-specific normalizer and is clipped only for display.
Panel f shows the predefined per-method width-trigger thresholds and the
validated decision not to run the conditional width sweep.

The orange outline marks the frozen central contract (`48 px` ROI and `15 px`
proposal-to-fit gate). The figure is descriptive and does not select a new
central configuration. Figure 2 contains no sensitivity panel.

Build from the repository root with:

```bash
python -m manuscript.figures.figure_s2_measurement_sensitivity.build_figure
```

The authenticated sources are in `source_data/`; `manifest.json` binds source,
implementation, rendering, and output hashes.
