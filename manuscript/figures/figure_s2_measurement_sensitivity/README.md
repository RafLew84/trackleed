# Figure S2 — measurement sensitivity

Figure S2 is the sole sensitivity figure. Panels a–d show the nine-element
ROI × proposal-to-fit matrix and six one-factor-at-a-time perturbations for
coverage, recovery, geometry inconsistency, and conditional pair-distance
residual. Panel e shows the predefined per-method gate-margin thresholds and
the validated decision not to run the conditional width sweep.

The orange outline marks the frozen central contract (`48 px` ROI and `15 px`
proposal-to-fit gate). The figure is descriptive and does not select a new
central configuration. Figure 2 contains no sensitivity panel.

Build from the repository root with:

```bash
python -m manuscript.figures.figure_s2_measurement_sensitivity.build_figure
```

The authenticated sources are in `source_data/`; `manifest.json` binds source,
implementation, rendering, and output hashes.
