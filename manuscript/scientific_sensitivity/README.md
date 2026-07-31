# Ag(111) measurement-window and capture-contract sensitivity

This assessment reuses the authenticated five-method, four-case Ag(111)
inputs and stored tracking proposals. It does not run tracker inference, select
a preferred configuration, or modify the frozen central measurement contract.

Run the complete headless CPU reconstruction and validation with:

```text
python -m scripts.reconstruct_publication --clean --validate
```

The generated sensitivity artifacts are written below
`manuscript/scientific_panel/output/sensitivity/` and are intentionally ignored
by Git. They include the three unique ROI candidate sets, all threshold-only
evaluations, per-case and aggregate results, first and full rejection-cause
counts, the gate-margin audit, provenance, hashes, and a fail-closed validation
report.

## Predeclared assessment

- Nine `ROI 32/48/64 x proposal-to-fit 12/15/18 px` combinations.
- Branch-tolerance OFAT variants at `12` and `24 px` around the central `18 px`.
- R-squared OFAT variants at `0.80` and `0.90` around the central `0.85`.
- Amplitude-gate OFAT variants at `0.75x` and `1.25x` around the central `1.00x`.
- Fixed reference sets of `221` baseline-visible opportunities and `24`
  evaluable I(V) gaps for every method and variant.
- Explicit variant-specific denominators for geometry-inconsistent continuation
  and conditional pair-distance residuals.

ROI changes refit the stored proposals against the frozen raw images. Threshold
changes reclassify the corresponding candidate rows without rerunning or
copying Gaussian fits. Width bounds remain inclusive post-fit publication gates;
the optimizer hard bounds are unchanged.

## Conditional width decision

The central gate-margin audit did not activate either predeclared materiality
condition, so the five width contracts were not run:

| Method | Finite candidates | Near a width boundary | `ceil(5% N)` | Width-only rejections | Trigger |
|---|---:|---:|---:|---:|---|
| EasyLEED | 313 | 5 | 16 | 0 | no |
| LocoTrack | 324 | 3 | 17 | 0 | no |
| OSTrack | 324 | 3 | 17 | 0 | no |
| DAM4SAM | 309 | 1 | 16 | 0 | no |
| Lucas-Kanade | 242 | 0 | 13 | 0 | no |

The width-only threshold is `12/221` for every method. Per-case values are
retained as diagnostics but do not drive the global decision. This is a
heuristic materiality rule, not a statistical-significance test.
