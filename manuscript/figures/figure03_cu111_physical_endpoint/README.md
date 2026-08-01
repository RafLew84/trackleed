# Figure 3 - propagation to the Cu(111) physical endpoint

Status: generated and validated.

## Scientific role

This main-text figure is the Cu(111) positive control. It asks whether
tracker-dependent proposal differences survive the common Gaussian refinement
sufficiently to change the pairwise relative spot-radius endpoint in one
701-image thermal cycle acquired at a fixed 56 eV.

The five strategies each provide `2804/2804` valid Gaussian measurements.
Both frozen primary pairs, 1-6 and 3-5, provide `701/701` valid endpoints per
method. Pair 2-4 is intentionally absent because its earlier saved-YOLO
proposal trajectory is outside the five-method run; that separate missingness
diagnostic is Figure S6.

## Panels

- **a.** Temperature profile and the frozen reference, 800 °C plateau,
  outcome-selected post-plateau minimum, and final cooled-state landmarks.
- **b,c.** Pairwise `Delta R/R = Delta D/D` for pairs 1-6 and 3-5, shown as
  the five-method median, min-max envelope, and magnified method-minus-median
  residuals.
- **d.** Between-method endpoint range at the plateau, post-plateau minimum,
  and final cooled state.

## Caption

**Figure 3. Tracker proposal differences do not materially alter the Cu(111)
pairwise relative spot-radius endpoint after common Gaussian refinement.**
**a,** Temperature profile of the 701-image constant-energy thermal cycle.
**b,c,** Pairwise `Delta R/R = Delta D/D` after proposals from LocoTrack,
OSTrack, DAM4SAM, Lucas-Kanade, and EasyLEED were passed through the same
Gaussian estimator. Every method produced `2804/2804` accepted Gaussian
measurements and `701/701` valid endpoints for each pair. **d,** Between-method
endpoint range at the predefined plateau, the
common post-plateau minimum, and the final cooled state. Maximum separation
over the complete sequence was 0.001710 percentage points for pair 1-6 and
0.002237 percentage points for pair 3-5. This agreement is evidence from one
positive-control sequence, not absolute accuracy or a tracker ranking. The two
pairs do not establish crystallographic anisotropy, a calibrated
thermal-expansion coefficient, a phase transition, or a structural mechanism.

## Rebuild and validate

```bash
python freeze_source_data.py
python freeze_source_data.py --validate-only
python build_figure.py
python build_figure.py --validate-only
python test_build_figure.py
```

Final artifacts are written to `output/`; `manifest.json` authenticates the
contract, source data, builder and outputs.
