# Figure 4 — Ag(111) motion and visibility-gap stress test

Status: generated and validated.

## Scientific role

Figure 4 is the main Ag(111) stress-test display. It keeps the expected radial
branch, measured pair distance and frozen visibility regimes together, then
presents the four outcomes in the required order:

1. branch-consistent measurement coverage over 221 baseline-visible
   opportunities per method;
2. post-gap branch-consistent recovery over 24 evaluable natural I(V) gaps;
3. geometry-inconsistent continuation over each method's accepted fits;
4. conditional pair-distance residual with its method-specific surviving `n`.

The figure uses stored proposals and common Gaussian measurements. It is a
physics-constrained consistency assessment, not image-by-image identity ground
truth, an absolute-accuracy experiment or a tracker ranking. Pair-midpoint
drift and the EasyLEED–ViPErLEED compatibility check are deliberately absent
from the main figure and appear in Figures S3 and S4.

## Caption

**Figure 4. Ag(111) motion and visibility-gap stress test.** **a,** Expected
pair-distance contraction, framewise cross-method median and min–max range for
the predeclared representative Scan A pair, with the two frozen branch
visibility regimes shown separately. The expected `D(E) ∝ 1/sqrt(E)` curve is
a physics-consistency reference rather than absolute ground truth. **b,**
Branch-consistent Gaussian measurement coverage among 221 baseline-visible
spot opportunities per method. **c,** Return to the same frozen radial branch
after 24 evaluable natural I(V) gaps, without manual reinitialization. **d,**
Accepted Gaussian fits outside the 18 px branch-consistency gate, reported over
each method's accepted-fit denominator; these events indicate possible false
continuity but do not prove identity switches. **e,** Median absolute
pair-distance residual conditional on both branches passing proposal,
Gaussian-fit and geometry gates, with the method-specific conditional `n`.
Coverage and recovery precede inconsistency and conditional residual so the
conditional metric cannot hide missing measurements.

## Artifacts and rebuild

- `source_data/` contains the five-method aggregate, all four case strata,
  representative pair measurements, branch visibility regimes and provenance.
- `contract.json` freezes the panel and denominator contract.
- `manifest.json` hash-links code, source data and PNG/PDF outputs.

```bash
python freeze_source_data.py --validate-only
python build_figure.py --validate-only
python -m unittest test_build_figure.py test_freeze_source_data.py
```
