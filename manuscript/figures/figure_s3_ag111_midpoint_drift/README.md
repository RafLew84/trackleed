# Figure S3 — Pair-midpoint drift and apparatus/projection diagnostics

Status: generated and validated.

Figure S3 reports every valid physics-baseline pair measurement for the four
frozen Scan A/Scan B pair cases. Each panel shows the pair-distance residual,
pair-midpoint `dx`, `dy` and displacement magnitude together with the two
branch-specific visibility ribbons. Valid-pair denominators are 17/35, 20/35,
25/46 and 27/46; missing pair positions are neither interpolated nor joined.

**Caption.** **a–d,** Pair-distance residual relative to the frozen radial law
and pair-midpoint displacement for Scan A pair 1, Scan A pair 2, Scan B pair 1
and Scan B pair 2, respectively. Colored B1/B2 ribbons retain apparatus
absence, beam onset/termination, baseline-visible and true I(V)-gap regimes.
Midpoint motion is an apparatus/projection nuisance diagnostic kept separate
from the translation-invariant pair-distance outcome; it is not absorbed into
tracker error and does not turn the radial law into absolute position ground
truth.

`freeze_source_data.py` builds the four-case source package and provenance;
`build_figure.py` creates deterministic PNG/PDF outputs and `manifest.json`.

```bash
python freeze_source_data.py --validate-only
python build_figure.py --validate-only
python -m unittest test_build_figure.py
```
