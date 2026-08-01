# Figure S4 — EasyLEED–ViPErLEED compatibility

Status: generated and validated.

This is a separate 15-image, two-pair external-reference contract. EasyLEED
uses one Kalman tracker per frozen spot; ViPErLEED uses a registered 18-beam
whole-pattern trajectory. Each method supplies 60 branch opportunities to the
same Gaussian estimator, yielding 38/60 accepted common fits, four of four
branch-gap checks, and conditional pair denominators of five P1 and three P2
images per method.

**Caption.** **a,** EasyLEED–ViPErLEED separation before and after common
Gaussian refinement for matched branch/frame records. **b,** Accepted common
fits for the four branches, each over 15 images. **c,** Conditional median
absolute pair-distance residual with `n=5` for P1 and `n=3` for P2. **d,**
Initialization, denominator and interpretation contract. ViPErLEED is not a
sixth method in the main five-strategy matrix, and neither method supplies
position or identity ground truth.

The source package is a byte-identical freeze of the completed high-energy
compatibility artifact plus a normalized contract and hash provenance.

```bash
python freeze_source_data.py --validate-only
python build_figure.py --validate-only
python -m unittest test_build_figure.py
```
