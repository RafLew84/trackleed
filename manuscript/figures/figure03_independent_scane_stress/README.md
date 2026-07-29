# Figure 3 - independent Ag(111) energy-scan stress test

Status: generated and validated.

## Final artifacts

- [PNG](output/figure03_independent_scane_stress.png)
- [PDF](output/pdf/figure03_independent_scane_stress.pdf)
- [manifest](manifest.json)
- [frozen source contract](contract.json)

## Caption

**Figure 3. Independent Ag(111) energy-scan stress test.** **a,** Pair
distance as a function of electron energy for a representative frozen pair
from Scan A. The case was selected by the predeclared physics rule before
method outcomes were inspected: it is the first ordered case that passed the
coherent pairwise energy-law check and contains natural I(V) visibility gaps.
The dashed curve is the expected \(D(E)\propto1/\sqrt{E}\) trajectory and the
neutral markers are framewise cross-method medians of the available valid
Gaussian pair measurements. Vertical whiskers show the corresponding
cross-method minimum-to-maximum range. Between two and five methods contribute
per displayed image; the largest range is 0.066 px and is smaller than the
marker at this scale. The two ribbons show the tracker-independent visibility
regimes of the individual branches. Image identity is retained, so repeated
terminal-energy acquisitions are not collapsed. **b,** Branch-consistent
Gaussian measurement coverage among 221
baseline-visible spot opportunities per method. **c,** Recovery of the same
physics-constrained branch after 24 evaluable natural I(V) gaps, without
manual reinitialization. **d,** Accepted Gaussian measurements lying outside
the frozen 18-pixel branch-consistency gate, reported over all accepted
measurements in all visibility regimes. These events indicate possible false
continuity but are not external ground-truth proof of an identity switch.
**e,** Median absolute pair-distance residual conditional on both branches
passing the proposal, Gaussian-fit, and geometry gates; the method-specific
numbers of jointly consistent pair images are shown as \(n\). This conditional
quantity must be interpreted after coverage and recovery rather than as an
overall accuracy ranking. **f,** Visually separated external domain reference
over a restricted high-energy interval. EasyLEED uses one Kalman tracker per
frozen spot, whereas ViPErLEED uses a registered 18-beam whole-pattern
trajectory. Both supply proposals to the same Gaussian estimator. The inset
has a separate 60-opportunity denominator and initialization contract;
its pair-residual medians are conditional on 5 jointly valid P1 images and
3 jointly valid P2 images per method.
ViPErLEED is neither a sixth method in the main matrix nor position ground
truth.

## Frozen source package

The self-contained `source_data/` directory contains:

- `aggregate_method_results.csv` - five-method headline counts and
  conditional residuals;
- `case_results.csv` - all four frozen cases used to reproduce aggregate
  numerators;
- `representative_pair_measurements.csv` - pair-distance measurements for all
  five methods in the selected explanatory case;
- `representative_visibility_regimes.csv` - both branch states for every
  image in that case;
- `representative_case.json` - selection rule, scientific labels, acquisition
  range, and frozen natural-gap events;
- `external_reference_summary.json` - the separate EasyLEED-ViPErLEED
  contract and results;
- `source_provenance.json` - upstream paths and SHA-256 hashes.

The archive identifiers remain in provenance and the machine-readable case
record. Reader-facing figure labels use `Scan A`, `Scan B`, and pair identity.

## Rebuild and validation

Freeze the source package from the upstream artifacts:

```bash
python freeze_source_data.py
```

Build and validate the figure:

```bash
python build_figure.py
python build_figure.py --validate-only
```

Run the contract tests:

```bash
python test_freeze_source_data.py
python test_build_figure.py
```
