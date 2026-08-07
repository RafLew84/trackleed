# Native ViPErLEED pilot contract

Status: frozen calibration contract

This package converts the existing 15-frame `26a24_ScanE` pilot from a
common-Gaussian compatibility check into a reproducible native ViPErLEED
reference contract. Native significance and raw position deviations are
evaluated directly; ViPErLEED coordinates are not passed through the TrackLEED
Gaussian estimator to decide native detectability.

The normative configuration is [`contract.json`](contract.json). The frozen
source copies are under [`frozen/`](frozen/), the normalized four-branch result
is `pilot_records.csv`, and `pilot_summary.json`, `manifest.json`, and
`validation_report.json` provide the deterministic outcome and integrity chain.

## Frozen native configuration

- ViPErLEED ImageJ plugin `1.03`, commit
  `2ed1044b72bb43a936f9456cdb8b9846e0d23e96`;
- 18-beam Ag(111) pattern initialized at `187.9 eV` with a cubic screen model;
- annular `1000 x 1000 px` mask centered at
  `(491.000307792, 472.011220864) px`, inner radius `125 px`, outer radius
  `470 px`;
- no dark, flat-field, `I0`, or `I00` correction supplied to the plugin;
- concentric-circle integration/background geometry, integration radius
  `8.33 -> 8.28 px` across the pilot;
- noise rejection `2.5`, unseen-beam reset `10 eV`, position smoothing
  `30 eV`, minimum beam range `0 eV`, and bright-neighbor background
  subtraction disabled;
- no per-frame correction, beam deletion, or manual recovery after tracking.

With a zero minimum beam range, ViPErLEED may retain inferred or extrapolated
positions below threshold. Therefore a **ViPEr-native direct detectability**
record requires both `significance >= 2.5` and finite `dx_raw` and `dy_raw`.
Final or smoothed coordinates alone never qualify.

## Frozen branch map

| TrackLEED branch | ViPErLEED beam |
| --- | --- |
| `P1_B1` | `(-1, 0)` |
| `P1_B2` | `(1, 0)` |
| `P2_B1` | `(1, -1)` |
| `P2_B2` | `(-1, 1)` |

## Pilot finding and calibration

The package retains 60 branch-frame opportunities. The old common Gaussian
contract rejects 22; 21/22 have native significance above `2.5`, and all 21
also have finite raw deviations. The remaining rejection has significance zero
and no finite raw position. Thus 59/60 records meet **ViPEr-native direct
detectability**.

The positional tolerance uses a predeclared high-signal subset only:

- Gaussian-contract measurable;
- ViPEr-native direct detectability;
- native significance at least `20`;
- no principal-scan record or result.

The 22 eligible records cover all four branches. Their maximum distance between
the reconstructed raw native position and accepted Gaussian-refined position is
`0.4548687654 px`. The frozen rule rounds `2 x maximum` upward to the next
whole pixel, producing a cross-contract tolerance of `1.0 px`.

For every jointly measurable record, `pilot_records.csv` reports Cartesian,
radial, and tangential displacement. The radial axis runs from the frozen
reciprocal-space origin to the native screen-model position. Image `y` is
converted from downward to upward before projection, so positive tangential
displacement is counter-clockwise in the physical plane.

## Evidence boundary

This pilot calibrates the native contract, beam mapping, and positional
tolerance. It is not reused as an independent confirmation dataset, is not
absolute position or spot-identity ground truth, and contains no principal-scan
outcome used by later classification.

## Validate

From the repository root:

```text
pytest trackleed/tests/test_viperleed_pilot_contract.py -q
python scripts/freeze_viperleed_pilot_contract.py --validate-only
```

The freeze mode is build-once and refuses to overwrite an existing frozen
package.
