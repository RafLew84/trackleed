# Native ViPErLEED principal scans

Status: frozen principal-scan inference

This package preserves the native whole-pattern ViPErLEED runs for the two
principal Ag(111) scans. The acquisitions already existed; the native tracking
results are new inference. The package therefore requires a controlled
scientific-core rebaseline after the cross-contract analysis and claim-boundary
approval.

The normative pre-result configuration is `contract.json`. The exact native
exports are under `frozen/`, their immutable run inventory is
`run_authentication.json`, and `manifest.json` closes the integrity chain.
`four_branch_records.csv` is a lossless reduction to the four predeclared
branches; it does not replace the retained 18-beam native tables.

## Execution boundary

Both scans used ImageJ 1.54u and ViPErLEED 1.03 with the frozen pilot tracker
parameters, pattern, annular mask, and final native pilot screen fit. Indices
were initialized once per scan at 188.5 eV. No per-frame correction, manual
recovery decision, selective beam deletion, or post-result parameter tuning was
performed.

The first `scan_a` launch stopped before tracking because `LEEDsetIndices`
rejected the older pre-pilot registration model with “array in LEEDsetIndices
does not fit”. Before any principal tracking result existed, a one-time
initialization amendment replaced that model with the final screen fit written
by the frozen native pilot log. The amendment, its source hash, and the
regenerated macros were committed before the successful runs; no tracking
parameter changed.

## Frozen outcome

The source inventory retains 81 distinct observations keyed by scan, source
frame, and energy: 35 for `scan_a` and 46 for `scan_b`. The final two frames of
each scan remain distinct despite their duplicate energies (220.5 eV and
220.4 eV respectively).

All 13 native tables per scan preserve 18 beams across every frame, for 1458
beam-frame records. The four preregistered branches contain 324 opportunities:

| Scan | Opportunities | Direct | Retained non-direct | Final positions |
| --- | ---: | ---: | ---: | ---: |
| `scan_a` | 140 | 129 | 11 | 140 |
| `scan_b` | 184 | 176 | 8 | 184 |
| Total | 324 | 305 | 19 | 324 |

ViPEr-native direct detectability requires significance at least 2.5 and
finite raw x/y deviations. A final or smoothed position alone does not count
as direct. Every subthreshold, inferred, or missing native observation remains
in `four_branch_records.csv` and in the denominator.

## Validate

From the repository root:

```text
python scripts/freeze_principal_viperleed_results.py --validate-only
pytest trackleed/tests/test_viperleed_principal_contract.py -q
```

Freeze mode is build-once and refuses to overwrite the authenticated package.
