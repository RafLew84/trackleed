# Principal-scan cross-contract outcomes

Status: frozen denominator-aware analysis

This package joins the immutable five-method Gaussian central panel to the
native ViPErLEED principal-scan package at the predeclared scan, source-frame,
energy, and branch seam. It retains all 324 opportunities per method and does
not treat frames, branches, or the two contracts on one acquisition as
independent experimental replicates.

The normative rules were committed in `analysis_contract.json` before the new
cross-contract counts were computed. `branch_frame_matrix.csv` contains all
1620 method-opportunity rows and the four operational outcomes: both contracts
observe, Gaussian only, ViPErLEED only, and neither contract observes.

## Nominal outcomes

| Method | Both | Gaussian only | ViPErLEED only | Neither |
| --- | ---: | ---: | ---: | ---: |
| EasyLEED | 222 | 0 | 83 | 19 |
| LocoTrack | 262 | 10 | 43 | 9 |
| OSTrack | 259 | 10 | 46 | 9 |
| DAM4SAM | 262 | 10 | 43 | 9 |
| Lucas-Kanade | 205 | 1 | 100 | 18 |

`method_summary.csv` deliberately separates two denominators:

- **branch-consistent measurement coverage** uses the fixed 221
  physics-baseline-visible opportunities per method;
- **all-frame conditional out-of-branch fraction** uses all accepted Gaussian
  fits for that method (206-272), not the fixed reference-visible set.

There are 124 accepted out-of-branch events. At the frozen 1.0 px positional
tolerance, `out_of_branch_events.csv` classifies 83 as branch-model
disagreement, 16 as cross-contract target inconsistency, and 25 as unresolved
because no direct native observation is available. These labels express
cross-contract evidence, not ground-truth identity or absolute accuracy.

## Contract-specific gaps

`gaussian_gap_events.csv` contains 114 maximal bounded runs without
Gaussian-contract measurability across the five methods. The native contract
contains two bounded gaps, both for `scan_a/P2_B2` at source frames 4 and 8.
The remaining native non-direct observations occur at unbounded edges or runs
and are retained in the matrix without being relabeled as bounded gaps.

`gap_comparisons.csv` contains 121 overlap-connected components: 111 Gaussian
only, 7 ViPErLEED only, and 3 shared. All three shared components have exact
boundary agreement. Both contracts observe at the component exit for 118/121
components; the three exceptions are explicit rather than dropped.

## Predeclared sensitivity

`sensitivity_results.csv` reports all 45 method-variant rows from the full
Cartesian grid of native significance thresholds 2.0/2.5/3.0 and positional
tolerances 0.5/1.0/2.0 px. No variant is selected and no method ranking is
computed. Measurability counts are stable throughout this limited grid;
positional-tolerance changes only reallocate some direct out-of-branch events
between branch-model disagreement and cross-contract target inconsistency.

## Validate

From the repository root:

```text
python scripts/build_cross_contract_outcomes.py --validate-only
pytest trackleed/tests/test_cross_contract_outcomes.py -q
```

Build mode is staged, validated before publication, and refuses to replace an
existing frozen analysis.
