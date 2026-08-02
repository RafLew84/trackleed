# Figure 5 - propagation to the Cu(111) physical endpoint

Status: generated and validated.

## Final artifacts

- [PNG](output/figure05_cu111_physical_endpoint.png)
- [PDF](output/pdf/figure05_cu111_physical_endpoint.pdf)
- [render manifest](manifest.json)
- [frozen source contract](contract.json)

## Scientific question

Do tracker-dependent proposal differences survive the common Gaussian
refinement sufficiently to change the pairwise relative spot-radius endpoint
in the Cu(111) thermal cycle?

The answer supported by this positive control is no at the observed scale.
Each of the five methods supplies 2,804/2,804 proposals and valid common fits,
and both primary pairs supply 701/701 valid endpoints per method. The maximum
between-method separation over the complete sequence is 0.001710 percentage
points for pair 1-6 and 0.002237 percentage points for pair 3-5. This agreement
does not establish an overall tracker ranking.

## Frozen scope

The primary comparison contains one 701-image Cu(111) thermal cycle acquired
at a fixed energy of 56 eV. LocoTrack, OSTrack, DAM4SAM, Lucas-Kanade, and
EasyLEED are initialized once on the first image for spots 1, 3, 5, and 6.
There is no correction or reinitialization. The primary endpoint is reported
separately for frozen pairs 1-6 and 3-5.

For each method and pair,

`Delta R/R = Delta D/D = D_ij(t) / D_ij,ref - 1`,

where `D_ij(t)` is the distance between the two common Gaussian-refined
centers. `D_ij,ref` is the median of valid pair distances in the first ten
images, with at least eight of ten required. No pattern center is used and
missing values are not interpolated or forward-filled.

Frame identifiers are retained in the source package only for reproducibility.
Reader-facing landmark labels use thermal-cycle phase and temperature:
reference at 10-16 °C, the predefined 800 °C plateau, the common
post-plateau minimum at 797 °C, and the final cooled state at 92 °C. The
post-plateau minimum is explicitly marked as outcome-selected; the other
landmarks are predefined.

## Final panels

- **a. Thermal cycle.** Temperature profile with the reference interval,
  800 °C plateau, post-plateau minimum, and final cooled state.
- **b. Pair 1-6.** Cross-method median `Delta R/R` trajectory with the
  min-max envelope and a magnified strip of each method minus the
  cross-method median.
- **c. Pair 3-5.** The same physical-trajectory and method-residual
  representation on a residual scale shared with panel b.
- **d. Endpoint spread.** Between-method ranges at the plateau,
  post-plateau minimum, and final cooled state. The complete-sequence maxima
  remain in the source package as supporting statistics but are not an
  additional displayed landmark.
- **e. Separate missing-fit diagnostic.** The 26-image context around pair
  2-4, showing 24 consecutive rejected fits despite a saved proposal being
  present. All rejected fits pass the `R²`, center, and width gates and fail
  only the common amplitude threshold of 0.05.

Panel e uses an earlier saved YOLO proposal trajectory. It is outside the
five-method run and must be visually labelled as a separate diagnostic.

## Caption

**Figure 5. Common Gaussian refinement preserves the Cu(111) pairwise
relative spot-radius endpoint across proposal strategies.** **a,** Temperature
profile of the 701-image constant-energy cycle with reference, 800 °C
plateau, post-plateau minimum, and final cooled state; frame identifiers remain
in source metadata. **b,c,** Pairwise \(\Delta R/R=\Delta D/D\) for pairs 1–6
and 3–5 after five proposal strategies enter the same estimator. Upper axes
show the cross-method median and range; lower axes magnify deviations from the
median in \(10^{-3}\) percentage points on a shared scale. Every method yields
701/701 endpoints per pair. Common image translation cancels in pair distance,
whereas non-rigid distortion does not. **d,** Between-method endpoint range at
the three predefined landmarks; maxima over the sequence are 0.001710 and
0.002237 percentage points. **e,** Separate saved-YOLO diagnostic for pair
2–4: proposals remain present throughout the displayed context while 24
consecutive Gaussian fits fail only the amplitude gate. This distinguishes
proposal from measurement availability; it does not show identity loss or
compare the five methods on that pair. The two primary pairs do not establish
crystallographic anisotropy, thermal expansion, phase transition, or a
structural mechanism.

## Frozen source package

- [source contract](contract.json)
- [five-method run contract](source_data/five_method_run_contract.json)
- [temperature profile](source_data/temperature_profile.csv)
- [cycle landmarks](source_data/cycle_landmarks.csv)
- [five-method endpoint trajectories](source_data/pair_endpoint_timeseries.csv)
- [endpoint summaries](source_data/endpoint_summaries.csv)
- [between-method spread](source_data/between_method_spread.csv)
- [separate pair 2-4 diagnostic](source_data/pair_2_4_missingness_diagnostic.csv)
- [source provenance](source_data/source_provenance.json)

Rebuild and validate:

```bash
python freeze_source_data.py
python freeze_source_data.py --validate-only
python build_figure.py
python build_figure.py --validate-only
python test_build_figure.py
python test_freeze_source_data.py
python test_source_contract.py
```
