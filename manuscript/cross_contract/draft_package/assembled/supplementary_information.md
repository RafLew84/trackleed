---
title: "Supplementary Information: Contract validation and bounded sensitivity"
subtitle: "Scientific draft - not submission-ready"
lang: "en-US"
link-citations: true
---

# S1. Scope and evidence boundary

This Supplementary Information supports the focused cross-contract manuscript
by exposing the validation steps that are necessary to interpret its principal
claims. It does not create a second results narrative. The main manuscript
already reports the fixed opportunity counts, the four joint observability
states, the contract-specific gaps, and the three interpretations of accepted
out-of-branch continuation. Repeating those complete results here would add
length without adding verification. The narrower purpose is to document how
the native whole-pattern endpoint was frozen, how the restricted pilot was
used before the principal outcomes were available, and how the limited
sensitivity grid constrains the interpretation of direct-event disagreement.

The evidence hierarchy is unchanged. Principal claims use two existing energy
scans, 35 and 46 retained source frames, four mapped branches, and 324 fixed
branch-frame opportunities per method. The local side is a proposal-to-region
to Gaussian-refinement path. The external side is the indexed whole-pattern
ViPErLEED path. Their endpoints are operationally distinct. Neither endpoint
is physical-visibility, absolute-position, or LEED spot identity ground truth,
and the restricted pilot cannot provide such ground truth. The restricted pilot
is not an independent experimental replicate and not an absolute accuracy
reference. Its observations calibrated a positional tolerance and checked a
transferred registration; they were not added to the principal denominators.

All displays below are explicitly **Supporting validation** items. They are not
uncited supplementary claims standing outside the approved XC1-XC7 boundary.
Figure S1 and Table S2 document the restricted calibration pilot. Figure S2
documents the complete predeclared sensitivity logic through a compact view of
the tolerance-dependent classifications. Table S1 records the endpoint and
denominator protections required to read both. Detailed records stay in the
manifested repository packages identified in Section S5.

This separation also defines what the supplement does not contain. It contains
no full image sequence, per-frame branch table, per-opportunity cross-contract
matrix, runtime inventory, hash listing, or executed macro. Those materials are
essential for reconstruction but unsuitable as reader-facing displays. Their
absence from this document is an editorial constraint, not a loss of evidence:
the frozen manifests preserve the exact files, while this supplement preserves
the limited scientific explanation needed to audit the claims.

# S2. Frozen measurement contracts

The local endpoint is **Gaussian-contract measurability**. For each method and
predeclared branch-frame opportunity, a native proposal locates a common image
region. The same two-dimensional Gaussian estimator is then applied. The state
is positive only when the estimator returns a finite Gaussian-refined spot
position and every frozen local acceptance gate passes. Branch consistency is
evaluated after this endpoint and is not folded into measurability. A rejected
fit remains a retained negative state; it is not relabeled as deliberate
tracker abstention or physical disappearance.

The external endpoint is **ViPEr-native direct detectability**. ViPErLEED uses
an indexed whole-pattern registration, the frozen significance threshold, and
raw native position deviations. A mapped observation is direct only when
significance is at least 2.5 and both raw deviations are finite. A smoothed,
extrapolated, or final native coordinate alone cannot make the state positive.
The principal package therefore retains 305 direct and 19 non-direct outcomes
among 324 mapped branch opportunities even though a final native position is
available for every opportunity.

These rules were frozen before cross-contract outcome computation. The same
pilot pattern, final pilot screen fit, annular mask, native tracking parameters,
and source-frame sequences were used without per-frame correction, selective
beam deletion, or tuning after outcomes became visible. Each principal scan
received one unchanged transferred initialization. ViPErLEED remained outside
the shared Gaussian estimator: it is an external measurement contract, not a
sixth proposal method in the local comparison.

**Table S1. Supporting validation.** Frozen measurement endpoints and denominator protection. The compact three-column form is intended for an 88 mm single-column layout.

| Measurement path | Qualifying endpoint | Denominator and retained missingness |
| --- | --- | --- |
| Local proposal-to-Gaussian | Finite Gaussian-refined spot position passing every frozen local gate | 324 opportunities per method; rejection remains `G=0`, and branch consistency is separate |
| Native indexed whole-pattern | Significance at least 2.5 with finite raw x and y deviations | 324 mapped opportunities; subthreshold, non-finite, smoothed-only, and edge observations remain `V=0` |

The joint four-state matrix is formed only after the two endpoints are assigned.
It retains both-contract, Gaussian-only, ViPErLEED-only, and neither-contract
states for every method opportunity. The 1,620 method rows are five strata over
the same acquisitions and native opportunities, not independent experimental
replicates. This is why method rows are reported separately over 324 rather
than pooled into one apparently larger sample.

Outcome-specific denominators remain distinct. Branch-consistent measurement
coverage uses 221 frozen physics-baseline-visible opportunities per method.
The all-frame out-of-branch fraction conditions on the method's 206-272
accepted Gaussian fits. Interpretation conditions on the 124 accepted
out-of-branch events. Sensitivity comprises all nine combinations of three
native significance thresholds and three positional tolerances for all five
method strata, giving 45 method-variant rows. A missing state or Unresolved
cross-contract outcome stays in the relevant denominator; no conditional
surviving set replaces a fixed opportunity inventory.

Gap construction preserves the same contract boundary. A
**Gaussian-contract gap** is a maximal contiguous run of local non-measurability
for one method, scan, and branch with Gaussian-contract measurability
immediately before and after the run. A **ViPEr-native visibility gap** applies
the corresponding bounded-run rule to one indexed beam and requires direct
native detections at both boundaries. Edge runs and isolated non-direct states
without both boundaries remain in the opportunity matrix but are not promoted
to bounded gaps. Consequently, a gap count describes absence under one named
endpoint, not physical disappearance or an estimator-independent I(V) visibility gap.

Comparison components are constructed only within method, scan, and branch.
Closed gap intervals from both contracts are connected when they overlap; a
component is then Gaussian-only, ViPErLEED-only, or shared according to the
intervals it contains. Exact boundary agreement is stricter than overlap and
requires one gap from each contract with identical first and last gap frames.
The first observation after a component is an exit diagnostic. Even when both
contracts observe there, the record does not prove recovery of physical LEED
spot identity. These distinctions explain why the full gap-event and component
tables stay in the repository while their operational rules remain in the
reader-facing supplement.

# S3. Restricted pilot calibration evidence

The pilot used 15 existing high-energy frames and four preregistered mapped
branches, producing 60 branch-frame records. It initialized an 18-beam Ag(111)
pattern at 187.9 eV. Sixteen registration spots were directly available at that
reference frame; two pattern beams were unobserved. Among tested polynomial
screen models, the frozen cubic registration had 2.589 px root-mean-square,
2.249 px median, and 4.824 px maximum residual over the detected registration
spots. All detected spots remained inside the 8.33 px search radius.

This registration diagnostic asks whether the indexed pattern can be placed
consistently enough to run the native pathway. It is not a framewise
ground-truth error estimate for the four mapped branches. The residuals are
registration-to-detection discrepancies under one model at one reference
energy, not absolute spot-position errors across the scan. The final screen fit
was transferred unchanged to the two principal scans before their native
tracking outcomes were inspected.

The pilot contained 59 ViPEr-native directly detectable records under the 2.5
threshold and retained one non-direct record. A total of 38 records also had an
accepted Gaussian-refined position and therefore supported a raw-native to
Gaussian displacement. Calibration used a stricter subset: both endpoints had
to be available and native significance had to be at least 20. This produced
22 predeclared high-signal records. Their largest raw-native to Gaussian
distance was 0.4549 px. The frozen rule doubled that maximum and rounded upward
to the next 1.0 px increment, yielding the nominal 1.0 px cross-contract
positional tolerance.

![Figure S1. Supporting validation. Restricted-pilot high-signal calibration distances between raw native and accepted Gaussian-refined positions. All 22 predeclared calibration records remain at or below the observed maximum of 0.455 px and below the frozen 1.0 px cross-contract tolerance. This is calibration-only evidence, not an independent experimental replicate or an absolute accuracy reference.](manuscript/cross_contract/supplement/output/figure_s1_pilot_calibration.png){#figure-s1 width=88mm}

The distance is a disagreement between two operational endpoints after the
declared alignment. Its small value in the calibration subset supports a
conservative tolerance for later cross-contract interpretation, but does not
validate either endpoint against an unobserved true position. The subset was
not selected from the principal results, and its 22 rows are not added to any
principal-scan method denominator.

```{=openxml}
<w:p><w:r><w:br w:type="column"/></w:r></w:p>
```

**Table S2. Supporting validation.** Restricted-pilot validation evidence and its permitted evidential role. The compact three-column form is intended for an 88 mm single-column layout.

| Pilot diagnostic | Frozen value | Evidential role and limit |
| --- | --- | --- |
| Pilot opportunity inventory | 15 frames x 4 branches = 60 | Calibration-only stratum; not an independent experimental replicate |
| Native direct endpoint | 59/60 | Contract check; not estimator-independent physical visibility |
| Cross-contract displacement records | 38/60 | Endpoint comparison where both states qualify; not absolute accuracy |
| High-signal calibration subset | 22 records at significance >=20 | Predeclared tolerance input; not a principal result subset |
| Maximum calibration distance | 0.4549 px | Observed operational disagreement, not ground-truth error |
| Frozen positional tolerance | 1.0 px | Twice the maximum rounded upward to 1.0 px; used without optimization |

The pilot also retained 22 Gaussian rejections. Twenty-one occurred where the
native significance exceeded 2.5 and raw deviations were finite. This contrast
is consistent with the endpoints testing different evidence, but it is not a
miniature replication of the principal observability result. The pilot dataset,
energy range, and calibration use were fixed for method preparation, so its
counts remain supporting diagnostics only.

# S4. Limited predeclared sensitivity

The nominal analysis used native significance 2.5 and positional tolerance
1.0 px. The predeclared sensitivity grid crossed significance thresholds 2.0,
2.5, and 3.0 with tolerances 0.5, 1.0, and 2.0 px. Every combination was
reported for every method. No variant was selected and no threshold was tuned
to improve a method's apparent result.

Across these 45 method-variant rows, changing the native significance threshold
did not change the four-state measurability totals, Gaussian-contract gap
counts, ViPEr-native visibility gap counts, comparison components, fixed
branch-consistent coverage, or accepted out-of-branch denominators. This
invariance is specific to the tested 2.0-3.0 range and the frozen observations.
It does not establish threshold invariance outside that limited grid.

Positional tolerance changes only the interpretation of direct accepted
out-of-branch events. At 0.5 px, a stricter agreement requirement assigns more
direct events to **Cross-contract target inconsistency**. At 2.0 px, more direct
events become branch-model disagreement. Events without a required direct
native observation remain **Unresolved cross-contract outcomes** at every
tolerance because changing a distance boundary cannot supply missing direct
evidence.

![Figure S2. Supporting validation. Predeclared positional-tolerance sensitivity for accepted out-of-branch events at the nominal native significance threshold. Increasing tolerance reallocates direct events from Cross-contract target inconsistency to branch-model disagreement; Unresolved cross-contract outcome counts remain fixed. The complete 45-row grid is retained in the manifested outcome package.](manuscript/cross_contract/supplement/output/figure_s2_sensitivity.png){#figure-s2 width=88mm}

EasyLEED is invariant at 4 branch-model disagreements, 0 Cross-contract target
inconsistencies, and 0 Unresolved cross-contract outcomes. LocoTrack and DAM4SAM
move from 17/11 to 26/2 direct classifications as tolerance changes from 0.5 to
2.0 px, with 8 Unresolved cross-contract outcomes. OSTrack moves from 15/11 to
24/2, also with 8 Unresolved cross-contract outcomes. Lucas-Kanade moves from
7/6 to 13/0, with one Unresolved cross-contract outcome. Each triplet continues
to sum to that method's fixed accepted
out-of-branch event count: 4, 36, 34, 36, or 14.

These envelopes bound the strongest interpretation. The nominal 16
Cross-contract target inconsistencies are not a tolerance-free fact, although
the existence of direct contract-level disagreement within the predeclared
range remains visible. Likewise, movement into branch-model disagreement does
not prove that the local continuation has the correct physical identity. The
grid is a bounded robustness check, not parameter optimization, a complete
capture-basin map, or evidence for a universal tracker ranking.

Each sensitivity row is rebuilt from the same 324 opportunities per method.
Changing native significance can alter only the direct native state and the
events that depend on it; changing positional tolerance can alter only the two
direct-event interpretation classes. Neither parameter changes the stored
Gaussian proposal, region, fit, or acceptance result. The validator therefore
checks the fixed opportunity count, reference-visible count, accepted-fit
count, and accepted out-of-branch event count for every variant before
accepting the compact display data. It also verifies that all three thresholds,
all three tolerances, and all five method strata are present rather than
accepting a selected envelope endpoint.

Figure S2 uses the nominal significance threshold only to avoid drawing three
identical sets of lines. Its source contains all five methods at all three
tolerances, while the authenticated upstream table retains all 45 rows. This
display reduction is therefore lossless for the tolerance pattern shown, but
is not a replacement for the complete sensitivity artifact. The unchanged
Unresolved cross-contract outcome count remains visible in the source data and
caption even though it does not require a third plotted panel.

# S5. Repository evidence boundary

The reader-facing supplement ends at the level required to audit the claims.
The complete sensitivity grid remains in
`manuscript/cross_contract/outcomes/sensitivity_results.csv`; the full
branch-frame matrix and event tables remain beside it under the authenticated
outcome manifest. The 60 pilot records, raw and smoothed native tables,
registration evidence, executed macro, serialized parameters, and input hashes
remain in `manuscript/cross_contract/viperleed_pilot/`. Principal native tables,
both scan input manifests, executed macros, checkpoints, and the lossless
four-branch reduction remain in
`manuscript/cross_contract/viperleed_principal/`.

Those repository artifacts preserve failed and missing outcomes. A record is
not removed because Gaussian-contract measurability, ViPEr-native direct
detectability, or direct cross-contract interpretation is unavailable. Full
image sequences and per-frame records are consequently evidence assets rather
than supplementary displays. Hash inventories and runtime details authenticate
the reconstruction but are not scientific prose.

The compact source-data files packaged with this supplement contain only the
22 calibration points used by Figure S1, the 15 nominal-threshold method and
tolerance rows used by Figure S2, and the concise rows printed in Tables S1 and
S2. They are derived views with explicit upstream bindings, not alternate
analysis inputs. Validation rebuilds each view and both figure formats from the
frozen repository evidence and rejects byte drift, unexpected nested content,
or a display outside its declared physical dimensions.

This boundary keeps the Supplementary Information short enough for a
two-column production workflow while preserving auditability. The two figures
and two tables fit the declared 88 mm single-column width and use at least 7 pt
type in their generated assets. The canonical editable and rendered documents now include these displays.
Page-level render acceptance is complete. The outstanding Authorized deferred
publication inputs still prevent this scientific draft from being
`submission-ready`.
