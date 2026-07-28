# Evaluating Spot-Tracking Strategies for Quantitative LEED Analysis

Draft status: structured outline, not submission-ready prose.

## Authors

TODO[AUTHOR]: author list, affiliations, corresponding author, ORCID identifiers.

## Abstract

Structure the abstract as one continuous paragraph:

1. **Context:** quantitative LEED analysis requires the same diffraction spot to
   remain associated with the measurement ROI while the pattern moves and spot
   visibility changes.
2. **Problem:** native point, bounding-box, mask, optical-flow, and domain-specific
   trackers produce heterogeneous outputs, so their proposal coordinates are not
   directly comparable as final physical measurements.
3. **Approach:** compare five tracking strategies using one downstream `48 x 48`
   ROI and one two-dimensional Gaussian estimator; evaluate a positive control,
   known translations, independent energy scans with natural I(V) visibility gaps,
   and a controlled SAM2-family memory experiment.
4. **Main results:** known-translation median full-chain errors span
   `0.0043-0.0265 px` with `92.5-100%` Gaussian-measurement coverage; on the
   independent ScanE series, coverage spans `188/221-217/221` and gap recovery
   `17/24-23/24`; memory-aware SAM2 variants improve continuity but also increase
   geometry-inconsistent accepted fits.
5. **Conclusion:** once the intended spot remains inside the Gaussian estimator's
   capture region, proposal differences are strongly suppressed; the practically
   discriminating outcomes are measurement coverage, post-gap identity recovery,
   and physically inconsistent continuation.

TODO[TEXT]: write the final abstract only after the main figures and uncertainty
reporting are frozen.

## Keywords

Low-energy electron diffraction; spot tracking; quantitative image analysis;
Gaussian localization; visibility gaps; measurement robustness.

## 1. Introduction

### 1.1 Quantitative LEED requires persistent spot identity

- Explain why tracking is upstream of quantitative position, distance, width, and
  intensity observables.
- Distinguish a smooth trajectory from preservation of physical spot identity.
- Introduce motion caused by experimental evolution and energy-dependent radial
  contraction.
- Introduce I(V) visibility gaps as intensity-driven loss and reappearance, not
  generic video occlusion.

TODO[CITATION]: quantitative LEED analysis and conventional spot localization.

### 1.2 Existing LEED tracking and the remaining evaluation problem

- Position EasyLEED as the physics-informed Kalman tracking baseline.
- Position ViPErLEED as an external whole-pattern LEED reference.
- State explicitly that automated LEED tracking, energy-dependent motion models,
  Gaussian localization, and propagation through invisible ranges have precedent.
- Identify the gap addressed here: a common downstream position estimator and an
  evidence ladder that separates conditional localization quality from coverage,
  recovery, and false continuity.

TODO[CITATION]: EasyLEED, ViPErLEED, and the upstream publications for all general
tracking methods.

### 1.3 Study objective and contributions

Working thesis:

> When a tracking proposal keeps the correct LEED spot inside the Gaussian
> estimator's capture region, the common fit suppresses method-specific proposal
> differences. During motion and visibility loss, methods differ chiefly in
> measurement coverage, post-gap recovery, and physically inconsistent
> continuation, which conditional position residuals alone do not reveal.

Contributions:

1. A common proposal-to-ROI-to-Gaussian measurement contract for heterogeneous
   point, bounding-box, mask, optical-flow, and LEED-domain trackers.
2. A controlled equivariance test of the complete measurement chain under known
   subpixel image translations.
3. A five-strategy comparison on independent ScanE series using common
   denominators for visible measurement opportunities and natural I(V) gaps.
4. A controlled SAM2/SAMURAI/DAM4SAM analysis that isolates the effect of memory
   strategy under byte-identical checkpoint weights.
5. Propagation of tracker proposals to the pairwise physical endpoint
   `Delta R/R = Delta D/D` in a 701-frame constant-energy Cu(111) sequence.

## 2. Data and physical measurement model

### 2.1 Data regimes

Describe four evidence roles without combining them into a single ranking:

1. **Cu(111) positive control:** 701 frames at 56 eV, varying temperature.
2. **Controlled equivariance:** four frozen Cu(111) source frames, six known
   translations, four spots.
3. **Energy-scan stress test:** `26a21_ScanE` and `26a21b_ScanE`, four frozen
   diametric pairs.
4. **Controlled memory analysis:** `26a24_ScanE`, two frozen pairs.

The limited EasyLEED/ViPErLEED run is an external reference, not a row in the
common five-method matrix.

TODO[TEXT]: add acquisition conditions available in the frozen inventories,
including detector geometry and energy/temperature sampling, without inferring
missing instrument metadata.

### 2.2 Tracking proposal and measurement ROI

Define the native representation for each method:

- LocoTrack: propagated point;
- OSTrack: center of the propagated bounding box;
- DAM4SAM: centroid of the propagated mask;
- Lucas-Kanade: accepted optical-flow point;
- EasyLEED: Kalman position.

Each native output is converted to the original `1000 x 1000` image coordinates.
The proposal defines a `48 x 48` ROI. It is not itself the reported spot position.

### 2.3 Common Gaussian-refined position

Define the same elliptical two-dimensional Gaussian fit and acceptance gates for
all methods. Report the fitted global center as

$$
\mathbf{p}_{i,t} = (x_{i,t}, y_{i,t}).
$$

State the rejection rules, missing-value representation, and absence of
interpolation. Keep tracker visibility/confidence separate from Gaussian amplitude
or integrated volume.

### 2.4 Pairwise position endpoints

For a frozen pair of spots `i,j`, define

$$
D_{ij}(t) = \left\|\mathbf{p}_{i,t} - \mathbf{p}_{j,t}\right\|_2.
$$

At constant energy and geometry, use the pairwise relative change

$$
\frac{\Delta R}{R}
=
\frac{\Delta D}{D}
=
\frac{D_{ij}(t)}{D_{ij,\mathrm{ref}}} - 1.
$$

For ScanE, compare the measured contraction with the reference law

$$
D(E) \propto \frac{1}{\sqrt{E}},
$$

while retaining the pair midpoint

$$
\mathbf{m}_{ij}(E) =
\frac{\mathbf{p}_{i}(E)+\mathbf{p}_{j}(E)}{2}
$$

as a separate apparatus/projection diagnostic.

## 3. Tracking strategies and comparison contract

### 3.1 Main five-strategy panel

Summarize LocoTrack, OSTrack, DAM4SAM, Lucas-Kanade, and EasyLEED. For each method
report:

- upstream source and version/revision;
- checkpoint and configuration hash where applicable;
- initialization representation;
- native output reduced to the common proposal coordinate;
- direction of propagation and reset policy;
- native diagnostic signal retained but excluded from the physical observable.

### 3.2 Controlled SAM2-family panel

Compare:

- SAM2: vanilla propagation memory;
- SAMURAI: motion-aware memory;
- DAM4SAM: distractor-aware memory.

Freeze byte-identical SAM2.1 base-plus weights, the same `48 x 48` bounding-box
prompt, initialization, image sequence, and downstream Gaussian measurement.
Interpret this panel as a memory-mechanism experiment, not a general ranking of
backbone quality.

### 3.3 Domain-specific references

- EasyLEED participates in the main comparison as the LEED-domain tracking
  baseline.
- ViPErLEED is evaluated on the frozen high-energy ScanE subset as an external
  whole-pattern reference.
- Explain the different initialization and output contracts before comparing
  their downstream Gaussian centers.

## 4. Validation protocol

### 4.1 Reporting principles

- Keep coverage, recovery, geometry inconsistency, and conditional residuals
  separate.
- Retain every opportunity in the denominator, including unavailable proposals
  and rejected Gaussian fits.
- Do not generate a composite score or an absolute-accuracy ranking.
- Do not interpret tracker confidence as LEED intensity.
- Do not treat frames from one trajectory as statistically independent samples.

TODO[DECISION]: select uncertainty intervals that respect clustering by series,
pair, and trial; do not bootstrap individual frames as independent observations.

### 4.2 Controlled translations

Use source frames `0`, `372`, `378`, and `700`. Apply the closed six-step path

$$
(0,0),\ (-14.25,-9.5),\ (-8.5,11.25),\
(13.25,-6.5),\ (10.5,12.25),\ (0,0)
$$

in original-image pixels. Reinitialize only between the four trials, never within
a trial.

For proposal or Gaussian center `q_t`, calculate displacement equivariance error

$$
\epsilon_t =
\left\|(\mathbf{q}_t-\mathbf{q}_0) -
(\Delta x_t,\Delta y_t)\right\|_2.
$$

Report five non-seed steps for each of four trials and four spots:
`80` spot opportunities per tracker. Report `40` pair opportunities. Include an
oracle ROI control that quantifies interpolation plus Gaussian-estimator error and
is not counted as a tracker.

### 4.3 Cu(111) positive control

- Run each method once through frames `0-700` with four common seeds.
- Prohibit manual correction and reinitialization.
- Evaluate pairs `1-6` and `3-5` on all 701 frames.
- Use pair `2-4` only as a missing-measurement diagnostic.
- Use frames `0-9` for the pairwise reference distance, with the frozen validity
  requirement.

### 4.4 Independent ScanE stress test

- Use two independently acquired series and two disjoint frozen pairs per series.
- Start from one seed per spot at the frozen reference energy.
- Propagate separately toward lower and higher energy without post-gap
  reinitialization.
- Use the common denominators of `221` baseline-visible spot opportunities and
  `24` evaluable natural I(V) gaps per method.

### 4.5 Visibility regimes and recovery

Separate:

1. outside-screen or apparatus-mask absence;
2. below diffraction-beam onset;
3. true I(V) visibility gap.

Count post-gap identity recovery only for the third regime and only when the
returning Gaussian measurement lies on the frozen physics-consistent branch.
Report accepted but geometry-inconsistent fits as false continuity.

## 5. Results

### 5.1 Known translations isolate the measurement-chain behavior

Evidence anchors:

- LocoTrack, OSTrack, and DAM4SAM: `80/80` Gaussian measurements.
- Lucas-Kanade: `76/80`.
- EasyLEED: `74/80`.
- Median full-chain displacement error: `0.0043-0.0265 px`.
- Oracle ROI median: `0.0053 px`.
- Maximum absolute `Delta D/D` below `0.002` percentage points for LocoTrack,
  OSTrack, DAM4SAM, and Lucas-Kanade; `0.062943` percentage points for EasyLEED.

Interpretation: conditional error can approach the estimator floor after a valid
proposal, but coverage remains an independent outcome.

TODO[FIGURE]: Figure 2.

### 5.2 Cu(111) is a positive control, not a tracker ranking

Evidence anchors:

- `2804/2804` valid Gaussian measurements per method.
- `701/701` valid endpoints for both primary pairs.
- Native proposal separation reaches `16.759 px`, whereas the largest separation
  between Gaussian-refined centers is `0.0502 px`.
- Maximum between-method difference in `Delta R/R` is `0.001710` percentage
  points for pair `1-6` and `0.002237` for pair `3-5`.
- All five methods place the minimum of both pairs at frame `378`.

Report the plateau at frames `372-376`, the minimum at frame `378`, and final-frame
return. Preserve the sector-to-sector difference without interpreting two pairs as
crystallographic anisotropy.

TODO[FIGURE]: Figure 5.

### 5.3 Energy scans reveal coverage, recovery, and false continuity

| Method | Coverage | Gap recovery | Geometry-inconsistent accepted fits | Conditional median residual |
|---|---:|---:|---:|---:|
| LocoTrack | 217/221 | 23/24 | 36/272 | 1.837 px |
| OSTrack | 216/221 | 21/24 | 34/269 | 1.770 px |
| DAM4SAM | 216/221 | 23/24 | 36/272 | 1.788 px |
| Lucas-Kanade | 188/221 | 17/24 | 14/206 | 1.764 px |
| EasyLEED | 214/221 | 22/24 | 4/222 | 1.737 px |

Interpret the residual only on jointly geometry-consistent pair frames. Use the
Lucas-Kanade row to demonstrate why a similar conditional residual cannot
compensate for missing coverage.

TODO[FIGURE]: Figure 3.

### 5.4 Memory improves continuity but can preserve a wrong association

| Method | Coverage | Gap recovery | Geometry-inconsistent accepted fits |
|---|---:|---:|---:|
| SAM2 | 126/182 | 9/13 | 8/149 |
| SAMURAI | 181/182 | 13/13 | 36/236 |
| DAM4SAM | 179/182 | 13/13 | 36/233 |

Interpret this as a continuity-versus-false-persistence trade-off under identical
weights. Keep it supporting rather than primary evidence because it contains one
series and two pairs.

TODO[FIGURE]: Figure 4.

### 5.5 Domain-specific external reference

Report the frozen 15-frame, two-pair EasyLEED/ViPErLEED comparison:

- 60 proposals per method;
- 38/60 accepted common Gaussian fits per method;
- four frozen absence-to-recovery checks passed by both methods;
- median native proposal separation `0.626 px`;
- median common Gaussian-center separation `0.000986 px`.

Use this result to establish comparability of the downstream measurement contract,
not superiority of either domain-specific method.

## 6. Discussion

### 6.1 Tracker performance is mediated by the estimator capture region

Connect the controlled translation and Cu(111) results. Explain why heterogeneous
native proposals can converge after a common Gaussian fit without claiming that
the trackers are equally accurate in an absolute sense.

### 6.2 Coverage and identity failures cannot be replaced by conditional error

Use the independent ScanE results to distinguish:

- failure to return a proposal;
- rejected quantitative measurement;
- correct recovery after a true I(V) gap;
- accepted but physics-inconsistent continuation.

### 6.3 Memory has a measurable continuity trade-off

Discuss why motion-aware or distractor-aware memory can bridge gaps while also
maintaining an incorrect association. Limit the conclusion to the frozen
SAM2-family experiment.

### 6.4 Consequences for quantitative LEED observables

Explain how the measurement-chain behavior propagates to pairwise `Delta R/R`.
Keep Gaussian I(V) amplitude/volume separate from tracker visibility and avoid a
full structural LEED I(V) interpretation.

### 6.5 Limitations

- No external framewise ground truth for absolute spot identity or position.
- Controlled translations test equivariance, not experimental dynamics.
- Cu(111) is an easy positive-control regime and does not discriminate trackers.
- ScanE contains four primary pairs across two independent series.
- The SAM2-family experiment contains one series and two pairs.
- DAM4SAM uses `sam21pp-L` on Cu(111) and `sam21pp-B` in the independent ScanE
  comparison.
- ViPErLEED uses a different whole-pattern initialization contract.
- No full structural I(V) analysis, thermal-expansion coefficient, or material
  transformation is inferred.

## 7. Reproducibility and data availability

Describe:

- immutable raw-data inventory and hashes;
- native tracker payloads and frozen configurations;
- per-frame proposals, Gaussian measurements, fit diagnostics, and pair endpoints;
- deterministic artifact builders and validators;
- headless reconstruction from saved predictions without requiring the original
  inference GPU;
- planned dataset DOI and code release.

TODO[DECISION]: repository/archive location, license, DOI, and public release scope.

## 8. Conclusions

Draft as three concise statements:

1. A common Gaussian estimator strongly suppresses the endpoint's sensitivity to
   substantial proposal differences, conditional on a valid captured spot.
2. Measurement coverage, post-gap identity recovery, and false continuity are
   necessary alongside conditional localization residuals.
3. The appropriate evaluation target is the complete measurement chain and its
   physical endpoint, not a composite ranking of native tracker coordinates.

## CRediT authorship contribution statement

TODO[AUTHOR].

## Declaration of competing interest

TODO[AUTHOR].

## Acknowledgements

TODO[AUTHOR]: funding, instrument access, computational resources, collaborators.

## Data and code availability

TODO[DECISION]: final archive statement and persistent identifiers.
