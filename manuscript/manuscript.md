# Evaluating Spot-Tracking Strategies for Quantitative LEED Analysis

Draft status: working manuscript; Sections 2 and 4 are drafted, while the
remaining sections retain structured-outline placeholders.

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

We used four data regimes with distinct evidential roles rather than pooling
them into a single tracker ranking. The positive-control sequence comprised 701
Cu(111) diffraction images acquired at a constant electron energy of 56 eV
during a thermal cycle. The recorded temperature ranged from 10 to 800 °C; the
first image was acquired at 10 °C and the final image at 92 °C. Four spots
(identifiers 1, 3, 5, and 6) and two predefined diametric pairs (1–6 and
3–5) formed the common five-method comparison. This sequence tests whether
different proposal mechanisms lead to the same physical endpoint when all
methods maintain the intended spot within the measurement region. It is
therefore a positive control for agreement and not an absolute-accuracy
benchmark.

Controlled translation experiments used four images from the same sequence:
frames 0, 372, 378, and 700, acquired at 10, 800, 797, and 92 °C,
respectively. Each image was transformed by the same closed path of six
subpixel translations. The four spots and two pairs used in the full Cu(111)
sequence were retained. This regime provides a known image transformation and
therefore evaluates equivariance of the complete proposal-to-measurement chain
without requiring a manually annotated trajectory.

The principal stress test used two independently acquired energy scans.
`26a21_ScanE` contains 35 frames spanning 40.0–220.5 eV and
`26a21b_ScanE` contains 46 frames spanning 40.0–220.4 eV. In both series,
188.5 eV was selected as the reference energy (frame 27 and frame 36,
respectively). Two disjoint diametric pairs were detected and frozen in each
series before the compared trackers were evaluated. These data combine
energy-dependent radial motion with intensity-driven disappearance and
reappearance of diffraction spots.

A third energy scan, `26a24_ScanE`, was reserved for the controlled
SAM2-family analysis. It contains 69 frames spanning 45.0–220.2 eV, with
frame 54 at 187.9 eV used as the reference, and two frozen diametric pairs.
SAM2, SAMURAI, and DAM4SAM were compared on this series under identical
checkpoint weights and a common downstream measurement contract. A restricted
15-frame high-energy interval (frames 54–68, 187.9–220.2 eV) was also used
to place EasyLEED and ViPErLEED under the same downstream estimator. That
limited comparison is an external domain reference, not a row in the common
five-method matrix.

All numerical acquisition conditions stated here are taken from the frozen
image inventories. TODO[TEXT]: add microscope, detector, sample-preparation,
exposure, and heating-rate metadata when the corresponding experimental
records are available; these quantities are not inferred from filenames.

### 2.2 Tracking proposal and measurement ROI

The native output of a tracker was treated as a *tracking proposal*: a
coordinate used to place the downstream measurement region, not the reported
spot position. LocoTrack supplied a propagated point, OSTrack the center of its
propagated bounding box, DAM4SAM the centroid of its propagated mask,
Lucas–Kanade an accepted optical-flow point, and EasyLEED its Kalman-filtered
position. In the controlled memory experiment, the centroid of the propagated
mask was used for SAM2, SAMURAI, and DAM4SAM. Native visibility flags,
confidence scores, mask areas, and component counts were retained as tracker
diagnostics but were not interpreted as diffraction intensity.

Regardless of the inference resolution used by a method, its proposal was
mapped back to the original 1000 × 1000 image coordinate system. The proposal
coordinate was rounded to the nearest pixel and used to place a 48 × 48 pixel
region of interest (ROI) on the original-intensity image. ROIs that extended
beyond the image boundary were rejected rather than clipped or padded. Thus,
different methods could place different ROIs, but every method within an
experiment passed the selected raw-image region to the same position
estimator.

### 2.3 Common Gaussian-refined position

Within an ROI, the intensity was represented by a rotated elliptical Gaussian
with a constant background,

$$
I(x,y) = B + A\exp\left[
-\frac{1}{2}\left(
\frac{u^2}{\sigma_x^2}+\frac{v^2}{\sigma_y^2}
\right)\right],
$$

where

$$
\begin{pmatrix}u\\v\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}x-x_c\\y-y_c\end{pmatrix}.
$$

The fitted parameters were the amplitude \(A\), local center
\((x_c,y_c)\), principal widths \((\sigma_x,\sigma_y)\), axial angle
\(\theta\), and background \(B\). Nonlinear least squares was initialized from
the ROI intensity distribution. Adding the ROI origin to the local fitted
center gave the reported global position of spot \(i\) in frame \(t\),

$$
\mathbf{p}_{i,t} = (x_{i,t}, y_{i,t}).
$$

The estimator implementation was common to all methods, whereas acceptance
gates were frozen separately for the Cu(111) and ScanE regimes and then held
constant across methods within each regime. In the Cu(111) sequence and
controlled-translation experiment, a fit required finite diagnostics and
center uncertainties, \(R^2\geq0.85\), \(A\geq0.05\), local center
coordinates strictly between -2 and 50 pixels, and both fitted widths strictly
between 0.5 and 48 pixels. In the ScanE experiments, the gates were
\(R^2\geq0.85\), \(A\geq0.012\),
\(0.6\leq\sigma_x,\sigma_y\leq13\) pixels, and a distance of at most 15
pixels between the proposal and refined center. The ScanE solver used the same
Gaussian model; its initial center was the smoothed intensity maximum within a
16.32-pixel radius of the ROI center, and its fitted center was bounded to
±16.32 pixels from that ROI center along each coordinate.

If the tracker did not return a proposal, the ROI lay outside the image, the
fit failed, or any acceptance gate failed, the position was recorded as
missing. Rejected values were neither interpolated nor carried forward.
Gaussian amplitude and integrated volume were properties of the image fit and
remained distinct from any native tracker confidence or visibility score.

### 2.4 Pairwise position endpoints

For a frozen pair of spots \(i,j\), the primary geometrical observable was
their Euclidean separation,

$$
D_{ij}(t) = \left\|\mathbf{p}_{i,t} - \mathbf{p}_{j,t}\right\|_2.
$$

For an approximately diametric pair, its diffraction radius is
\(R=D_{ij}/2\). At constant beam energy and fixed projection geometry, the
relative radial change can therefore be evaluated without estimating a
framewise pattern center:

$$
\frac{\Delta R}{R}
=
\frac{\Delta D}{D}
=
\frac{D_{ij}(t)}{D_{ij,\mathrm{ref}}} - 1.
$$

For an energy scan, the expected radial contraction was defined relative to
the frozen reference energy \(E_{\mathrm{ref}}\):

$$
D_{ij,\mathrm{pred}}(E)
=
D_{ij,\mathrm{ref}}
\sqrt{\frac{E_{\mathrm{ref}}}{E}},
$$

which is the pairwise form of \(D(E)\propto1/\sqrt{E}\). The corresponding
distance residual was
\(\delta D_{ij}(E)=D_{ij}(E)-D_{ij,\mathrm{pred}}(E)\). Pair distance is
invariant to a common translation of both spots, but it does not eliminate
non-rigid image distortion, incorrect spot association, or sector-dependent
behavior.

The pair midpoint,

$$
\mathbf{m}_{ij}(E) =
\frac{\mathbf{p}_{i}(E)+\mathbf{p}_{j}(E)}{2},
$$

was consequently retained as a separate diagnostic of common image motion,
apparatus geometry, and projection changes. It was not substituted for the
pair distance and was not used to recenter the primary endpoint.

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

The validation separates four outcomes that answer different questions:
physics-consistent measurement coverage, recovery after natural I(V)
visibility gaps, accepted measurements that are inconsistent with the expected
branch geometry, and position or pair-distance residuals conditional on a
valid branch-consistent measurement. Every predefined opportunity remains in
its denominator when a tracker proposal is unavailable or a Gaussian fit is
rejected. Conditional residuals are reported only with their corresponding
coverage and are not used to replace failures with favorable surviving-frame
statistics.

All results are descriptive. Counts and rates retain their integer
denominators, and continuous errors are summarized by medians, 95th
percentiles, and maxima where specified in the frozen artifacts. We do not
construct a composite score or claim an absolute-accuracy ranking. Frames
within one propagated trajectory are repeated observations of the same
tracking event and are not treated as statistically independent replicates.
All per-frame outputs were calculated automatically and no individual
proposal annotations were created. For the 701-frame Cu(111) run, visual
review was limited to a frozen set of 24 combined overlays selected
independently of tracker output (12 temporal quantiles and 12
temperature-cycle anchors), rather than all 14,020 proposals. Visual review
was not used to correct a trajectory or change an outcome.
TODO[DECISION]: before submission, decide whether to add uncertainty intervals
at the trial, pair, or series level; framewise bootstrap intervals are excluded.

### 4.2 Controlled translations

For each of Cu(111) frames 0, 372, 378, and 700, the original image was
subjected to the closed translation path

$$
(0,0),\ (-14.25,-9.5),\ (-8.5,11.25),\
(13.25,-6.5),\ (10.5,12.25),\ (0,0)
$$

in original-image pixels. Transforms were generated with first-order
interpolation, a constant zero-valued boundary, and no prefiltering. The
same physical translation was applied independently on two frozen grids: the
1000 × 1000 floating-point raw image used for Gaussian measurement and the
256 × 256 tracker cache, with displacements scaled by 0.256 before conversion
of the transformed cache to unsigned 8-bit values. Thus, the tracker cache was
not generated by downsampling the already transformed raw image.

Each trial used the same four frozen seed centers. A seed was the
coordinate-wise median of the five available common Gaussian centers from the
completed Cu(111) comparison in that source frame. Point-based methods
received the corresponding seed coordinates, box- or mask-based methods
received a symmetric 48 × 48 box derived from the same center, and EasyLEED
used the same center with its frozen tracking-radius and Kalman contract.
Seeds were inputs to the experiment and were not treated as ground truth. No
correction or reinitialization was allowed within a six-frame trial.

For either a native proposal or a Gaussian-refined center
\(\mathbf{q}_t\), the displacement-equivariance error was

$$
\epsilon_t =
\left\|(\mathbf{q}_t-\mathbf{q}_0) -
(\Delta x_t,\Delta y_t)\right\|_2.
$$

Subtracting each method's own initial coordinate removes a constant
initialization offset and tests recovery of the commanded displacement. The
seed step was excluded from aggregation, leaving five evaluated steps for
four trials and four spots, or 80 spot opportunities per method. Applying the
same rule to two predefined pairs produced 40 pair opportunities per method.
For each pair we additionally measured
\(\left|D_{ij}(t)-D_{ij}(0)\right|\) and the corresponding change in
\(D_{ij}\) relative to its initial value.

An oracle-ROI control placed the measurement ROI at the commanded translated
seed without using a tracker. It estimates the combined interpolation and
Gaussian-estimator floor under the same image transformations and acceptance
gates. The oracle is a control, not a sixth tracking method.

### 4.3 Cu(111) positive control

Each of the five methods was initialized once on frame 0 with common seeds for
spots 1, 3, 5, and 6 and then propagated forward through frame 700. Manual
corrections and reinitialization were prohibited. Every proposal was converted
to the common 48 × 48 ROI and Gaussian measurement described in Section 2.
The primary endpoints were evaluated for pairs 1–6 and 3–5 on all 701
frames.

For each method and pair,
\(D_{ij,\mathrm{ref}}\) was the median of valid pair distances in frames
0–9, provided that at least 8 of the 10 distances were valid. If this
criterion was not met, the entire method–pair relative endpoint was defined
as unavailable. The thermal-cycle summaries were the median relative change
over the predefined 800 °C plateau (frames 372–376), the minimum and its
frame, and the final-frame value.

Pair 2–4 was not included in the five-method run because spots 2 and 4 were
outside its frozen four-spot contract. It was retained as a separate
missing-measurement diagnostic based on an earlier saved YOLO proposal
trajectory. In that diagnostic the proposal remained available while 24
consecutive Gaussian fits (frames 357–380) failed only the common amplitude
gate. These missing endpoints were not interpolated. The diagnostic
demonstrates that proposal availability and quantitative measurement
availability are distinct; it is not evidence that all five methods behaved
identically for pair 2–4.

### 4.4 Independent ScanE stress test

The independent ScanE comparison used two series and two disjoint frozen pairs
per series, for four cases in total. Candidate spots were detected
automatically in the 188.5 eV reference image, refined by the common Gaussian
fit, combined into approximately diametric pairs, and frozen before method
comparison. The two branches of each pair were tracked as independent
single-object trajectories. From the reference frame, propagation was
performed separately toward lower and higher energies, with one seed per spot
and no correction or reinitialization after a visibility gap. Equal-energy
terminal images were retained as distinct records by frame identifier.

A tracker-independent physics baseline assigned the evaluable status of each
branch and frame. It used the frozen reference center and the expected radial
scaling

$$
\mathbf{p}_{\mathrm{pred}}(E)
=
\mathbf{c}+
\left[\mathbf{p}(E_{\mathrm{ref}})-\mathbf{c}\right]
\sqrt{\frac{E_{\mathrm{ref}}}{E}},
$$

the active detector annulus, and the common ScanE Gaussian gates. An accepted
measurement was classified as branch-consistent when its distance from
\(\mathbf{p}_{\mathrm{pred}}(E)\) did not exceed the frozen tolerance of
18 pixels. This tolerance was an evaluation gate and did not move the
tracker proposal or the fitted center.

Across the four cases, the baseline defined 221 visible spot opportunities
and 24 evaluable natural I(V) gaps for every method. Physics-consistent
measurement coverage was the number of baseline-visible opportunities with a
valid, branch-consistent Gaussian measurement divided by 221. Geometry
inconsistency was computed separately as the number of accepted Gaussian fits
more than 18 pixels from the expected branch divided by all accepted fits in
all visibility regimes. Pair-distance residuals were calculated only on
frames in which both branches were valid and branch-consistent, and were
summarized as
\(\left|D_{ij}(E)-D_{ij,\mathrm{pred}}(E)\right|\).

One of the four frozen pairs, pair 1 in `26a21_ScanE`, did not pass the
auxiliary coherent-pairwise-energy-law check. It was retained as a difficult
case rather than removed after observing tracker results. The native
EasyLEED Kalman tracker also diverged once in `26a21_ScanE` pair 2; the
exception was retained as a failure and the tracker was not reset.

### 4.5 Controlled SAM2-family analysis

SAM2, SAMURAI, and DAM4SAM were evaluated on both frozen pairs of
`26a24_ScanE`. Each branch received the same single 48 × 48 bounding-box
prompt at 187.9 eV and was propagated toward increasing energy and, in
reversed frame order, toward decreasing energy. The three environments used
byte-identical SAM2.1 base-plus checkpoint weights. The methods differed in
their propagation and memory mechanisms, while the prompt, image sequence,
mask-centroid proposal reduction, Gaussian estimator, acceptance gates, and
post-processing were fixed.

The same coverage, gap-recovery, and geometry-inconsistency definitions used
for the independent ScanE comparison were applied. The pooled denominators
were 182 baseline-visible spot opportunities and 13 evaluable natural I(V)
gaps per method. Because the experiment contains one series and two pairs, it
is interpreted as a controlled mechanism comparison rather than a general
ranking of SAM2-family trackers.

### 4.6 Domain-specific external reference

The restricted EasyLEED–ViPErLEED comparison used the 15 high-energy frames
of `26a24_ScanE` and the same two pairs as the controlled memory analysis.
EasyLEED produced single-spot Kalman trajectories, whereas ViPErLEED supplied
a whole-pattern LEED trajectory. Their native outputs were mapped to the same
four branch coordinates and evaluated with the common 48 × 48 Gaussian
contract.

EasyLEED was initialized once at 187.9 eV with one Kalman tracker at each of
the four frozen branch centers and a 12-pixel native tracking radius.
ViPErLEED was initialized once at the same reference frame using the frozen
18-beam Ag(111) pattern and reciprocal-space-to-screen registration, after
which it tracked the complete pattern. Both methods processed the 15-frame
segment toward non-decreasing energy without manual correction or
reinitialization, giving 60 predefined branch-frame opportunities per method.
Frame identity, rather than energy alone, was used as the record key so that
the two distinct terminal images at 220.2 eV were preserved. This comparison
tests whether the heterogeneous domain-specific outputs can enter the same
measurement chain; it is not included in the five-method comparison matrix or
used as external position ground truth.

### 4.7 Visibility regimes and recovery

The physics baseline distinguished three causes of an unavailable spot:
apparatus absence when the predicted position fell outside the active detector
annulus or inside an apparatus shadow; beam onset or termination when an
unmeasurable run occurred before the first or after the last visible fit; and
a natural I(V) visibility gap when an unmeasurable run was bracketed by
visible fits on the same physical branch. Only the third class entered the
gap-recovery denominator.

For a gap below the reference energy, recovery was sought while continuing
toward lower energy; for a gap above the reference, it was sought toward
higher energy. A gap was counted as recovered if a valid branch-consistent
Gaussian measurement occurred in the first contiguous baseline-visible run
after the gap, before another non-visible regime interrupted that run.
Recovery delay in frames was retained. No manual reinitialization was
permitted.

An accepted Gaussian fit more than 18 pixels from its expected branch was
reported as a geometry-inconsistent continuation. This is an operational
indicator of possible false continuity: the tracker has maintained a
measurable image feature, but the feature is not compatible with the frozen
energy-dependent branch geometry. It is not, by itself, external
identity-ground-truth proof of a specific misassociation.

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
