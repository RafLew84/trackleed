# Evaluating Heterogeneous LEED Spot-Tracking Methods under a Common Spot-Position Measurement Contract

Draft status: venue-portable scientific revision. Experimental description,
author information, submission metadata, and release authorization remain as
explicitly deferred inputs.

## Authors

TODO[AUTHOR]: author list, affiliations, corresponding author, ORCID identifiers.

## Abstract

Quantitative low-energy electron diffraction (LEED) analysis requires
persistent association between a measurement region and its target
diffraction spot as patterns move and visibility changes. We compared five
heterogeneous tracking strategies through a common position-measurement
contract: each native output supplied a proposal for a 48 × 48-pixel ROI,
and the same two-dimensional Gaussian estimator produced the reported spot
position. Robustness was assessed with fixed opportunity sets for coverage
and post-gap recovery, accepted-fit denominators for geometry inconsistency,
and method-specific surviving \(n\) for conditional residuals.

Validation combined known translations, a 701-image Cu(111) positive
control, two independent Ag(111) energy scans with natural \(I(V)\) gaps,
and a controlled SAM2-family memory experiment. Translation errors were
0.0043–0.0265 px with 74/80–80/80 accepted measurements. All methods
produced 2,804/2,804 Cu(111) measurements and 701/701 endpoints per primary
pair. In the Ag(111) scans, coverage was 188/221–217/221, recovery
17/24–23/24, and conditional median pair-distance residuals 1.737–1.837 px
(\(n=68–91\)). SAMURAI and DAM4SAM increased coverage over SAM2 but also
increased geometry-inconsistent continuation; this mechanism result is not
a general ranking.

Across 15 predeclared perturbations of ROI geometry and acceptance gates,
absolute outcomes changed, but Lucas–Kanade retained the lowest coverage and
the denominator-aware interpretation remained stable. The results assess
measurement-chain robustness, not absolute tracking accuracy. Proposal
differences were strongly suppressed when the target spot remained inside
the common estimator's capture region, making this a supporting mechanism
rather than the primary contribution.

## Keywords

Low-energy electron diffraction; spot tracking; quantitative image analysis;
Gaussian localization; visibility gaps; measurement robustness.

## 1. Introduction

### 1.1 Quantitative LEED requires persistent spot association

Low-energy electron diffraction (LEED) probes the periodic structure of a
surface through the energy-dependent diffraction of low-energy electrons
[@pendry1974leed]. A recorded LEED pattern contains discrete diffraction spots
whose positions, profiles, and integrated intensities can be extracted from
image sequences [@mayer2012easyleed; @schmid2025viperleed]. Quantitative
structural LEED most commonly uses intensity as a function of incident
electron energy, \(I(V)\) [@pendry1974leed; @schmid2025viperleed]. In time- or
temperature-dependent image series, localized spot centers and profiles can
additionally support measurements of displacement, pair distance, and profile
width. Each of these image-derived quantities requires the measurement region
to remain associated with the same physical diffraction spot throughout the
analysed sequence.

Tracking is therefore upstream of the physical measurement but is not itself
the measurement. A tracker supplies a location at which image information can
be evaluated; the quantitative position, width, or intensity is then obtained
from the corresponding pixels. This distinction matters because a smooth
trajectory can follow a background feature or a neighbouring diffraction
spot, while an irregular proposal can still yield a valid position if the
intended spot remains inside the downstream estimator's capture region.
Preservation of LEED spot identity consequently cannot be inferred from
trajectory smoothness alone.

The tracking problem becomes more difficult when the diffraction pattern
moves. Changes in the experimental state can translate or deform the observed
pattern, and varying the electron energy produces an approximately radial
motion proportional to \(1/\sqrt{E}\). LEED-specific tracking systems already
use this energy dependence to predict spot positions
[@mayer2012easyleed; @schmid2025viperleed]. A position estimator must then
accommodate both the expected physical motion and common-mode image motion
caused by projection or apparatus drift.

Spot visibility introduces a separate challenge. Because diffraction
intensity varies strongly with energy, a spot can fall below the measurement
threshold and later become measurable again. We refer to an internal,
intensity-driven interval bracketed by valid measurements of the same branch
as an \(I(V)\) visibility gap. Such a gap is not generic video occlusion:
image evidence for the intended diffraction feature is temporarily
insufficient even though other nearby spots or background structures may
remain visible. A useful tracking strategy must therefore support recovery of
the same physical branch without silently continuing on a distractor.

### 1.2 Existing LEED tracking and the remaining evaluation problem

Automated LEED spot tracking and localization have clear precedents. EasyLEED
tracks individually initialized spots with a Kalman filter informed by an
energy-dependent motion model and elliptical two-dimensional Gaussian image
measurements. If an image measurement is rejected, its prediction can continue
until a later update becomes available
[@mayer2012easyleed]. ViPErLEED-ImageJ instead identifies and tracks a complete
diffraction pattern using reciprocal-lattice geometry, the expected energy
scaling, and local corrections from neighbouring spots. It supports weak or
temporarily missing beams and is designed for extraction and processing of
LEED \(I(V)\) curves [@schmid2025viperleed]. Thus, automated tracking,
energy-dependent prediction, Gaussian localization, and propagation through
low-visibility intervals are not introduced here.

General-purpose computer-vision methods provide different representations of
the same association problem. These include propagated points based on local
correspondence [@cho2024locotrack], transformer-based bounding-box tracking
[@ye2022ostrack], classical optical flow [@lucas1981iterative], and
prompt-conditioned mask propagation with visual memory
[@ravi2025sam2; @yang2024samurai; @videnovic2025dam4sam]. Their native outputs
are heterogeneous: a point trajectory, updated bounding box, optical-flow
displacement, Kalman state, or binary mask. The box center and mask centroid
used in this study are local adapters that reduce those outputs to a TrackLEED
tracking proposal; they are not native position estimates. Even after this
reduction, the proposal coordinates do not have the same operational meaning
or uncertainty. Directly ranking them would therefore conflate the proposal
mechanism with the downstream position estimator.

This heterogeneity motivates a narrower evaluation question than whether LEED
spots can be tracked automatically. The relevant question for quantitative
analysis is when different proposal strategies lead to the same accepted
physical measurement and when motion or loss of visibility causes missing or
physically inconsistent results. A common downstream estimator makes that
question testable by converting every native proposal into the same
measurement ROI and deriving the reported position from the same image model.
The comparison then concerns the complete measurement chain rather than the
coordinate convention of a particular tracker.

Conditional localization error alone is insufficient for this purpose. A
method can have a small residual on the subset of images for which it returns
a proposal and the fit passes all gates, yet lose many measurable spots or
resume on an incompatible diffraction branch after a visibility gap. These
events disappear if only successful measurements enter the denominator. The
evaluation must therefore keep measurement coverage, post-gap branch-consistent recovery,
geometry-inconsistent continuation, and conditional position or pair-distance
residuals as separate outcomes. Without an external image-by-image reference,
such outcomes establish equivariance, reproducibility, and physical
self-consistency rather than absolute tracking accuracy.

### 1.3 Study objective and contributions

This study evaluates whether heterogeneous spot-tracking strategies support
an available and physically self-consistent quantitative LEED position under
one common proposal-to-ROI-to-Gaussian measurement contract. The primary
comparison keeps fixed opportunity sets for measurement coverage and
post-gap branch-consistent recovery separate from accepted-fit geometry
outcomes and conditional residuals with method-specific surviving \(n\). We
then test whether native proposal differences are suppressed when the target
spot remains inside the common estimator's capture region and whether the
resulting measurements support the pairwise endpoint
\(\Delta R/R=\Delta D/D\). TrackLEED is the reference implementation and
reproducibility vehicle; its graphical application and feature set are not
the subject of the study.

The contributions are:

1. A common proposal-to-ROI-to-Gaussian position-measurement contract for
   heterogeneous point, bounding-box, mask, optical-flow, and LEED-domain
   trackers.
2. A denominator-aware assessment that retains fixed opportunity sets for
   coverage and recovery and outcome-specific surviving sets for conditional
   geometry and residual statistics.
3. Controlled and case-based evidence for measurement-chain equivariance,
   endpoint agreement, motion and natural \(I(V)\)-gap robustness, and the
   continuity–consistency trade-off.
4. A predeclared 15-variant sensitivity analysis that tests whether the claim
   boundary depends on measurement-window geometry or acceptance gates without
   selecting an optimal contract.
5. A mechanistic account of when the common Gaussian estimator suppresses
   proposal differences inside its tested capture region.

The four evidence regimes are assigned distinct evidential roles rather
than pooled into one composite ranking. Controlled translations test
measurement-chain equivariance, Cu(111) provides a positive control for
endpoint agreement, independent Ag(111) energy scans test motion and natural
visibility gaps, and the matched SAM2-family panel examines a
continuity--consistency trade-off. This design does not provide absolute
spot-identity ground truth; it evaluates whether each strategy supports an
available and physically self-consistent quantitative measurement under the
frozen conditions.

## 2. Measurement chain and validation design

### 2.1 Tracking proposal and common measurement ROI

A tracker's native output was treated as a *tracking proposal*: information
used to place a downstream measurement region, not the reported spot position.
Point and Kalman methods supplied a coordinate, box methods supplied a box
center, mask methods supplied a mask centroid, and accepted optical flow
supplied a displaced point. Native confidence, visibility, mask, and fit
diagnostics could determine proposal availability under the frozen method
contract, but they were not interpreted as diffraction intensity.

Every available proposal was mapped to the original image coordinate system,
rounded to the nearest pixel, and used to place a 48 × 48-pixel region of
interest (ROI) on the original-intensity image. An ROI outside the image was
rejected rather than clipped or padded. Thus, methods could select different
regions, but each selected raw-image region entered the same position
estimator.

### 2.2 Gaussian-refined position and pairwise observable

Within the ROI, intensity was represented by a rotated elliptical Gaussian
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

The fitted amplitude \(A\), center \((x_c,y_c)\), widths
\((\sigma_x,\sigma_y)\), angle \(\theta\), and background \(B\) defined the
common image measurement. Adding the ROI origin to the local fitted center
gave the reported global position,

$$
\mathbf{p}_{i,t} = (x_{i,t}, y_{i,t}).
$$

Regime-specific acceptance contracts were frozen before comparison and held
constant across methods within a regime. They require finite fit diagnostics
and apply predefined quality, amplitude, center, width, and
proposal-to-center gates. Exact thresholds, bound semantics, and solver
initialization are reported in Table S2. An unavailable proposal,
out-of-image ROI, failed fit, or rejected fit produced an explicit missing
position; rejected values were not interpolated or carried forward.

For a frozen pair \(i,j\), the primary geometrical observable was

$$
D_{ij}(t) = \left\|\mathbf{p}_{i,t} - \mathbf{p}_{j,t}\right\|_2.
$$

For an approximately diametric pair, \(R=D_{ij}/2\), so at fixed beam energy
and projection geometry,

$$
\frac{\Delta R}{R}
=
\frac{\Delta D}{D}
=
\frac{D_{ij}(t)}{D_{ij,\mathrm{ref}}} - 1.
$$

For an energy scan, the expected radial contraction and residual were

$$
D_{ij,\mathrm{pred}}(E)
=
D_{ij,\mathrm{ref}}
\sqrt{\frac{E_{\mathrm{ref}}}{E}},
\qquad
\delta D_{ij}(E)=D_{ij}(E)-D_{ij,\mathrm{pred}}(E).
$$

Pair distance cancels a common translation of both spots but not non-rigid
distortion, sector-dependent behavior, or misassociation. The pair midpoint,
\(\mathbf{m}_{ij}(E)=[\mathbf{p}_i(E)+\mathbf{p}_j(E)]/2\), was therefore
retained as a separate apparatus-motion diagnostic and was not used to
recenter the primary endpoint.

### 2.3 Measurement-chain outcomes

The validation separates branch-consistent measurement coverage, post-gap
branch-consistent recovery, accepted measurements inconsistent with the
expected branch geometry, and position or pair-distance residuals conditional
on a valid branch-consistent measurement. Every predefined opportunity remains
in its fixed denominator when a proposal is unavailable or a Gaussian fit is
rejected. Conditional residuals are reported only with their method-specific
surviving sample size and never replace missing or inconsistent outcomes.

The energy-scan baseline distinguished apparatus absence, terminal beam onset
or disappearance, and an internal natural \(I(V)\) visibility gap bracketed
by visible measurements on the same branch. Only the last class entered the
recovery denominator. Recovery required a valid Gaussian measurement on the
expected branch in the first contiguous baseline-visible run after a gap,
without manual reinitialization. An accepted fit beyond the frozen branch
tolerance was recorded separately as geometry-inconsistent continuation. This
is an operational physical-consistency outcome, not external
identity-ground-truth proof of a particular misassociation. Table S2 records
the gates, fixed opportunity sets, and conditional denominators.

### 2.4 Evidence roles and independence

Controlled translations test displacement equivariance against a known image
transform; their four separately initialized source trials are not four
acquisitions. The Cu(111) sequence is a positive control for agreement of the
pairwise endpoint. Two independent Ag(111) energy scans provide four frozen
scan–pair stress-test cases. A separate one-series, two-pair scan supports the
controlled SAM2-family mechanism comparison, while the restricted
EasyLEED–ViPErLEED check is a subset with its own initialization contract and
denominator. Images, branches, pairs, series, and acquisitions are not treated
as interchangeable replication units.

TODO[EXPERIMENT]: The experimental team will supply and approve the
reader-facing description of the apparatus, sample preparation, acquisition
conditions, and experimental parameters in a later revision. No missing value
is inferred from filenames, retained metadata, or laboratory convention in the
current manuscript.

<!-- BEGIN TABLE 1 -->
**Table 1. Data and evidence roles.** Rows describe evidence roles, not five independent datasets. Controlled translations are derived from the Cu(111) sequence, and the external-reference row is a subset of the memory-analysis scan and outside the common five-method matrix. Outcomes remain separate rather than being reduced to a composite tracker ranking.

| Scientific series label | Image count | Frozen spots/pairs | Opportunity set | Evidence role | Primary observable | Independence/replication unit |
|---|---|---|---|---|---|---|
| Cu(111) thermal-cycle positive control | 701 acquired images | Spots 1, 3, 5, and 6; pairs 1–6 and 3–5 | 2,804 spot and 1,402 pair opportunities per method (14,020 spot-method rows total) | Positive control for agreement of the five strategy-to-measurement chains | ΔR/R = ΔD/D | One sequence, stratified by two frozen pairs; no population-level interval |
| Controlled translations | 24 digital image states: four source trials × (one seed state + five displaced states) | Spots 1, 3, 5, and 6; pairs 1–6 and 3–5 | 80 displaced-spot and 40 displaced-pair evaluation opportunities per method after exclusion of four seed states | Equivariance and pair invariance under a known image transform | Gaussian-refined displacement error; maximum absolute ΔD/D | Four separately initialized source trials from the Cu(111) sequence; not four acquisitions |
| Ag(111) energy-scan stress test, Scans A and B | 81 acquired images (35 + 46) | Two disjoint pairs (four spots) per scan | 221 baseline-visible branch opportunities and 24 evaluable gaps per method | Principal five-strategy test of radial motion and natural I(V) visibility gaps | Measurement coverage; post-gap branch-consistent recovery; geometry-inconsistent continuation; conditional absolute pair-distance residual | Two independent series, stratified by four frozen scan–pair cases |
| Ag(111) memory-analysis scan | 69 acquired images | Two disjoint pairs (four spots) | 182 baseline-visible branch opportunities and 13 evaluable gaps per method | Controlled SAM2-family continuity–consistency mechanism test | Measurement coverage; post-gap branch-consistent recovery; geometry-inconsistent continuation; conditional D(E) residual | One series, stratified by two frozen pairs; supporting mechanism evidence |
| Restricted high-energy external reference | 15-image subset of the memory-analysis scan | Same two pairs; four branch coordinates | 60 frozen branch opportunities per method; 4 gap checks | EasyLEED/ViPErLEED domain-contract compatibility under the common Gaussian estimator | Accepted-measurement coverage; EasyLEED–ViPErLEED common-center distance | Subset of the memory-analysis scan with a separate initialization contract and denominator; not an independent replicate and outside the common five-method matrix |
<!-- END TABLE 1 -->

Four translation trials provide 80 displaced-spot and 40 displaced-pair
evaluation opportunities per method after seed states are excluded; the oracle
ROI is a measurement control rather than a sixth tracker. The Cu(111) positive
control contributes 2,804 spot measurements per method and 701 endpoints for
each primary pair. The principal Ag(111) stress test uses 221
baseline-visible branch opportunities and 24 evaluable gaps per method. The
controlled SAM2-family test uses 182 branch opportunities and 13 gaps, and the
restricted compatibility check uses a separate 60-opportunity denominator.

All results are descriptive. Successive images are repeated observations
within propagated trajectories, and pairs within one series are retained as
strata rather than promoted to independent acquisitions. Tables S3 and S5
report case- and trial-stratified summaries. No composite score,
population-level interval, or absolute-accuracy ranking is constructed.

## 3. Tracking strategies and reproducible comparison contract

The comparison was organized by the representation propagated by each method
rather than by treating all native coordinates as equivalent position
measurements. The main panel comprised a point tracker, a bounding-box tracker,
a mask tracker, optical flow, and a LEED-specific Kalman tracker. Each method
received the prompt native to its family, derived from the same
Gaussian-refined seed center: point methods received the center coordinate,
whereas box- and mask-based methods received a 48 × 48 pixel box centered on
that coordinate. The native output was reduced to one proposal coordinate and
then passed to the common measurement chain defined in Section 2. Tracker
scores and internal fit diagnostics could determine whether a native proposal
was available according to the frozen method contract, but they never replaced
the common Gaussian-refined position or served as a physical intensity.

The executed revisions, checkpoints, configuration hashes, prompt conventions,
failure policies, propagation directions, and proposal-to-ROI adapters are
retained in the committed provenance package. Table 2 gives the reader-facing
comparison contract, while Table S1 records software, models, hashes, and
execution environments. The upstream citations establish the published method
designs; the frozen records establish the configurations used here.

### 3.1 Main five-strategy panel

**Point.** LocoTrack propagates a queried point and supplies native occlusion
and distance diagnostics [@cho2024locotrack]. Its finite point, subject to the
frozen native availability rule, was used directly as the proposal.

**Box.** OSTrack propagates a single-object bounding box in a joint
template--search transformer [@ye2022ostrack]. A finite positive-area box was
available, and its center was the proposal; the native score did not weight the
downstream Gaussian fit.

**Mask.** DAM4SAM propagates a segmentation mask with distractor-aware memory
and introspective updates [@videnovic2025dam4sam]. A finite nonempty mask was
available, and its pixel centroid was the proposal. Mask scores, areas, and
component diagnostics were not treated as physical intensity.

**Optical flow.** Lucas--Kanade estimates local displacement from image
gradients [@lucas1981iterative]. A point passing the frozen native flow gates
was the proposal. Rejection produced an explicit missing value; the last
successful point remained the later anchor without interpolation.

**LEED-domain single-spot tracking.** EasyLEED combines an energy-dependent
motion model, Kalman state, and internal Gaussian observation
[@mayer2012easyleed]. The exported finite Kalman coordinate was the proposal;
its internal image measurement and intensity remained diagnostics and never
replaced the common estimator.

The families retained their native inference representations upstream of the
comparison. Each proposal was mapped to the original image coordinates before
the common ROI was placed, so the controlled quantity was the downstream
physical measurement rather than an assertion that the trackers' internal
representations were identical.

### 3.2 Controlled SAM2-family panel

SAM2, SAMURAI, and DAM4SAM form a related family of promptable mask
propagators. SAM2 provides the reference streaming-memory architecture for
promptable video segmentation [@ravi2025sam2]. SAMURAI adapts SAM2 to
zero-shot visual object tracking by adding a Kalman motion model and
motion-aware mask and memory selection [@yang2024samurai]. DAM4SAM instead
uses distractor-aware memory and introspection-based updates
[@videnovic2025dam4sam].

All three methods used byte-identical backbone weights, the same single-object
box prompt in the frozen reference image, the same ordered images in each
propagation direction, and no later prompt or manual reset
[@sam2Software; @samuraiSoftware; @dam4samSoftware]. Each nonempty mask was
reduced to its pixel centroid and entered the common ROI, Gaussian estimator,
and physical-consistency gates. Exact configurations and hashes are in Table
S1.

This matched design controls the backbone weights, prompt information, input
sequence, proposal reduction, and downstream position measurement. It does
not reduce the three trackers to a one-line ablation: their motion prediction,
mask selection, visibility logic, and memory-update implementations are part
of the strategies being compared. The panel is therefore interpreted as a
mechanistic comparison of vanilla, motion-aware, and distractor-aware memory
strategies under a matched measurement contract, not as proof that one
isolated memory component caused every observed difference and not as a
general ranking of SAM2-derived trackers.

### 3.3 LEED-domain references and reproducibility

EasyLEED participated in the common five-strategy panel because one
independently initialized Kalman tracker yields one spot proposal and can
therefore enter the same single-spot contract as the four general computer
vision strategies. Its LEED-specific radial motion model makes it a
physics-informed domain baseline rather than another generic point tracker.

ViPErLEED has a different purpose and input contract. It identifies and tracks
a LEED pattern using reciprocal-lattice geometry, the expected
\(1/\sqrt{E}\) scaling, a global screen mapping, and local information from
neighboring spots, and is designed primarily for extraction and processing of
LEED \(I(V)\) curves [@schmid2025viperleed]. Its frozen whole-pattern output
was reduced to the coordinates corresponding to the two pairs and then passed
to the same Gaussian measurement as the EasyLEED coordinates
[@viperleedSoftware].

Because ViPErLEED received global pattern information rather than one seed per
tracked spot, it was not inserted as a sixth row in the common strategy
matrix. The restricted high-energy comparison instead asks whether the
single-spot EasyLEED and whole-pattern ViPErLEED outputs place the intended
spots within the capture region of the same downstream estimator. It does not
use either output as external position ground truth.

Exact versions, revisions, checkpoints, and environments are retained in Table
S1; its accompanying supplementary native-method contract records
reader-facing preprocessing and thresholds. Frozen native payloads preserve
the representation-specific outputs before proposal reduction, while the
required headless reconstruction route starts from those payloads and rebuilds
ROI placement, common measurements, outcomes, tables, and figures without
tracker inference or the graphical application.

<!-- BEGIN TABLE 2 -->
**Table 2. Methods mapped to the common measurement contract.** The first five rows form the common strategy matrix. The controlled SAM2-family rows used matched checkpoint weights and a common prompt. Every available proposal entered the same raw-image ROI and two-dimensional elliptical-Gaussian position estimator. ViPErLEED is a separate whole-pattern external reference, not a sixth row in the common five-method matrix. Exact versions, revisions, configurations, thresholds, hashes, preprocessing, and environments are reported in Table S1 and its accompanying supplementary native-method contract.

| Method | Native output | Proposal adapter | Native availability | Shared downstream measurement | Reset policy | Study role |
|---|---|---|---|---|---|---|
| LocoTrack | Point plus occlusion and expected-distance diagnostics | Native point | Finite point; energy scans also require the frozen native visibility gate | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | General point tracker |
| OSTrack | Bounding box plus native score | Box center | Finite positive-area box | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | General bounding-box tracker |
| DAM4SAM | Binary mask plus native diagnostics | Mask centroid | Finite nonempty mask | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | General mask tracker with distractor-aware memory |
| Lucas–Kanade | Optical-flow point plus residual and closure diagnostics | Accepted point | Match passing the frozen native flow gates | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited; last successful point remains the anchor | Classical motion baseline |
| EasyLEED | Kalman x,y plus native intensity and tracking radius | Kalman x,y | Finite exported Kalman x,y | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | LEED-domain single-spot baseline |
| SAM2.1 | Binary mask | Mask centroid | Finite nonempty mask | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | Vanilla streaming-memory control |
| SAMURAI | Binary mask plus motion-aware selection | Mask centroid | Finite nonempty mask | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | Motion-aware memory strategy |
| DAM4SAM | Binary mask plus distractor-aware diagnostics | Mask centroid | Finite nonempty mask | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | Distractor-aware memory strategy |
| ViPErLEED | Whole-pattern beam coordinate | Coordinate of corresponding beam | Finite pattern-associated coordinate for the corresponding beam | 48 × 48 raw-image ROI → common elliptical-Gaussian position | Prohibited | Separate whole-pattern reference |
<!-- END TABLE 2 -->

## 4. Results

### 4.1 Controlled translations isolate measurement-chain behavior

LocoTrack, OSTrack, and DAM4SAM each returned a proposal and a valid common
Gaussian measurement for all 80 evaluated spot opportunities (Figure 2;
Table S7). Lucas--Kanade returned 76/80 proposals, all of which produced valid
Gaussian measurements. EasyLEED returned 80/80 Kalman proposals, but six did
not yield an accepted common fit, giving 74/80 measurement coverage. The
corresponding pairwise endpoint coverage was 40/40 for LocoTrack, OSTrack, and
DAM4SAM, 36/40 for Lucas--Kanade, and 34/40 for EasyLEED. Missingness
therefore arose at different stages of the measurement chain and was not
absorbed into the displacement-error statistic.

Before Gaussian refinement, the median displacement-equivariance error was
0.433 px for LocoTrack, 2.183 px for OSTrack, 1.493 px for DAM4SAM, and
0.087 px for Lucas--Kanade. After refinement, the corresponding medians were
0.0043, 0.0048, 0.0060, and 0.0050 px. These values were of the same order as
the 0.0053 px median of the oracle-ROI control, which includes interpolation
and estimator error but no tracker. EasyLEED showed the same suppression at a
larger scale: its median native error of 14.758 px was reduced to 0.0265 px
after the common fit. Its 95th-percentile refined error nevertheless remained
0.382 px, compared with 0.0127--0.0168 px for the other four methods.

The pair-distance endpoint was also insensitive to a common translation when
both spots remained measurable. The maximum absolute change in
\(\Delta D/D\) was 0.001229 percentage points for LocoTrack, 0.001952 for
OSTrack, 0.001869 for DAM4SAM, and 0.001042 for Lucas--Kanade, compared with
0.001147 for the oracle control. EasyLEED reached 0.062943 percentage points
and had the lowest pair coverage. Thus, a successfully captured spot could be
localized near the estimator floor even when the native proposal was much less
accurate, but this conditional result did not eliminate differences in
measurement availability. The experiment establishes equivariance of the
complete measurement chain under a known image transform, not absolute
tracking accuracy or identity ground truth.
The four independently initialized trial-level summaries are retained
separately in Table S5.

### 4.2 Cu(111) positive control establishes endpoint agreement

All five strategies completed the Cu(111) positive control without a missing
proposal or rejected common fit: each method produced 2,804/2,804 Gaussian
measurements, and each primary pair produced 701/701 valid \(\Delta R/R\)
endpoints (Figure 3; Table S7). The largest separation between two native
method proposals was 16.759 px, whereas the largest separation between the
corresponding Gaussian-refined centers was 0.0502 px. At the endpoint level,
the maximum between-method difference over the entire sequence was only
0.001710 percentage points for pair 1--6 and 0.002237 percentage points for
pair 3--5.

The five methods also reproduced the same predefined sequence landmarks. Over
the five-image plateau, their median \(\Delta R/R\) estimates ranged from
\(-0.686676\%\) to \(-0.686244\%\) for pair 1--6 and from
\(-1.570833\%\) to \(-1.570300\%\) for pair 3--5. All methods placed the
post-plateau minimum of both pairs in the same image. The minimum values ranged
from \(-0.726482\%\) to \(-0.726195\%\) for pair 1--6
and from \(-1.580123\%\) to \(-1.579795\%\) for pair 3--5. In the final
state, pair 1--6 reached \(+0.121606\%\) to \(+0.121691\%\),
whereas pair 3--5 reached \(-0.152478\%\) to \(-0.152338\%\). The contraction
therefore largely reversed by the final state, but the two sectors retained
different endpoint values. With only two primary pairs, this difference was not
interpreted as crystallographic anisotropy or used to infer a thermal-expansion
coefficient or structural transition.
Exact method- and pair-specific landmark values are reported in Table S4.

### 4.3 Independent Ag(111) scans separate availability from conditional residuals

The independent Ag(111) energy scans separated the strategies in outcomes that
were absent from the Cu(111) positive control (Figure 4; Table S7). LocoTrack
had 217/221 branch-consistent measurements (98.19% coverage), followed by
OSTrack and DAM4SAM with 216/221 each (97.74%), EasyLEED with 214/221
(96.83%), and Lucas--Kanade with 188/221 (85.07%). The denominators were
identical and included every baseline-visible opportunity, so the 33
Lucas--Kanade measurements missing from the branch-consistent set were not
discarded before comparison.

Post-gap branch-consistent recovery produced a similar but non-identical
ordering. LocoTrack and DAM4SAM recovered 23/24 natural I(V) gaps (95.83%),
EasyLEED recovered 22/24 (91.67%), OSTrack recovered 21/24 (87.50%), and
Lucas--Kanade recovered 17/24
(70.83%). No method was manually reinitialized after a gap. The frozen
difficult pair that failed the auxiliary coherent-energy-law check remained in
these denominators. The single EasyLEED Kalman divergence was likewise retained
as a failure rather than corrected.

Continuity did not always imply physical branch consistency. LocoTrack and
DAM4SAM each produced 36 geometry-inconsistent accepted fits among 272
accepted measurements in all visibility regimes (13.24%). OSTrack produced
34/269 (12.64%), Lucas--Kanade 14/206 (6.80%), and EasyLEED 4/222 (1.80%).
These events indicate that an image feature remained measurable outside the
frozen radial branch; they do not constitute external ground-truth proof of a
particular identity switch.

Among images on which both pair branches survived all proposal, fit, and
geometry gates, the conditional median absolute pair-distance residuals
occupied a narrow range: 1.737 px (n=84) for EasyLEED, 1.764 px (n=68) for
Lucas--Kanade, 1.770 px (n=90) for OSTrack, 1.788 px (n=91) for DAM4SAM, and
1.837 px (n=91) for LocoTrack. In particular, Lucas--Kanade had a conditional
residual similar to the other methods while losing substantially more visible
measurements and gap recoveries. Conditional distance quality therefore did
not summarize measurement-chain robustness; coverage, recovery, and
geometry-inconsistent continuation remained separate outcomes.

The case-stratified results and leave-one-case-out analysis are reported in
Table S3. Across omission of each frozen scan--pair case, coverage ranged from
97.4--98.8% for LocoTrack, 97.2--98.8% for OSTrack and DAM4SAM, 81.9--87.6%
for Lucas--Kanade, and 95.5--97.5% for EasyLEED. The corresponding
gap-recovery ranges were 94.1--100.0%, 84.2--88.9%, 94.1--100.0%,
63.2--76.5%, and 88.9--100.0%, respectively. Geometry-inconsistency ranges
were 12.6--13.5%, 11.8--13.5%, 12.6--13.6%, 3.4--9.5%, and 0.0--2.4%.
These ranges show how the pooled count summaries depend on any one predefined
case; they are descriptive sensitivity results, not confidence intervals or
population-level method rankings.

### 4.4 Predeclared measurement-window and capture-contract sensitivity

To test whether the Ag(111) conclusions depended on the central 48 × 48-pixel
ROI and acceptance contract, we re-evaluated the stored tracker proposals
under 15 predeclared measurement-window and capture-contract variants
(Figure S2; Table S6). The set comprised nine ROI-side × proposal-to-fit
combinations (32, 48, or 64 px × 12, 15, or 18 px), including the central
48 px × 15 px contract, and paired one-factor perturbations of branch
tolerance (12 or 24 px), the Gaussian \(R^2\) threshold (0.80 or 0.90), and
the amplitude gate (0.75× or 1.25× its central value). Tracker inference and
the reference opportunity sets were held fixed: coverage retained 221
baseline-visible opportunities and recovery retained 24 evaluable natural
\(I(V)\) gaps. Geometry inconsistency remained conditional on each variant's
accepted fits, and pair-distance residuals retained their method- and
variant-specific surviving \(n\).

The perturbations changed absolute outcomes. Across all methods and variants,
branch-consistent coverage spanned 170–221/221, post-gap branch-consistent
recovery 17–24/24, geometry-inconsistency fractions 1/222–69/272
(0.45–25.37%; accepted-fit denominators across all method–variant
combinations, 187–286), and conditional median pair-distance residuals
1.499 px (n=79)–1.939 px (n=99; pair-frame n across all combinations,
54–104). The 64-pixel ROI and \(R^2=0.90\) variants reduced coverage for
every method relative to the central contract, while changing branch
tolerance altered geometry classification and the surviving residual set by
construction. Both amplitude-gate perturbations reproduced the reported
central outcomes. Despite these numerical changes, Lucas–Kanade retained the
lowest coverage in every variant, and the central conclusion remained:
conditional residuals cannot summarize robustness without fixed-denominator
coverage and recovery, geometry-consistency outcomes, and their own surviving
\(n\). These are sensitivity results for the tested contracts, not a parameter
optimization or a comparative accuracy ranking.

The predeclared width-materiality trigger was not activated. No method
produced a width-only rejection, and the counts of fits near the width gate
remained below their method-specific trigger thresholds. Width sensitivity
was therefore not run, no alternate width contract was generated, and the
central measurement contract was not changed.

### 4.5 Controlled SAM2 family exposes a continuity-consistency trade-off

Under byte-identical checkpoint weights, vanilla SAM2 produced 126/182
branch-consistent measurements (69.23% coverage) and recovered 9/13 natural
I(V) gaps (69.23%; Figure 5). SAMURAI increased coverage to 181/182 (99.45%)
and DAM4SAM to 179/182 (98.35%); both memory-aware variants recovered all
13 gaps. The matched panel therefore associated the motion-aware and
distractor-aware strategies with substantially greater continuity than the
vanilla propagation contract.

The continuity gain was accompanied by more accepted measurements outside the
frozen physical branch. SAM2 produced 8 such fits among 149 accepted
measurements across all regimes (5.37%), compared with 36/236 for SAMURAI
(15.25%) and 36/233 for DAM4SAM (15.45%). Thus, the variants that most
consistently bridged visibility gaps also more often continued on a measurable
but geometrically implausible feature. Because the implementations differ in
motion prediction, mask selection, visibility handling, and memory update, the
matched experiment supports a strategy-level continuity--false-persistence
trade-off rather than attributing the entire effect to one isolated memory
component. The result is supporting mechanism evidence from one series and two
pairs, not a general ranking of SAM2-derived trackers.

### 4.6 Supporting compatibility and missingness checks

EasyLEED and ViPErLEED each produced all 60 expected branch proposals over the
15-image high-energy interval, despite using single-spot and whole-pattern
initialization contracts, respectively (Figure S4). For both methods, the
common Gaussian estimator accepted the same 38/60 branch measurements with
the same per-branch counts: 13/15 and 7/15 for the two branches of pair 1, and
13/15 and 5/15 for pair 2. Both methods also passed all four predefined
absence-to-recovery checks.

Across the 60 corresponding native proposals, the EasyLEED--ViPErLEED
separation had a median of 0.626 px and a maximum of 1.696 px. For the 38
jointly accepted common measurements, the separation between
Gaussian-refined centers fell to a median of 0.000986 px and a maximum of
0.0240 px. At the two terminal images, the returning branch of each pair
yielded a valid fit for both methods, while the opposite branch failed the
Gaussian gate. These events established branch recovery but did not provide a
terminal pair-distance measurement.

The agreement after common refinement shows that both heterogeneous
domain-specific outputs placed the corresponding high-energy spot features
inside the capture region of the same estimator. It does not show equivalence
outside this restricted interval or establish superiority of either native
contract. This compatibility is not external absolute tracking accuracy or
identity ground truth.

The separate saved-trajectory diagnostic for Cu(111) pair 2--4 produced 24
consecutive missing measurements despite a proposal being present in every
image (Figure S6). All 24 candidates passed the \(R^2\), center, and width
gates and failed only the common amplitude threshold. This supporting check
shows that proposal availability does not guarantee an accepted quantitative
position. The trajectory was outside the five-method run and was neither used
for strategy comparison nor treated as spot-identity ground truth.

## 5. Discussion

### 5.1 Proposal quality and measurement quality

Across the controlled translations and Cu(111) positive control, the common
estimator reduced much larger native proposal differences to closely agreeing
refined positions when the intended spot remained inside its capture region.

This convergence should not be interpreted as evidence that the trackers have
equal absolute coordinate accuracy. The known-translation test subtracts each
method's own initial coordinate and therefore evaluates equivariance rather
than a fixed position offset. The Cu(111) sequence, in turn, provides
between-method agreement without an external position reference. Both results
instead identify a practical capture-region effect: proposal errors smaller
than the estimator's effective capture range can have little influence on the
refined center, whereas a proposal that excludes the intended spot cannot be
repaired downstream. The appropriate unit of evaluation is consequently the
combined proposal-to-ROI-to-fit chain, with proposal availability and fit
acceptance retained alongside the conditional localization error.

### 5.2 Conditional residuals require coverage and consistency denominators

The independent Ag(111) energy scans exposed differences that were hidden in
the positive-control sequence. Their radial motion and natural I(V) visibility
gaps created opportunities for four distinct outcomes: absence of a native
proposal, rejection of the downstream Gaussian measurement, recovery on the
expected radial branch after a gap, and continuation on a measurable feature
outside that branch. Combining these outcomes into a single successful-frame
error would remove the events that most directly determine whether a
quantitative time or energy series remains usable.

Lucas–Kanade combined the lowest fixed-denominator coverage and recovery with
a conditional residual inside the narrow five-method range; the residual alone
therefore concealed unavailable or inconsistent measurements.

Conversely, high proposal or fit availability does not by itself establish
correct identity preservation. Accepted Gaussian fits outside the frozen
energy-dependent branch occurred for every method and were most frequent in
absolute count for LocoTrack and DAM4SAM in the independent scans. These
events are operational evidence of physically inconsistent continuation, not
external proof of a particular identity switch. They nevertheless show why
measurement coverage must be paired with a geometry constraint. In the
present comparison, a defensible report therefore requires coverage against
the 221 visible opportunities, recovery against the 24 evaluable gaps,
geometry-inconsistent fits relative to each method's accepted-fit count, and
the residual distribution only for pairs whose two branches passed all
proposal, fit, and geometry gates.

The measurement-contract sensitivity qualifies rather than universalizes
these conclusions. The common-estimator interpretation and the separation
between fixed-denominator completeness and conditional residual quality
survived all 15 predeclared variants, but the numerical envelopes changed
with ROI geometry and acceptance gates. The central residual range is
therefore not generalized to every contract: across the sensitivity set,
medians spanned 1.499–1.939 px and retained method- and variant-specific
\(n\). Likewise, Lucas–Kanade's consistently lowest coverage is a result for
the four frozen case strata and tested contracts, not a population-level
ranking. Sensitivity thus supports claims about denominator-aware
measurement-chain robustness within the acquisition-closed case boundary;
it does not select an optimal configuration or establish absolute tracking
accuracy.

### 5.3 Visual persistence and physical consistency

Under matched weights and prompts, the memory-aware variants increased
continuity while also increasing continuation outside the frozen physical
branch.

This observation is particularly relevant to diffraction data because the
temporary unmeasurability of a spot is often driven by the physical I(V)
response rather than ordinary visual occlusion. During such an interval,
image evidence for the intended feature may be insufficient even though
nearby diffraction features remain visible. Under the frozen contract, the
memory-aware strategies therefore showed a different balance between
discontinuity and possible false persistence rather than an unconditional
improvement. A physics-aware branch check supplies information that the visual
memory mechanisms do not encode directly and makes the two outcomes separable.

The experiment does not isolate one internal memory component. SAM2, SAMURAI,
and DAM4SAM also differ in motion prediction, mask selection, visibility
handling, and update logic. Moreover, the comparison covers one energy scan
and two pairs. The supported conclusion is therefore limited to a
strategy-level continuity--consistency trade-off under the frozen contract,
not a universal ordering of SAM2-derived methods or a causal attribution to
one architectural operation.

### 5.4 Consequences for quantitative LEED observables

Complete Cu(111) measurement coverage yielded mutually consistent pairwise
endpoints across all five proposal strategies, demonstrating an easy positive
control rather than a tracker benchmark.

Pairwise distance is useful in both data regimes because it cancels a common
translation of the observed pattern. In the energy scans, comparison with
\(D(E)\propto1/\sqrt{E}\) then provides a physically constrained
self-consistency test without requiring an image-by-image manual center. The
separate pair-midpoint motion shows, however, that the ideal radial law is not
an exact pixelwise ground truth for the apparatus. Pair distance cannot remove
non-rigid distortion, sector-dependent projection effects, or an association
with the wrong diffraction feature. Its role is therefore to test whether a
measurement remains compatible with the frozen physical branch, not to supply
absolute position labels.

Missingness must also be propagated explicitly to the endpoint. The separate
Cu(111) pair 2--4 diagnostic showed that a tracking proposal can remain
available while the common fit fails because the spot amplitude falls below
the acceptance threshold. Interpolation or forward filling would hide this
distinction and could create an apparently continuous physical curve without
a supported position measurement. For the same reason, native tracker
confidence, occlusion probability, or mask area cannot be interpreted as a
LEED I(V) observable. An intensity curve, if analysed in future work, must be
derived from the amplitude or integrated volume of the common image fit and
validated separately from tracking continuity.

The pair 2--4 result has a narrower evidential role than the five-method
positive control. It used an earlier saved YOLO proposal trajectory, lay
outside the frozen five-method experiment, and did not provide ground truth
for spot identity. It supports only the distinction between proposal
availability and downstream quantitative-measurement availability.

### 5.5 Limitations and case scope

The study has no external image-by-image reference for absolute spot position
or identity. Controlled translations provide an exact commanded displacement,
but they test measurement-chain equivariance on transformed images rather
than the full dynamics of an experiment. The physical branch model and
pairwise geometry provide independent constraints in the energy scans, but a
geometry-inconsistent fit is an indicator of possible false continuity rather
than manually verified proof of misassociation. Accordingly, the results do
not establish an absolute-accuracy ranking.

The number of independent experimental units is also limited. The principal
stress test contains four pairs from two Ag(111) energy scans, while the
controlled SAM2-family analysis contains two pairs from one additional scan.
The Cu(111) sequence is an easy positive-control regime in which the endpoint
did not discriminate the five strategies, not an additional difficult
tracking comparison. Successive images are repeated observations within
propagated trajectories and are not independent replicates. The reported
rates and residuals should therefore be read as descriptive measurements of
the frozen cases rather than population-level estimates for all LEED patterns,
materials, instruments, or visibility regimes.

The comparison does not hold every implementation detail constant across all
experiments. DAM4SAM used different frozen configurations across the Cu(111),
translation, and independent energy-scan regimes; comparisons across those
regimes cannot be attributed solely to the data. Only the controlled
SAM2-family experiment used byte-identical checkpoint weights. EasyLEED and
ViPErLEED also differ fundamentally in initialization:
one tracks individually seeded spots, whereas the other registers a complete
reciprocal-lattice pattern. Their restricted high-energy comparison
demonstrates compatibility with the common estimator, not equivalence of the
native tracking contracts.

Finally, the central Gaussian gates, 48 × 48-pixel ROI, seeds, and branch
tolerance define the estimator capture contract and affect the observed
balance between coverage and inconsistency. We tested 15 predeclared
perturbations of ROI size, proposal-to-fit distance, branch tolerance,
\(R^2\), and amplitude acceptance using stored proposals from the Ag(111)
stress test. This local sensitivity is not parameter optimization, a full
capture-basin map, or evidence for untested detector geometries and data
regimes; its claims remain limited to the observed numerical envelopes and
outcome-specific denominators. Width sensitivity was not run because the
predeclared materiality trigger was not activated, not because width limits
were shown to be irrelevant. The study also does not test a full structural
LEED \(I(V)\) workflow. The Cu(111) endpoint is consequently limited to
relative changes of the measured diffraction-pattern scale; it is not used
to infer a thermal-expansion coefficient, structural transformation, or
crystallographic anisotropy.

## 6. Conclusions

Across controlled translations and the Cu(111) positive control, the common
Gaussian estimator strongly reduced the influence of differences between
native point, bounding-box, mask, optical-flow, and LEED-specific proposals
when the intended spot remained inside its capture region. The resulting
pairwise \(\Delta R/R\) endpoints were mutually consistent across the five
strategies. This establishes measurement-chain equivariance and agreement in
the tested cases, not instrument-traceable absolute tracking accuracy or
spot-identity ground truth.

Across four frozen pairs from two independent Ag(111) energy scans, similar
conditional pair-distance residuals coexisted with substantial differences in
measurement coverage, post-gap branch-consistent recovery, and
geometry-inconsistent continuation. The controlled SAM2-family experiment
further showed, in one series and two pairs, that greater visual continuity can
coexist with more persistence outside the expected physical branch.
Conditional residuals must therefore be reported with their method-specific
surviving n, alongside coverage and recovery evaluated on fixed opportunity
sets and the corresponding geometry-consistency outcomes.

The appropriate evaluation target for quantitative LEED is the complete chain
from tracking proposal through the common measurement ROI and Gaussian-refined
position to the physical endpoint. The case-based evidence supports separate,
denominator-aware outcomes under the tested contracts; it does not support a
single universal ranking, new surface physics, or a structural LEED \(I(V)\)
analysis.

## 7. Modular end matter

### Reproducibility

The reproducibility record is organized around the frozen measurement chain,
not the graphical application state. For the Cu(111) sequence and the three
Ag(111) energy scans, the planned data record will contain the ordered raw-image
inventory with relative paths, acquisition metadata, file sizes, and SHA-256
hashes. The controlled-translation experiment introduces no additional raw
acquisition: its inputs are four frozen source states from the Cu(111)
sequence, and its transform contract and source hashes are retained separately.
The data record will also contain the frozen spot and pair definitions,
initialization prompts, preprocessing inputs where applicable, complete method
configurations, upstream code revisions, and checkpoint identifiers. Native
tracker payloads will be retained before their reduction to a proposal
coordinate, so that point coordinates, bounding boxes, masks, optical-flow
diagnostics, and Kalman states can be audited independently of the common
measurement step.

The processed record will preserve one explicit row for every expected
method--spot--image combination. These records distinguish an unavailable
proposal from an out-of-image ROI, failed fit, rejected fit, and accepted
measurement rather than dropping or interpolating missing values. For accepted
fits, the stored fields include the proposal and Gaussian-refined coordinates,
fit parameters and diagnostics, acceptance outcome, and the information needed
to reconstruct pair distances, \(\Delta D/D=\Delta R/R\), post-gap branch-consistent
recovery, and geometry-consistency outcomes. The completed Cu(111) run
additionally links each reduced proposal to its saved native payload and records
configuration, input-inventory, output-table, and summary hashes in its run
manifest.

In the working manuscript, each main figure is stored with a self-contained
source-data directory, an upstream-provenance record, a deterministic builder,
validation tests, and manifests containing hashes of the source, code, and
rendered outputs. The stratified supplementary tables likewise contain frozen
source tables, a deterministic builder, tests, and a manifest recording source
and output hashes. These committed display-level artifacts permit the results
to be rebuilt and checked without rerunning tracker inference. Their provenance
links remain necessary: the compact display tables are publication views of
the frozen experiment outputs, not replacements for the complete per-image
records.

The scientific core is reconstructed and validated from the repository root by
`python -m scripts.reconstruct_publication --clean --validate`. The command
starts from the saved native tracker payloads, validates their frozen hashes and
schemas, reconstructs proposal reduction, ROI placement, Gaussian measurements,
the central result and sensitivity, rebuilds all main and supplementary tables
and figures, and checks the final claim boundary. It runs headless in the frozen
CPU environment without the TrackLEED GUI, an inference GPU, third-party
checkpoint files, or tracker inference. The final validation step checks a
semantic fingerprint covering the prose, claim--evidence map, source data,
builders, validators, displays, Supplementary Information inventory, central
outcomes, sensitivity, denominators, limitations, and claim boundaries. The
command cannot update that fingerprint automatically.

The public release remains planned as two linked, versioned records: a data
record containing the raw images, native tracker payloads, processed
measurements, and figure source data, and a code record containing the headless
runner, artifact builders, validators, tests, and machine-readable environment
records. Full tracker inference remains a separate optional path. Each required
external checkpoint is identified by its official source, variant, version or
upstream revision, and SHA-256 hash; checkpoint files will be redistributed
only where their licenses permit.

TODO[RELEASE-AUTHORIZATION]: select and authorize the public data and code
repositories, licenses, versioned release identifiers, DOI values, checkpoint
redistribution permissions, and final release scope. Until those records are
deposited, committed hashes document computational provenance but do not
constitute a public data-availability statement.

### Submission status

TODO[SUBMISSION]: apply the selected journal's current article type, formatting,
word-limit, declaration, and repository-access requirements after the
scientific text is approved.

### CRediT authorship contribution statement

TODO[AUTHOR].

### Declaration of competing interest

TODO[AUTHOR].

### Acknowledgements

TODO[AUTHOR]: funding, instrument access, computational resources, collaborators.

### Data and code availability

The final archive statement and persistent identifiers are subject to the
release authorization above.
