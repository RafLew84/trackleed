# Evaluating Spot-Tracking Strategies for Quantitative LEED Analysis

Draft status: working manuscript; Sections 2–5 are drafted, while the
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
   independent Ag(111) energy scans, coverage spans `188/221-217/221` and gap recovery
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
3. A five-strategy comparison on independent Ag(111) energy scans using common
   denominators for visible measurement opportunities and natural I(V) gaps.
4. A controlled SAM2/SAMURAI/DAM4SAM analysis that isolates the effect of memory
   strategy under byte-identical checkpoint weights.
5. Propagation of tracker proposals to the pairwise physical endpoint
   `Delta R/R = Delta D/D` in a 701-image constant-energy Cu(111) sequence.

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

Controlled translation experiments used four images representing the initial
low-temperature state (10 °C), the first image of the five-image 800 °C
plateau, the post-plateau \(\Delta R/R\) minimum (maximum contraction;
797 °C), and the final cooled state (92 °C). Each image was transformed by the
same closed path of six subpixel translations. The four spots and two pairs
used in the full Cu(111) sequence were retained. This regime provides a known
image transformation and therefore evaluates equivariance of the complete
proposal-to-measurement chain without requiring a manually annotated
trajectory.

The principal stress test used two independently acquired Ag(111) energy
scans, denoted **Scan A** and **Scan B**. Scan A contains 35 images spanning
40.0–220.5 eV, whereas Scan B contains 46 images spanning 40.0–220.4 eV. In
both series, 188.5 eV was selected as the reference energy. Two disjoint
diametric pairs were detected and frozen in each series before the compared
trackers were evaluated. These data combine energy-dependent radial motion
with intensity-driven disappearance and reappearance of diffraction spots.

A third Ag(111) energy scan, denoted the **memory-analysis scan**, was
reserved for the controlled SAM2-family analysis. It contains 69 images
spanning 45.0–220.2 eV, uses 187.9 eV as the reference energy, and contains
two frozen diametric pairs. SAM2, SAMURAI, and DAM4SAM were compared on this
series under identical checkpoint weights and a common downstream measurement
contract. A restricted 15-image high-energy interval spanning
187.9–220.2 eV was also used to place EasyLEED and ViPErLEED under the same
downstream estimator. That limited comparison is an external domain reference,
not a row in the common five-method matrix.

For traceability, the archive identifiers of Scan A, Scan B, and the
memory-analysis scan are `26a21_ScanE`, `26a21b_ScanE`, and
`26a24_ScanE`, respectively. These identifiers are used in the data manifests
and repository paths, not as scientific descriptions of the experiments.

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
center gave the reported global position of spot \(i\) in image \(t\),

$$
\mathbf{p}_{i,t} = (x_{i,t}, y_{i,t}).
$$

The estimator implementation was common to all methods, whereas acceptance
gates were frozen separately for the Cu(111) and energy-scan regimes and then held
constant across methods within each regime. In the Cu(111) sequence and
controlled-translation experiment, a fit required finite diagnostics and
center uncertainties, \(R^2\geq0.85\), \(A\geq0.05\), local center
coordinates strictly between -2 and 50 pixels, and both fitted widths strictly
between 0.5 and 48 pixels. In the energy-scan experiments, the gates were
\(R^2\geq0.85\), \(A\geq0.012\),
\(0.6\leq\sigma_x,\sigma_y\leq13\) pixels, and a distance of at most 15
pixels between the proposal and refined center. The energy-scan solver used the same
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
relative radial change can therefore be evaluated without estimating an
image-specific pattern center:

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

Run-specific statements in this section were transcribed from the frozen
[five-method Cu(111) run
manifest](../.scratch/trackleed-physics-method-paper/artifacts/cu111-tracker-evaluation-subset/runs/five-methods-v1/run_manifest.json),
[controlled-translation
contract](../.scratch/trackleed-physics-method-paper/artifacts/controlled-translation-chain/contract.json),
[independent Ag(111) energy-scan
artifact](../.scratch/trackleed-physics-method-paper/artifacts/independent-scane-validation/README.md),
[SAM2-family
artifact](../.scratch/trackleed-physics-method-paper/prototypes/scane_tracer/README.md),
and [domain-baseline execution
record](../.scratch/trackleed-physics-method-paper/artifacts/domain-baselines-installation.md).
These records establish the executed revisions, checkpoints and hashes, input
conversion, prompt convention, thresholds, failure policy, propagation
direction, and proposal-to-ROI adapters; the upstream citations below establish
the published method designs. The local links are working-draft provenance and
will be replaced by the archived dataset DOI and Table S1 in the submitted
manuscript.

### 3.1 Main five-strategy panel

**LocoTrack.** LocoTrack is a tracking-any-point model that combines local
all-pair correspondence with temporal modeling to propagate queried points
through a video [@cho2024locotrack]. We used the Base model in offline mode
from upstream revision `3385a23`, with the published `locotrack_base.ckpt`
checkpoint [@locotrackSoftware]. A query consisted of the image index and the
common seed point.
The finite predicted point was the proposal coordinate. The model also returned
occlusion and expected-distance logits. In the energy scans, the frozen
adapter combined these outputs as

$$
P_{\mathrm{vis}} =
\left[1-\operatorname{sigmoid}(o)\right]
\left[1-\operatorname{sigmoid}(d)\right],
$$

where \(o\) and \(d\) are the occlusion and expected-distance logits,
respectively. A threshold of 0.5 defined native proposal availability. In the
Cu(111) positive control and controlled translations, every finite point was
passed to the common measurement and the same probability was retained only
as a diagnostic. No visibility or uncertainty correction was applied to the
point coordinate.

**OSTrack.** OSTrack is a single-object bounding-box tracker that unifies
template and search-region feature extraction and their relation modeling in
one transformer stream [@ye2022ostrack]. We used upstream revision `33b5e12`
and the ViT-B/256 MAE-CE configuration
`vitb_256_mae_ce_32x4_ep300` with its epoch-300 checkpoint
[@ostrackSoftware]. The initial 48 × 48 box was treated as the object
template. A finite predicted box with positive width and height defined an
available proposal; its arithmetic center, \((x+w/2,y+h/2)\), was passed to
the common ROI. The native box score was retained as a diagnostic and did not
gate or weight the Gaussian fit.

**DAM4SAM.** DAM4SAM adapts the SAM2.1 video-segmentation architecture to
visual object tracking by adding distractor-aware memory and an
introspection-based memory-update strategy [@videnovic2025dam4sam]. We used
upstream revision `9c95450` [@dam4samSoftware]. The Cu(111) and
controlled-translation experiments used `sam21pp-L` with the SAM2.1 Hiera
Large checkpoint. The independent energy scans used `sam21pp-B` with the
Hiera Base+ checkpoint so that the mask representative was also one member of
the matched SAM2-family experiment described in Section 3.2. DAM4SAM received
only the common initial box. An available proposal required a finite, nonempty
binary mask, and the arithmetic centroid of its true pixels was used as the
proposal coordinate. Native mask scores, areas, bounding boxes, and
connected-component counts were retained as diagnostics.

**Lucas--Kanade optical flow.** The classical Lucas--Kanade method estimates a
local image displacement from spatial intensity gradients
[@lucas1981iterative]. We used the TrackLEED pyramidal *skip-anchor*
implementation rather than a learned tracker. Its frozen settings were a
41 × 41 window, a maximum pyramid level of 4 (five levels including the base
image), at most 50 iterations with a convergence tolerance of 0.01, and a
minimum eigenvalue threshold of \(10^{-5}\). A match was accepted only if the
native residual did not exceed 30 and the forward--backward closure error did
not exceed 8 pixels. Normalized cross-correlation and centroid recentering
were disabled. On a rejected image, the result was recorded as missing and the
last successful image and position remained the anchor for a later attempt; no
coordinate was interpolated. The Cu(111) and translation experiments used the
common 256 × 256 grayscale cache.
For the energy scans, the frozen TrackLEED implementation operated on the
original image grid after its standard grayscale, high-pass, and white-top-hat
preprocessing. The accepted optical-flow point was the proposal coordinate.

**EasyLEED.** EasyLEED is a LEED-specific tracker designed for automatic
extraction of intensity--energy spectra [@mayer2012easyleed]. Its tracker
combines an energy-dependent radial motion model, a Kalman state containing
position and velocity, and an internal elliptical-Gaussian image observation.
If the internal observation is rejected, the Kalman prediction can remain
available for a later update. We used EasyLEED 2.5.2 at upstream revision
`bbe4acd` with one tracker per spot, a 12-pixel tracking radius in the method's
input grid, Gaussian initialization, window scaling enabled, and no manual
correction [@easyleedSoftware]. The exported Kalman \(x,y\) coordinate was the
proposal. The adapter retained the native EasyLEED intensity and tracking
radius and recorded its internal fit thresholds as diagnostics. In particular,
the EasyLEED internal Gaussian observation was not substituted for the common
TrackLEED Gaussian fit. The Cu(111) and translation runs used the common
256 × 256 cache, whereas the energy-scan adapter preserved every original
image and its identity using EasyLEED's native image loader.

Input resolution was method- and regime-specific rather than shared across the
panel. LocoTrack and OSTrack used 256 × 256 input throughout; DAM4SAM,
Lucas--Kanade, and EasyLEED used the regime-specific inputs stated above. This
family-native inference was upstream of a common coordinate system: every
proposal was returned to the original 1000 × 1000 grid before placement of the
identical raw-image ROI. Thus, the experiment controls the physical measurement
after proposal generation rather than claiming that the internal image
representations of heterogeneous trackers are identical.

### 3.2 Controlled SAM2-family panel

SAM2, SAMURAI, and DAM4SAM form a related family of promptable mask
propagators. SAM2 provides the reference streaming-memory architecture for
promptable video segmentation [@ravi2025sam2]. SAMURAI adapts SAM2 to
zero-shot visual object tracking by adding a Kalman motion model and
motion-aware mask and memory selection [@yang2024samurai]. DAM4SAM instead
uses distractor-aware memory and introspection-based updates
[@videnovic2025dam4sam].

The comparison used upstream revisions `2b90b9f` for SAM2, `76ba195` for
SAMURAI, and `9c95450` for DAM4SAM
[@sam2Software; @samuraiSoftware; @dam4samSoftware]. All three environments
loaded byte-identical `sam2.1_hiera_base_plus` weights with SHA-256
`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.
For every branch, the three methods received the same single-object
48 × 48 box at 187.9 eV, the same ordered images in each propagation
direction, and no subsequent prompt or manual reset. Each nonempty mask was
reduced to its pixel centroid and passed to the same 48 × 48 raw-image ROI,
Gaussian estimator, and physical-consistency gates.

This matched design controls the backbone weights, prompt information, input
sequence, proposal reduction, and downstream position measurement. It does
not reduce the three trackers to a one-line ablation: their motion prediction,
mask selection, visibility logic, and memory-update implementations are part
of the strategies being compared. The panel is therefore interpreted as a
mechanistic comparison of vanilla, motion-aware, and distractor-aware memory
strategies under a matched measurement contract, not as proof that one
isolated memory component caused every observed difference and not as a
general ranking of SAM2-derived trackers.

### 3.3 Domain-specific references

EasyLEED participated in the common five-strategy panel because one
independently initialized Kalman tracker yields one spot proposal and can
therefore enter the same single-spot contract as the four general computer
vision strategies. Its LEED-specific radial motion model makes it a
physics-informed domain baseline rather than another generic point tracker.

ViPErLEED has a different purpose and input contract. It identifies and tracks
a LEED pattern using reciprocal-lattice geometry, the expected
\(1/\sqrt{E}\) scaling, a global screen mapping, and local information from
neighboring spots, and is designed primarily for extraction and processing of
LEED \(I(V)\) curves [@schmid2025viperleed]. We used ViPErLEED ImageJ plugin
1.03 from revision `2ed1044` with the frozen 18-beam Ag(111) pattern and a
reciprocal-space-to-screen registration at 187.9 eV [@viperleedSoftware]. Its
whole-pattern output was reduced to the four coordinates corresponding to the
two frozen pairs and then passed to the same Gaussian measurement as the
EasyLEED coordinates.

Because ViPErLEED received global pattern information rather than one seed per
tracked spot, it was not inserted as a sixth row in the common strategy
matrix. The restricted high-energy comparison instead asks whether the
single-spot EasyLEED and whole-pattern ViPErLEED outputs place the intended
spots within the capture region of the same downstream estimator. It does not
use either output as external position ground truth.

**Table 2. Tracking methods and their reduction to the common measurement
contract.** Full source revisions, configuration hashes, checkpoint hashes,
and execution environments are retained in the frozen records linked at the
start of this section and will be reported in Table S1.

| Panel | Method | Propagated representation and frozen configuration | Initialization | Native availability/output | Proposal coordinate | Manual reset/correction | Study role |
|---|---|---|---|---|---|---|---|
| Main | LocoTrack | Point; Base offline | Common seed point | Point plus occlusion and expected-distance logits | Native point | Prohibited | General point tracker |
| Main | OSTrack | Box; ViT-B/256 MAE-CE, epoch 300 | Common 48 × 48 box | Finite positive-area box plus score | Box center | Prohibited | General bounding-box tracker |
| Main | DAM4SAM | Mask; `sam21pp-L` on Cu(111), `sam21pp-B` on energy scans | Common 48 × 48 box | Nonempty binary mask plus mask diagnostics | Mask centroid | Prohibited | General mask tracker with distractor-aware memory |
| Main | Lucas--Kanade | Point; pyramidal skip-anchor with forward--backward gate | Common seed point | Accepted optical-flow match or explicit missing value | Accepted point | Prohibited; last success remains anchor | Classical motion baseline |
| Main | EasyLEED 2.5.2 | Position; LEED motion model and Kalman filter | Common seed point, tracking radius, and pattern center | Kalman position plus native intensity and radius | Kalman \(x,y\) | Prohibited | LEED-domain single-spot baseline |
| Controlled family | SAM2.1 | Mask; Hiera Base+ | Common 48 × 48 box | Nonempty mask | Mask centroid | Prohibited | Vanilla streaming-memory control |
| Controlled family | SAMURAI | Mask; same Hiera Base+ weights | Common 48 × 48 box | Nonempty mask plus motion-aware selection | Mask centroid | Prohibited | Motion-aware memory strategy |
| Controlled family | DAM4SAM | Mask; `sam21pp-B`, same Hiera Base+ weights | Common 48 × 48 box | Nonempty mask plus distractor-aware diagnostics | Mask centroid | Prohibited | Distractor-aware memory strategy |
| External reference | ViPErLEED 1.03 | Whole-pattern LEED geometry | Frozen 18-beam pattern and screen registration | Pattern-associated spot coordinates | Coordinate of corresponding beam | Prohibited | Separate whole-pattern reference |

## 4. Validation protocol

### 4.1 Reporting principles

The validation separates four outcomes that answer different questions:
physics-consistent measurement coverage, recovery after natural I(V)
visibility gaps, accepted measurements that are inconsistent with the expected
branch geometry, and position or pair-distance residuals conditional on a
valid branch-consistent measurement. Every predefined opportunity remains in
its denominator when a tracker proposal is unavailable or a Gaussian fit is
rejected. Conditional residuals are reported only with their corresponding
coverage and are not used to replace failures with favorable surviving-image
statistics.

All results are descriptive. Counts and rates retain their integer
denominators, and continuous errors are summarized by medians, 95th
percentiles, and maxima where specified in the frozen artifacts. We do not
construct a composite score or claim an absolute-accuracy ranking. Successive
images within one propagated trajectory are repeated observations of the same
tracking event and are not treated as statistically independent replicates.
All per-image outputs were calculated automatically and no individual
proposal annotations were created. For the 701-image Cu(111) run, visual
review was limited to a frozen set of 24 combined overlays selected
independently of tracker output (12 temporal quantiles and 12
temperature-cycle anchors), rather than all 14,020 proposals. Visual review
was not used to correct a trajectory or change an outcome.
TODO[DECISION]: before submission, decide whether to add uncertainty intervals
at the trial, pair, or series level; image-level bootstrap intervals are excluded.

### 4.2 Controlled translations

One image from each of the four frozen Cu(111) states—the initial
low-temperature state, the first image of the five-image 800 °C plateau, the
post-plateau \(\Delta R/R\) minimum (maximum contraction), and the final
cooled state—was subjected to the closed translation path

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
completed Cu(111) comparison in that source image. Point-based methods
received the corresponding seed coordinates, box- or mask-based methods
received a symmetric 48 × 48 box derived from the same center, and EasyLEED
used the same center with its frozen tracking-radius and Kalman contract.
Seeds were inputs to the experiment and were not treated as ground truth. No
correction or reinitialization was allowed within a six-image trial.

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

Each of the five methods was initialized once on the first image with common
seeds for spots 1, 3, 5, and 6 and then propagated forward through the full
701-image sequence. Manual corrections and reinitialization were prohibited.
Every proposal was converted to the common 48 × 48 ROI and Gaussian
measurement described in Section 2. The primary endpoints were evaluated for
pairs 1–6 and 3–5 on all 701 images.

For each method and pair,
\(D_{ij,\mathrm{ref}}\) was the median of valid pair distances in the first
10 images, provided that at least 8 of the 10 distances were valid. If this
criterion was not met, the entire method–pair relative endpoint was defined
as unavailable. The thermal-cycle summaries were the median relative change
over the predefined five-image 800 °C plateau, the post-plateau
\(\Delta R/R\) minimum and its associated temperature, and the final-image
value.

Pair 2–4 was not included in the five-method run because spots 2 and 4 were
outside its frozen four-spot contract. It was retained as a separate
missing-measurement diagnostic based on an earlier saved YOLO proposal
trajectory. In that diagnostic, the proposal remained available while 24
consecutive Gaussian fits spanning the hottest part of the thermal cycle
failed only the common amplitude gate. These missing endpoints were not
interpolated. The diagnostic demonstrates that proposal availability and
quantitative measurement
availability are distinct; it is not evidence that all five methods behaved
identically for pair 2–4.

### 4.4 Independent Ag(111) energy-scan stress test

The independent energy-scan comparison used Scan A and Scan B, with two
disjoint frozen pairs per series, for four cases in total. Candidate spots
were detected automatically in the 188.5 eV reference image, refined by the
common Gaussian fit, combined into approximately diametric pairs, and frozen
before method comparison. The two branches of each pair were tracked as
independent single-object trajectories. From the reference image, propagation
was performed separately toward lower and higher energies, with one seed per
spot and no correction or reinitialization after a visibility gap.
Equal-energy terminal images were retained as distinct records by image
identifier.

A tracker-independent physics baseline assigned the evaluable status of each
branch and image. It used the frozen reference center and the expected radial
scaling

$$
\mathbf{p}_{\mathrm{pred}}(E)
=
\mathbf{c}+
\left[\mathbf{p}(E_{\mathrm{ref}})-\mathbf{c}\right]
\sqrt{\frac{E_{\mathrm{ref}}}{E}},
$$

the active detector annulus, and the common energy-scan Gaussian gates. An
accepted measurement was classified as branch-consistent when its distance
from \(\mathbf{p}_{\mathrm{pred}}(E)\) did not exceed the frozen tolerance
of 18 pixels. This tolerance was an evaluation gate and did not move the
tracker proposal or the fitted center.

Across the four cases, the baseline defined 221 visible spot opportunities
and 24 evaluable natural I(V) gaps for every method. Physics-consistent
measurement coverage was the number of baseline-visible opportunities with a
valid, branch-consistent Gaussian measurement divided by 221. Geometry
inconsistency was computed separately as the number of accepted Gaussian fits
more than 18 pixels from the expected branch divided by all accepted fits in
all visibility regimes. Pair-distance residuals were calculated only on
images in which both branches were valid and branch-consistent, and were
summarized as
\(\left|D_{ij}(E)-D_{ij,\mathrm{pred}}(E)\right|\).

One of the four frozen pairs, pair 1 in Scan A, did not pass the
auxiliary coherent-pairwise-energy-law check. It was retained as a difficult
case rather than removed after observing tracker results. The native
EasyLEED Kalman tracker also diverged once in Scan A pair 2; the
exception was retained as a failure and the tracker was not reset.

### 4.5 Controlled SAM2-family analysis

SAM2, SAMURAI, and DAM4SAM were evaluated on both frozen pairs of the
memory-analysis scan. Each branch received the same single 48 × 48 bounding-box
prompt at 187.9 eV and was propagated toward increasing energy and, in
reversed image order, toward decreasing energy. The three environments used
byte-identical SAM2.1 base-plus checkpoint weights. The methods differed in
their propagation, mask-selection, visibility, and memory mechanisms, while
the prompt, image sequence, mask-centroid proposal reduction, Gaussian
estimator, acceptance gates, and physical-consistency evaluation were fixed.

The same coverage, gap-recovery, and geometry-inconsistency definitions used
for the independent energy-scan comparison were applied. The pooled
denominators were 182 baseline-visible spot opportunities and 13 evaluable
natural I(V) gaps per method. Because the experiment contains one series and
two pairs, it is interpreted as a controlled mechanism comparison rather than
a general ranking of SAM2-family trackers.

### 4.6 Domain-specific external reference

The restricted EasyLEED–ViPErLEED comparison used the 15-image high-energy
interval of the memory-analysis scan and the same two pairs as the controlled
memory analysis. EasyLEED produced single-spot Kalman trajectories, whereas
ViPErLEED supplied a whole-pattern LEED trajectory. Their native outputs were
mapped to the same four branch coordinates and evaluated with the common
48 × 48 Gaussian contract.

EasyLEED was initialized once at 187.9 eV with one Kalman tracker at each of
the four frozen branch centers and a 12-pixel native tracking radius.
ViPErLEED was initialized once at the same reference image using the frozen
18-beam Ag(111) pattern and reciprocal-space-to-screen registration, after
which it tracked the complete pattern. Both methods processed the 15-image
segment toward non-decreasing energy without manual correction or
reinitialization, giving 60 predefined branch-image opportunities per method.
Image identity, rather than energy alone, was used as the record key so that
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
Recovery delay in image steps was retained. No manual reinitialization was
permitted.

An accepted Gaussian fit more than 18 pixels from its expected branch was
reported as a geometry-inconsistent continuation. This is an operational
indicator of possible false continuity: the tracker has maintained a
measurable image feature, but the feature is not compatible with the frozen
energy-dependent branch geometry. It is not, by itself, external
identity-ground-truth proof of a specific misassociation.

## 5. Results

### 5.1 Known translations isolate the measurement-chain behavior

LocoTrack, OSTrack, and DAM4SAM each returned a proposal and a valid common
Gaussian measurement for all 80 evaluated spot opportunities (Figure 2;
Table 3). Lucas--Kanade returned 76/80 proposals, all of which produced valid
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
spot-identity accuracy.

### 5.2 Cu(111) is a positive control, not a tracker ranking

All five strategies completed the Cu(111) positive control without a missing
proposal or rejected common fit: each method produced 2,804/2,804 Gaussian
measurements, and each primary pair produced 701/701 valid \(\Delta R/R\)
endpoints (Figure 5; Table 3). The largest separation between two native
method proposals was 16.759 px, whereas the largest separation between the
corresponding Gaussian-refined centers was 0.0502 px. At the endpoint level,
the maximum between-method difference over the entire sequence was only
0.001710 percentage points for pair 1--6 and 0.002237 percentage points for
pair 3--5.

The five methods also reproduced the same thermal-cycle features. Over the
predefined five-image 800 °C plateau, their median \(\Delta R/R\) estimates
ranged from \(-0.686676\%\) to \(-0.686244\%\) for pair 1--6 and from
\(-1.570833\%\) to \(-1.570300\%\) for pair 3--5. All methods placed the
post-plateau minimum of both pairs in the same image acquired at 797 °C. The
minimum values ranged from \(-0.726482\%\) to \(-0.726195\%\) for pair 1--6
and from \(-1.580123\%\) to \(-1.579795\%\) for pair 3--5. In the final
cooled image at 92 °C, pair 1--6 reached \(+0.121606\%\) to \(+0.121691\%\),
whereas pair 3--5 reached \(-0.152478\%\) to \(-0.152338\%\). The contraction
therefore largely reversed on cooling, but the two sectors retained different
endpoint values. With only two primary pairs, this sector difference was not
interpreted as crystallographic anisotropy or used to infer a thermal-expansion
coefficient or structural transition.

The separate saved-trajectory diagnostic for pair 2--4 produced 24 consecutive
missing measurements during the hottest part of the cycle despite a proposal
being present in every image. Repeating the common Gaussian fit showed that
all 24 candidates satisfied the \(R^2\), center, and width gates and failed
only the amplitude threshold. This result demonstrates that a missing
quantitative endpoint can arise from insufficient spot intensity rather than
loss of a tracking proposal. Pair 2--4 was not part of the five-method run,
and the diagnostic was not used to compare the five strategies.

### 5.3 Energy scans separate coverage, recovery, and geometric consistency

The independent Ag(111) energy scans separated the strategies in outcomes that
were absent from the Cu(111) positive control (Figure 3; Table 3). LocoTrack
had 217/221 branch-consistent measurements (98.19% coverage), followed by
OSTrack and DAM4SAM with 216/221 each (97.74%), EasyLEED with 214/221
(96.83%), and Lucas--Kanade with 188/221 (85.07%). The denominators were
identical and included every baseline-visible opportunity, so the 33
Lucas--Kanade measurements missing from the branch-consistent set were not
discarded before comparison.

Post-gap recovery produced a similar but non-identical ordering. LocoTrack and
DAM4SAM recovered 23/24 natural I(V) gaps (95.83%), EasyLEED recovered 22/24
(91.67%), OSTrack recovered 21/24 (87.50%), and Lucas--Kanade recovered 17/24
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
geometry gates, the median absolute pair-distance residuals occupied a narrow
range: 1.737 px for EasyLEED, 1.764 px for Lucas--Kanade, 1.770 px for
OSTrack, 1.788 px for DAM4SAM, and 1.837 px for LocoTrack. In particular,
Lucas--Kanade had a conditional residual similar to the other methods while
losing substantially more visible measurements and gap recoveries. Conditional
distance quality therefore did not summarize measurement-chain robustness;
coverage, recovery, and geometry-inconsistent continuation remained separate
outcomes.

### 5.4 Memory-aware strategies show a continuity--consistency trade-off

Under byte-identical SAM2.1 Base+ weights, vanilla SAM2 produced 126/182
branch-consistent measurements (69.23% coverage) and recovered 9/13 natural
I(V) gaps (69.23%; Figure 4). SAMURAI increased coverage to 181/182 (99.45%)
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

### 5.5 Domain-specific external reference

EasyLEED and ViPErLEED each produced all 60 expected branch proposals over the
15-image high-energy interval, despite using single-spot and whole-pattern
initialization contracts, respectively (Figure 3f). For both methods, the
common Gaussian estimator accepted the same 38/60 branch measurements with
the same per-branch counts: 13/15 and 7/15 for the two branches of pair 1, and
13/15 and 5/15 for pair 2. Both methods also passed all four predefined
absence-to-recovery checks.

Across the 60 corresponding native proposals, the EasyLEED--ViPErLEED
separation had a median of 0.626 px and a maximum of 1.696 px. For the 38
jointly accepted common measurements, the separation between
Gaussian-refined centers fell to a median of 0.000986 px and a maximum of
0.0240 px. At the two terminal images acquired at 220.2 eV, the returning
branch of each pair yielded a valid fit for both methods, while the opposite
branch failed the Gaussian gate. These events established branch recovery but
did not provide a terminal pair-distance measurement.

The agreement after common refinement shows that both heterogeneous
domain-specific outputs placed the corresponding high-energy spot features
inside the capture region of the same estimator. It does not show equivalence
outside this restricted interval, provide external position ground truth, or
establish superiority of either the single-spot or whole-pattern contract.

## 6. Discussion

### 6.1 Tracker performance is mediated by the estimator capture region

Connect the controlled translation and Cu(111) results. Explain why heterogeneous
native proposals can converge after a common Gaussian fit without claiming that
the trackers are equally accurate in an absolute sense.

### 6.2 Coverage and identity failures cannot be replaced by conditional error

Use the independent energy-scan results to distinguish:

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

- No external image-by-image ground truth for absolute spot identity or position.
- Controlled translations test equivariance, not experimental dynamics.
- Cu(111) is an easy positive-control regime and does not discriminate trackers.
- The independent energy-scan validation contains four primary pairs across
  two series.
- The SAM2-family experiment contains one series and two pairs.
- DAM4SAM uses `sam21pp-L` on Cu(111) and `sam21pp-B` in the independent energy-scan
  comparison.
- ViPErLEED uses a different whole-pattern initialization contract.
- No full structural I(V) analysis, thermal-expansion coefficient, or material
  transformation is inferred.

## 7. Reproducibility and data availability

Describe:

- immutable raw-data inventory and hashes;
- native tracker payloads and frozen configurations;
- per-image proposals, Gaussian measurements, fit diagnostics, and pair endpoints;
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
