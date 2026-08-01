# Claim-evidence map

This file controls claim strength. A statement should not enter the abstract,
results, or conclusions unless its evidence and limitation are recorded here.

| ID | Permitted claim | Primary evidence | Quantitative anchor | Required limitation | Status |
|---|---|---|---|---|---|
| C1 | Across the tested data regimes and predeclared measurement contracts, the common Gaussian estimator can suppress method-specific proposal differences when the intended spot remains inside its capture region. | Controlled translations; five-method Cu(111) run; all 15 predeclared sensitivity variants | Tracker median full-chain error `0.0043-0.0265 px`; oracle `0.0053 px`; Cu maximum Gaussian-center separation `0.0502 px` despite native separation `16.759 px`; central variant reproduced and width trigger not activated | This is measurement-chain equivariance/agreement within the tested contracts, not instrument-traceable absolute tracking accuracy or identity ground truth. | Narrowed |
| C2 | In the recorded Cu(111) positive-control sequence, stable tracking yields mutually consistent pairwise Delta R/R across five heterogeneous strategies after common Gaussian refinement. | Five-method Cu(111) run | `2804/2804` fits per method; `701/701` endpoints per primary pair; maximum between-method `Delta R/R` difference `0.001710` and `0.002237` percentage points | Cu(111) is a positive control and does not resolve an overall tracker ranking or calibrated transition. | Preserved |
| C3 | Across the two recorded Ag(111) energy sequences, the four case-based stress-test strata differentiate strategies through branch-consistent coverage, post-gap recovery, and geometry-inconsistent continuation under the tested measurement contracts. | Independent Ag(111) energy-scan validation: Scan A and Scan B; all 15 predeclared sensitivity variants | Central coverage `188-217/221` and recovery `17-23/24`; all-variant envelopes `170-221/221` and `17-24/24` | The two scans are case series, not controlled population replicates; conditional residuals apply only to jointly consistent pair images and do not support calibrated structural I(V) or kinetics. | Narrowed |
| C4 | Under the frozen central contract, similar conditional pair-distance residuals do not imply similar measurement-chain robustness; across all predeclared variants, conditional residuals remain inseparable from their method- and variant-specific pair-frame n and fixed-denominator coverage. | Independent Ag(111) energy-scan validation; all 15 predeclared sensitivity variants | Central median residuals `1.7367102643215162-1.8366454823238882 px`; all-variant envelope `1.4988461207997261-1.9386438331731597 px`; Lucas–Kanade has the lowest coverage in all 15 variants | Do not detach residuals from their coverage denominators or generalize central residual similarity to every contract. | Narrowed |
| C5 | Under identical SAM2.1 weights in the one-series, two-pair mechanism experiment, memory-aware propagation improves continuity while increasing geometry-inconsistent accepted fits. | Controlled SAM2-family experiment on the Ag(111) memory-analysis scan | Coverage `126/182` for SAM2 versus `181/182` SAMURAI and `179/182` DAM4SAM; inconsistent fits `8/149` versus `36/236` and `36/233` | Supporting mechanism test with one series and two pairs; not a general memory-method ranking or preparation-state attribution. | Preserved |
| C6 | EasyLEED and ViPErLEED can be placed under the same downstream Gaussian measurement contract despite different native contracts in the frozen high-energy compatibility check. | Frozen high-energy domain-baseline comparison | Both `38/60` accepted fits; median native separation `0.626 px`; Gaussian-center separation `0.000986 px` | External-reference compatibility check with a separate contract and denominator, not proof of equivalent identity tracking, absolute accuracy, or superiority. | Preserved |
| C7 | Pairwise distance cancels common translation of the observed pattern but does not remove nonrigid distortion, misassociation, sector-specific behavior, or detector-calibration limitations. | Pair geometry definition; Ag(111) memory-analysis scan geometry artifact | `D(E)=||p_i(E)-p_j(E)||`; midpoint retained separately | It does not remove nonrigid distortion, misassociation, sector-specific behavior, apparatus projection, or detector-calibration limitations. | Preserved |
| C8 | In the recorded temperature-indexed Cu(111) positive-control sequence, the pairwise physical endpoint changes substantially and largely returns; the result does not establish kinetics, hysteresis, a calibrated thermal-expansion coefficient, a phase-transition temperature, or crystallographic anisotropy. | Cu(111) pairwise endpoint artifact | Plateau medians `-0.6868%` and `-1.5703%`; final values `+0.1217%` and `-0.1523%` | Incomplete temperature calibration, timing, preparation, and pressure metadata require a descriptive temperature-indexed interpretation; do not attribute a preparation state. | Narrowed |
| C9 | In the separate saved-proposal Cu pair 2-4 diagnostic, proposal availability does not guarantee an accepted quantitative position: all 24 consecutive missing fits fail only the common uncalibrated amplitude gate. | Separate Cu(111) pair 2–4 missingness diagnostic in the pairwise endpoint artifact | A saved proposal is present throughout; all `24/24` missing fits pass the center, width, and `R²` gates and fail only the common amplitude gate | Gaussian amplitude is not calibrated absolute intensity; the saved earlier-YOLO trajectory is outside the five-method run and is neither a comparison nor spot-identity ground truth. | Narrowed |

## Final adjudication checkpoint

The metadata- and sensitivity-constrained wording for every claim is recorded
in the [adjudication contract](claim_adjudication/adjudication_contract.json).
The proposed decisions are `4 preserved`, `5 narrowed`, and `0 removed`.
Generated numerical anchors and the boundary-bound author checkpoint are in the
[claim-adjudication output](claim_adjudication/output/). The publication author
confirmed this final boundary in the repository HITL checkpoint on 2026-08-01.

The global boundary is that all claims apply only under the tested measurement
contracts and data regimes. The evidence establishes reproducibility and
physical self-consistency of the measurement chain, not external absolute
tracking accuracy, spot-identity ground truth, a universal method ranking,
calibrated structural LEED I(V), or instrument-traceable detector geometry.

## Prohibited or restricted claims

Do not state that:

- this work introduces the first automated LEED spot tracker;
- any method is absolutely the most accurate without an external reference;
- controlled translations establish spot-identity ground truth;
- the `1/sqrt(E)` contraction law is a new physical discovery;
- a missing Gaussian fit is automatically a tracker failure;
- tracker confidence, mask area, or visibility is a physical I(V) observable;
- the Cu(111) result determines a thermal-expansion coefficient, phase
  transformation, or crystallographic anisotropy;
- the SAM2-family prototype establishes a general ranking;
- the paper evaluates the TrackLEED GUI or its complete software feature set.

## Evidence update rule

If any source artifact changes:

1. rerun its validator;
2. update the affected numerical anchor here;
3. update the corresponding result subsection and display item;
4. check the abstract and conclusions for stale or stronger wording.
