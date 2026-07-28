# Claim-evidence map

This file controls claim strength. A statement should not enter the abstract,
results, or conclusions unless its evidence and limitation are recorded here.

| ID | Permitted claim | Primary evidence | Quantitative anchor | Required limitation | Status |
|---|---|---|---|---|---|
| C1 | A common Gaussian estimator suppresses method-specific proposal differences when the intended spot remains inside its capture region. | Controlled translations; five-method Cu(111) run | Tracker median full-chain error `0.0043-0.0265 px`; oracle `0.0053 px`; Cu maximum Gaussian-center separation `0.0502 px` despite native separation `16.759 px` | This is measurement-chain equivariance/agreement, not absolute tracking accuracy or identity ground truth. | Supported |
| C2 | Stable Cu(111) tracking yields mutually consistent pairwise `Delta R/R` across five heterogeneous strategies. | Five-method Cu(111) run | `2804/2804` fits per method; `701/701` endpoints per primary pair; maximum between-method `Delta R/R` difference `0.001710` and `0.002237` percentage points | Cu(111) is a positive control and does not resolve an overall tracker ranking. | Supported |
| C3 | Energy-driven motion and natural visibility gaps differentiate strategies mainly through coverage, recovery, and false continuity. | Independent Ag(111) energy-scan validation: Scan A and Scan B | Common denominators `221` visible opportunities and `24` gaps; coverage `188-217`; recovery `17-23` | Four pairs across two series; conditional residuals apply only to jointly consistent pair images. | Supported |
| C4 | Similar conditional distance residuals do not imply similar measurement robustness. | Independent Ag(111) energy-scan validation | Median residuals `1.737-1.837 px`, while Lucas-Kanade coverage is `188/221` versus `214-217/221` for the strongest-coverage methods | Do not detach residuals from their coverage denominators. | Supported |
| C5 | Under identical SAM2.1 weights, memory-aware propagation improves continuity but increases geometry-inconsistent accepted fits. | Controlled SAM2-family experiment on the Ag(111) memory-analysis scan | Coverage `126/182` for SAM2 versus `181/182` SAMURAI and `179/182` DAM4SAM; inconsistent fits `8/149` versus `36/236` and `36/233` | Supporting mechanism test with one series and two pairs; not a general memory-method ranking. | Supported |
| C6 | EasyLEED and ViPErLEED can be placed under the same downstream Gaussian measurement contract despite different native contracts. | Frozen high-energy domain-baseline comparison | Both `38/60` accepted fits; median native separation `0.626 px`; Gaussian-center separation `0.000986 px` | External-reference comparison, not proof of equivalent identity tracking or superiority. | Supported |
| C7 | Pairwise distance avoids contamination by common translation of the observed pattern. | Pair geometry definition; Ag(111) memory-analysis scan geometry artifact | `D(E)=||p_i(E)-p_j(E)||`; midpoint retained separately | It does not remove nonrigid distortion, misassociation, or sector-specific behavior. | Analytical |
| C8 | The Cu(111) physical endpoint changes substantially during the thermal cycle and largely returns. | Cu(111) pairwise endpoint artifact | Plateau medians `-0.6868%` and `-1.5703%`; final values `+0.1217%` and `-0.1523%` | Do not infer a thermal-expansion coefficient, structural transition, or crystallographic anisotropy from two pairs. | Supported |
| C9 | Tracking-proposal availability does not guarantee an accepted quantitative position measurement. | Separate Cu(111) pair 2–4 missingness diagnostic in the pairwise endpoint artifact | A saved proposal is present throughout; all `24/24` missing fits pass the center, width, and `R²` gates and fail only the common amplitude gate | The saved trajectory used earlier YOLO proposals, lies outside the five-method run, and is neither a comparison of the five strategies nor spot-identity ground truth. | Supported |

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
