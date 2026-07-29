# Figures and tables

The display strategy follows the evidence ladder. Each figure should answer one
scientific question and preserve denominators; no panel should imply a composite
tracker ranking.

## Main figures

### Figure 1 - Measurement chain and evidence ladder

**Question:** What is evaluated, and where is the physical position measured?

**Status:** representative Cu(111) case frozen; final rendering pending.

Proposed panels:

- **a.** Heterogeneous native outputs: point, bounding box, mask, optical-flow
  point, and LEED-domain Kalman position.
- **b.** Common conversion: tracking proposal -> `48 x 48` ROI -> elliptical
  Gaussian-refined center.
- **c.** Pairwise endpoint: two refined centers -> `D` -> `Delta D/D`.
- **d.** Four evidence roles: Cu positive control, known translations, Ag(111)
  energy-scan stress test, and SAM2-family memory analysis.

Source: panels a-c assembled from the frozen representative image, empirical
native geometry, and measurement table; panel d is a contract-level schematic.
Primary claim: the tracker proposes where to measure; it does not directly define
the final quantitative position.

Frozen representative source case:

- [case README](figures/figure01_measurement_chain/README.md)
- [contract](figures/figure01_measurement_chain/contract.json)
- [shared initialization](figures/figure01_measurement_chain/source_data/representative_initialization.json)
- [five-method source table](figures/figure01_measurement_chain/source_data/representative_method_outputs.csv)
- [empirical native geometry](figures/figure01_measurement_chain/source_data/representative_native_geometry.json)
- [DAM4SAM native mask](figures/figure01_measurement_chain/source_data/representative_dam4sam_mask_cache.png)

The case freezes the same 701-image sequence, spot identity, Gaussian seed
center, and shared 48 x 48 prompt box for all five methods; no correction or
reinitialization is allowed.

A targeted full-sequence replay supplies the empirical OSTrack bbox and
DAM4SAM mask for this frozen case and reproduces their archived reduced
trajectories over all 701 images. Panel a must render these empirical outputs
from the frozen coordinates and pixels, without smoothing, redrawing, or
substituting schematic geometry.

### Figure 2 - Controlled equivariance and energy-scan geometry

**Question:** Does the complete measurement chain recover a known displacement,
and why is pair distance used for the energy scans?

**Status:** generated and validated.

Proposed panels:

- **a.** Closed six-step translation path over one representative Cu image.
- **b.** Native-proposal versus Gaussian-refined displacement error.
- **c.** Gaussian measurement coverage for the five trackers.
- **d.** Maximum absolute pairwise `Delta D/D` under common translation.
- **e.** Energy-scan pair geometry showing `D(E)` and midpoint drift as separate
  quantities.

Primary sources:

- [equivariance results](../.scratch/trackleed-physics-method-paper/artifacts/controlled-translation-chain/equivariance_results.csv)
- [pair invariance](../.scratch/trackleed-physics-method-paper/artifacts/controlled-translation-chain/pair_invariance.csv)
- [controlled-translation method summary](../.scratch/trackleed-physics-method-paper/artifacts/controlled-translation-chain/method_summary.csv)
- [energy-scan pair geometry](../.scratch/trackleed-physics-method-paper/prototypes/scane_tracer/output/pair_measurements.csv)

Final artifact:

- [artifact README](figures/figure02_controlled_equivariance/README.md)
- [PNG](figures/figure02_controlled_equivariance/output/figure02_controlled_equivariance.png)
- [PDF](figures/figure02_controlled_equivariance/output/pdf/figure02_controlled_equivariance.pdf)
- [manifest](figures/figure02_controlled_equivariance/manifest.json)

Required caption limitation: known translations test equivariance, not
experimental dynamics or absolute spot identity.

### Figure 3 - Five-strategy Ag(111) energy-scan stress test

**Question:** How do strategies differ under radial motion and true I(V)
visibility gaps?

Proposed panels:

- **a.** Representative pair trajectories in energy, with visible regimes and
  gaps marked.
- **b.** Branch-consistent Gaussian measurement coverage, common denominator
  `221`.
- **c.** Post-gap recovery, common denominator `24`.
- **d.** Geometry-inconsistent accepted fits over all accepted measurements.
- **e.** Conditional median absolute pair-distance residual,
  `|D_measured(E) - D_expected(E)|`, visually marked as conditional.
- **f.** Visually separated external-reference inset for the frozen
  EasyLEED/ViPErLEED high-energy comparison. Label its different whole-pattern
  initialization contract and do not present ViPErLEED as a row in the common
  five-method matrix.

Primary source:

- [independent energy-scan aggregate results](../.scratch/trackleed-physics-method-paper/artifacts/independent-scane-validation/aggregate_method_results.csv)
- [EasyLEED/ViPErLEED external reference](../.scratch/trackleed-physics-method-paper/artifacts/domain-baseline-high-energy/README.md)

Caption requirement: coverage and recovery must appear before the conditional
residual; the residual cannot be presented as an overall accuracy ranking. The
external-reference inset uses a separate contract and denominator and is not part
of the five-method ranking.

### Figure 4 - Controlled SAM2-family memory trade-off

**Question:** What changes when the memory strategy changes under identical
checkpoint weights?

Proposed panels:

- **a.** One representative visibility gap with SAM2, SAMURAI, and DAM4SAM
  proposals and common Gaussian centers.
- **b.** Coverage for the two frozen pairs.
- **c.** Gap recovery.
- **d.** Geometry-inconsistent accepted fits.
- **e.** Conceptual summary: continuity benefit versus false-persistence risk.

Primary source:

- [SAM2-family trade-off table](../.scratch/trackleed-physics-method-paper/artifacts/cross-regime-evidence-matrix/sam2_family_tradeoff.csv)

Caption limitation: one series and two pairs; mechanism evidence, not a general
ranking.

### Figure 5 - Propagation to the Cu(111) physical endpoint

**Question:** Do tracker differences materially change pairwise `Delta R/R`?

Proposed panels:

- **a.** Temperature profile and frozen reference/plateau/post-plateau
  `Delta R/R` minimum/final windows.
- **b.** `Delta R/R` for pair `1-6`, five methods.
- **c.** `Delta R/R` for pair `3-5`, five methods.
- **d.** Between-method spread at the plateau, post-plateau `Delta R/R`
  minimum, and final image.
- **e.** Separate saved-YOLO-trajectory diagnostic for pair `2-4`: 24
  amplitude-gate rejections despite available YOLO proposals. Mark this panel as
  outside the five-method run.

Primary sources:

- [five-method Cu pair endpoints](../.scratch/trackleed-physics-method-paper/artifacts/cu111-tracker-evaluation-subset/runs/five-methods-v1/pair_endpoints.csv)
- [Cu pairwise endpoint artifact](../.scratch/trackleed-physics-method-paper/artifacts/cu111-pairwise-delta-r/README.md)

Caption limitations: the two primary pairs do not establish anisotropy; missing
pair `2-4` measurements are not automatically tracker failures. Pair `2-4` uses
the earlier saved YOLO proposals and must not imply that the five evaluated
strategies were run on spots 2 and 4.

## Main tables

### Table 1 - Data and evidence roles

Columns:

- scientific series label;
- control variable and range;
- image count;
- frozen spots/pairs;
- evidence role;
- primary endpoint;
- independence/replication unit.

### Table 2 - Tracking methods and common output contract

Drafted in Section 3 of
[the working manuscript](manuscript.md#33-domain-specific-references).
Columns:

- method;
- family/representation;
- checkpoint or algorithm configuration;
- initialization;
- native output;
- reduction to proposal coordinate;
- reset policy;
- role in the study.

Keep ViPErLEED in a visibly separate external-reference row.

### Table 3 - Quantitative summary across evidence regimes

Use selected columns from:

- [cross-regime method evidence matrix](../.scratch/trackleed-physics-method-paper/artifacts/cross-regime-evidence-matrix/method_evidence_matrix.csv)

Group headers:

- controlled translation;
- Cu(111) positive control;
- independent Ag(111) energy-scan stress test.

Do not add an overall score, rank column, or color scale suggesting one.

## Supplementary display items

- **Figure S1:** all 24 frozen Cu diagnostic overlays.
- **Figure S2:** per-series and per-pair energy-scan trajectories and visibility
  regimes.
- **Figure S3:** native proposal error versus Gaussian-refined error for every
  controlled translation.
- **Figure S4:** EasyLEED/ViPErLEED high-energy external-reference overlays.
- **Figure S5:** accepted but geometry-inconsistent continuation examples.
- **Table S1:** full versions, hashes, checkpoints, and environments.
- **Table S2:** Gaussian fit gates and all validation denominators.
- **Table S3:** per-pair energy-scan coverage, recovery, and conditional
  residuals.
- **Table S4:** Cu plateau, post-plateau `Delta R/R` minimum, and final endpoint
  values by method and pair.

## Asset policy

- Generate final figures from CSV/JSON sources, not screenshots of exploratory
  notebooks.
- Store each final figure together with its source-data table and build script.
- Record dimensions, font sizes, color mapping, and output hash.
- Use a colorblind-safe method palette consistently across all figures.
- Show missingness explicitly; never connect across rejected fits without a marked
  gap.
