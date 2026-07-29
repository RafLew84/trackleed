# Figures and tables

The display strategy follows the evidence ladder. Each figure should answer one
scientific question and preserve denominators; no panel should imply a composite
tracker ranking.

## Main figures

### Figure 1 - Measurement chain and evidence ladder

**Question:** What is evaluated, and where is the physical position measured?

**Status:** generated and validated.

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

Final artifact:

- [artifact README and caption](figures/figure01_measurement_chain/README.md)
- [PNG](figures/figure01_measurement_chain/output/figure01_measurement_chain.png)
- [PDF](figures/figure01_measurement_chain/output/pdf/figure01_measurement_chain.pdf)
- [render manifest](figures/figure01_measurement_chain/render_manifest.json)

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

**Status:** generated and validated.

Proposed panels:

- **a.** Expected energy-law trajectory with framewise cross-method median
  pair distances, min-max whiskers, and the visibility regimes of both
  branches.
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

Final artifact:

- [artifact README and caption](figures/figure03_independent_scane_stress/README.md)
- [PNG](figures/figure03_independent_scane_stress/output/figure03_independent_scane_stress.png)
- [PDF](figures/figure03_independent_scane_stress/output/pdf/figure03_independent_scane_stress.pdf)
- [manifest](figures/figure03_independent_scane_stress/manifest.json)

### Figure 4 - Controlled SAM2-family memory trade-off

**Question:** What changes when the memory strategy changes under identical
checkpoint weights?

**Status:** generated and validated.

Proposed panels:

- **a.** One representative visibility gap with SAM2, SAMURAI, and DAM4SAM
  proposals and common Gaussian centers.
- **b.** Coverage for the two frozen pairs.
- **c.** Total and zero-delay gap recovery.
- **d.** Geometry-inconsistent accepted fits and conditional pair-distance
  residual `D(E)` (median and P95 with their conditional denominator).
- **e.** Conceptual summary: continuity benefit versus false-persistence risk.

Primary source:

- [SAM2-family trade-off table](../.scratch/trackleed-physics-method-paper/artifacts/cross-regime-evidence-matrix/sam2_family_tradeoff.csv)

Caption limitation: one series and two pairs; mechanism evidence, not a general
ranking.

Frozen source contract:

- [artifact README](figures/figure04_controlled_sam2_memory_tradeoff/README.md)
- [source contract](figures/figure04_controlled_sam2_memory_tradeoff/contract.json)
- [frozen pair seeds](figures/figure04_controlled_sam2_memory_tradeoff/source_data/pair_seeds.csv)
- [source provenance](figures/figure04_controlled_sam2_memory_tradeoff/source_data/source_provenance.json)

Final artifact:

- [artifact README and caption](figures/figure04_controlled_sam2_memory_tradeoff/README.md)
- [PNG](figures/figure04_controlled_sam2_memory_tradeoff/output/figure04_controlled_sam2_memory_tradeoff.png)
- [PDF](figures/figure04_controlled_sam2_memory_tradeoff/output/pdf/figure04_controlled_sam2_memory_tradeoff.pdf)
- [render manifest](figures/figure04_controlled_sam2_memory_tradeoff/manifest.json)

### Figure 5 - Propagation to the Cu(111) physical endpoint

**Question:** Do tracker-dependent proposal differences survive the common
Gaussian refinement sufficiently to change pairwise `Delta R/R`?

**Status:** generated and validated.

Final panels:

- **a.** Temperature profile and frozen reference/plateau/post-plateau
  `Delta R/R` minimum/final windows.
- **b.** Pair `1-6`: cross-method median `Delta R/R`, min-max envelope,
  and magnified method-minus-median residuals.
- **c.** Pair `3-5`: the same representation on the shared residual scale.
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

Frozen source contract:

- [artifact README and draft caption](figures/figure05_cu111_physical_endpoint/README.md)
- [source contract](figures/figure05_cu111_physical_endpoint/contract.json)
- [five-method run contract](figures/figure05_cu111_physical_endpoint/source_data/five_method_run_contract.json)
- [thermal-cycle landmarks](figures/figure05_cu111_physical_endpoint/source_data/cycle_landmarks.csv)
- [five-method endpoint trajectories](figures/figure05_cu111_physical_endpoint/source_data/pair_endpoint_timeseries.csv)
- [endpoint summaries](figures/figure05_cu111_physical_endpoint/source_data/endpoint_summaries.csv)
- [between-method spread](figures/figure05_cu111_physical_endpoint/source_data/between_method_spread.csv)
- [separate pair 2-4 diagnostic](figures/figure05_cu111_physical_endpoint/source_data/pair_2_4_missingness_diagnostic.csv)
- [source provenance](figures/figure05_cu111_physical_endpoint/source_data/source_provenance.json)

Final artifact:

- [artifact README and caption](figures/figure05_cu111_physical_endpoint/README.md)
- [PNG](figures/figure05_cu111_physical_endpoint/output/figure05_cu111_physical_endpoint.png)
- [PDF](figures/figure05_cu111_physical_endpoint/output/pdf/figure05_cu111_physical_endpoint.pdf)
- [render manifest](figures/figure05_cu111_physical_endpoint/manifest.json)

## Main tables

### Table 1 - Data and evidence roles

**Status:** generated and validated.

Columns:

- scientific series label;
- control variable and range;
- image count;
- frozen spots/pairs;
- evidence role;
- primary endpoint;
- independence/replication unit.

The five rows represent evidence roles, not five independent datasets.
Controlled translations are derived from the Cu(111) sequence, and the
restricted high-energy external reference is a subset of the memory-analysis
scan with a separate initialization contract and denominator.

Final artifact:

- [artifact README](tables/table01_data_evidence_roles/README.md)
- [publication-facing CSV](tables/table01_data_evidence_roles/output/table01_data_evidence_roles.csv)
- [manuscript Markdown](tables/table01_data_evidence_roles/output/table01_data_evidence_roles.md)
- [manifest](tables/table01_data_evidence_roles/manifest.json)

### Table 2 - Tracking methods and common output contract

**Status:** generated and validated.

Inserted in Section 3 of
[the working manuscript](manuscript.md#33-domain-specific-references).
Columns:

- panel;
- method;
- family/representation;
- checkpoint or algorithm configuration;
- initialization;
- native availability/output;
- reduction to proposal coordinate;
- reset policy;
- role in the study.

The table contains five main comparison rows, three controlled SAM2-family
rows, and one visibly separate ViPErLEED external-reference row. The three
controlled-family rows retain the same byte-identical SAM2.1 Hiera Base+
checkpoint hash; exact revisions, configurations, and artifact hashes remain
in the audit fields and manifest. A method-level executed-configuration hash
is reported only where one was recorded; whole-contract hashes are retained
solely as source-provenance hashes.

Final artifact:

- [artifact README](tables/table02_tracking_methods_contract/README.md)
- [publication-facing CSV](tables/table02_tracking_methods_contract/output/table02_tracking_methods_contract.csv)
- [manuscript Markdown](tables/table02_tracking_methods_contract/output/table02_tracking_methods_contract.md)
- [manifest](tables/table02_tracking_methods_contract/manifest.json)

### Table 3 - Quantitative summary across evidence regimes

**Status:** generated and validated.

Use selected columns from:

- [cross-regime method evidence matrix](../.scratch/trackleed-physics-method-paper/artifacts/cross-regime-evidence-matrix/method_evidence_matrix.csv)

Group headers:

- controlled translation;
- Cu(111) positive control;
- independent Ag(111) energy-scan stress test.

The publication table uses three vertically stacked blocks rather than one
over-wide matrix. Each block retains its own denominators and metric semantics,
including the surviving pair-frame count for the conditional energy-scan
residual. The DAM4SAM checkpoint-size change between regimes is disclosed in
the caption.

Do not add an overall score, rank column, or color scale suggesting one.

Final artifact:

- [artifact README](tables/table03_cross_regime_summary/README.md)
- [publication-facing CSV](tables/table03_cross_regime_summary/output/table03_quantitative_summary.csv)
- [manuscript Markdown](tables/table03_cross_regime_summary/output/table03_quantitative_summary.md)
- [manifest](tables/table03_cross_regime_summary/manifest.json)

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
- **Table S3:** per-scan and per-pair energy-scan coverage, recovery, geometry
  inconsistency, and conditional residuals with their surviving
  denominators; leave-one-case-out ranges for count-based outcomes are a
  sensitivity analysis, not confidence intervals.
- **Table S4:** Cu plateau, post-plateau `Delta R/R` minimum, and final endpoint
  values by method and pair.
- **Table S5:** per-source-trial controlled-translation proposal and
  measurement coverage, conditional median/P95/maximum displacement error,
  pair coverage, and maximum absolute `Delta D/D`.

The frozen source tables, deterministic builder, and rendered drafts for
Tables S3 and S5 are documented in the
[stratified descriptive reporting artifact](tables/stratified_uncertainty/README.md).

## Asset policy

- Generate final figures from CSV/JSON sources, not screenshots of exploratory
  notebooks.
- Store each final figure together with its source-data table and build script.
- Record dimensions, font sizes, color mapping, and output hash.
- Use a colorblind-safe method palette consistently across all figures.
- Show missingness explicitly; never connect across rejected fits without a marked
  gap.
