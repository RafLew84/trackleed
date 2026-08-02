# Final publication display inventory

This file is generated from the validated cross-regime inventory. It contains exactly five main figures, two main tables, Figures S1-S6, and Tables S1-S7. Historical artifact directories are not final display placements.

## Shared rendering contract

- Missing values: Missing or rejected measurements remain absent and are never interpolated or connected across gaps; fixed denominators retain every predefined opportunity.
- Method palette: dam4sam `#009E73`, easyleed `#D55E00`, locotrack `#0072B2`, lucas_kanade `#CC79A7`, ostrack `#E69F00`.
- Rounding rules: Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n. Diagram labels and exact counts are not converted into rounded estimates. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places. Counts and denominators are shown as exact integers in numerator/denominator form. Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally. Rates derived from exact counts are shown to 2 decimal places. Controlled-translation displacement errors are shown to 4 decimal places.

## Main figures

### Figure 1 — Measurement chain and evidence ladder

Defines native proposal, shared Gaussian refinement, pairwise observable, and the evidence ladder before quantitative comparisons.

- Bundle: [figures/figure01_measurement_chain](figures/figure01_measurement_chain/README.md)
- Caption/source display: [open](figures/figure01_measurement_chain/README.md)
- Provenance: [manifest](figures/figure01_measurement_chain/manifest.json), [manifest](figures/figure01_measurement_chain/render_manifest.json)
- Denominators: Conceptual display; downstream fixed and conditional denominators are defined in Tables 1 and S2.
- Rounding: Diagram labels and exact counts are not converted into rounded estimates. Counts and denominators are shown as exact integers in numerator/denominator form.
- Missing values: Unavailable native outputs are represented as unavailable, not imputed.
- Palette: Native-output geometries use semantic encodings rather than a performance rank.

### Figure 2 — Controlled estimator behavior

Shows controlled translation paths, proposal-to-measurement behavior, measurement coverage, and pair invariance.

- Bundle: [figures/figure02_controlled_equivariance](figures/figure02_controlled_equivariance/README.md)
- Caption/source display: [open](figures/figure02_controlled_equivariance/README.md)
- Provenance: [manifest](figures/figure02_controlled_equivariance/manifest.json)
- Denominators: Initialization states are separated from 80 displaced-spot and 40 displaced-pair evaluation opportunities per method.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Controlled-translation displacement errors are shown to 4 decimal places. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places.
- Missing values: Rejected or unavailable measurements remain gaps and coverage is reported first.
- Palette: Uses the frozen five-method colorblind-safe palette.

### Figure 3 — Independent Ag(111) energy-scan stress test

Reports branch-consistent coverage, post-gap branch-consistent recovery, geometry inconsistency, and conditional residual in denominator-first order.

- Bundle: [figures/figure03_independent_scane_stress](figures/figure03_independent_scane_stress/README.md)
- Caption/source display: [open](figures/figure03_independent_scane_stress/README.md)
- Provenance: [manifest](figures/figure03_independent_scane_stress/manifest.json)
- Denominators: Uses 221 visible opportunities, 24 evaluable gaps, accepted-fit denominators, and method-specific residual n.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n.
- Missing values: Visibility gaps and rejected measurements are shown explicitly and are not connected.
- Palette: Uses the frozen five-method colorblind-safe palette.

### Figure 4 — Controlled SAM2-family continuity–consistency trade-off

Provides a matched-weight, one-series mechanism comparison without extending it to a general strategy ranking.

- Bundle: [figures/figure04_controlled_sam2_memory_tradeoff](figures/figure04_controlled_sam2_memory_tradeoff/README.md)
- Caption/source display: [open](figures/figure04_controlled_sam2_memory_tradeoff/README.md)
- Provenance: [manifest](figures/figure04_controlled_sam2_memory_tradeoff/manifest.json)
- Denominators: Coverage and post-gap branch-consistent recovery keep fixed opportunities; residuals retain method- and pair-specific conditional n.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n.
- Missing values: Visibility gaps and invalid fits remain absent and contribute to fixed-denominator outcomes.
- Palette: Uses frozen family-specific colors without a rank-like scale.

### Figure 5 — Common Gaussian refinement preserves the Cu(111) pairwise relative spot-radius endpoint across proposal strategies

Propagates the common measurement contract to the descriptive Cu(111) pairwise endpoint and between-method spread.

- Bundle: [figures/figure05_cu111_physical_endpoint](figures/figure05_cu111_physical_endpoint/README.md)
- Caption/source display: [open](figures/figure05_cu111_physical_endpoint/README.md)
- Provenance: [manifest](figures/figure05_cu111_physical_endpoint/manifest.json)
- Denominators: Each method retains 2804/2804 Gaussian measurements and 701/701 endpoints for each primary pair.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places.
- Missing values: Missing endpoints would remain missing; no gaps are present for the two primary pairs.
- Palette: Uses the frozen five-method colorblind-safe palette.

## Main tables

### Table 1 — Data and evidence roles

Defines data regimes, opportunity sets, evidence roles, observables, and independence units.

- Bundle: [tables/table01_data_evidence_roles](tables/table01_data_evidence_roles/README.md)
- Caption/source display: [open](tables/table01_data_evidence_roles/output/table01_data_evidence_roles.md)
- Provenance: [manifest](tables/table01_data_evidence_roles/manifest.json)
- Denominators: Lists complete opportunity sets and separates initialization states from displacement evaluations.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form.
- Missing values: Unavailable elements are described explicitly in the evidence-role contract.
- Palette: Not applicable to a semantic table.

### Table 2 — Methods mapped to the common measurement contract

Maps each method's native output and adapter into the shared estimator, availability, reset, and study-role contract.

- Bundle: [tables/table02_tracking_methods_contract](tables/table02_tracking_methods_contract/README.md)
- Caption/source display: [open](tables/table02_tracking_methods_contract/output/table02_tracking_methods_contract.md)
- Provenance: [manifest](tables/table02_tracking_methods_contract/manifest.json)
- Denominators: Declares availability and reset rules that determine fixed opportunity outcomes downstream.
- Rounding: Diagram labels and exact counts are not converted into rounded estimates.
- Missing values: Unavailable native outputs remain unavailable under the common contract.
- Palette: Not applicable to a semantic table.

## Supplementary figures

### Figure S1 — Controlled-translation detail

Provides full per-trial paths, proposal-to-measurement diagnostics, and pair invariance.

- Bundle: [figures/figure_s1_controlled_translation_trials](figures/figure_s1_controlled_translation_trials/README.md)
- Caption/source display: [open](figures/figure_s1_controlled_translation_trials/README.md)
- Provenance: [manifest](figures/figure_s1_controlled_translation_trials/manifest.json)
- Denominators: Preserves 20 displaced-spot and 10 displaced-pair opportunities per trial and method.
- Rounding: Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally. Counts and denominators are shown as exact integers in numerator/denominator form.
- Missing values: Missing proposals and measurements are shown as gaps without path interpolation.
- Palette: Uses the frozen five-method palette plus a distinct oracle control.

### Figure S2 — Measurement-window and capture-contract sensitivity

Shows the predefined ROI/gate matrix, OFAT perturbations, gate margins, and width-trigger decision as one display item split across consecutive a–d and e–f (continued) pages at 6.25 in width.

- Bundle: [figures/figure_s2_measurement_sensitivity](figures/figure_s2_measurement_sensitivity/README.md)
- Caption/source display: [open](figures/figure_s2_measurement_sensitivity/README.md)
- Provenance: [manifest](figures/figure_s2_measurement_sensitivity/manifest.json)
- Denominators: Every variant retains 221 visible opportunities, 24 gaps, accepted-fit n, and method-specific pair-frame n.
- Rounding: Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally. Counts and denominators are shown as exact integers in numerator/denominator form.
- Missing values: Undefined conditional metrics remain missing and are not promoted to fixed-denominator outcomes.
- Palette: Uses the frozen method palette and gate-specific sensitivity normalizers.

### Figure S3 — Pair-midpoint drift and apparatus/projection diagnostics

Separates midpoint drift and apparatus/projection diagnostics from the main pair-distance stress test.

- Bundle: [figures/figure_s3_ag111_midpoint_drift](figures/figure_s3_ag111_midpoint_drift/README.md)
- Caption/source display: [open](figures/figure_s3_ag111_midpoint_drift/README.md)
- Provenance: [manifest](figures/figure_s3_ag111_midpoint_drift/manifest.json)
- Denominators: Uses only jointly valid pair frames and retains case-specific visibility branches.
- Rounding: Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally. Counts and denominators are shown as exact integers in numerator/denominator form.
- Missing values: Missing pairs are not interpolated or connected.
- Palette: Method palette is intentionally not used for apparatus/projection diagnostics.

### Figure S4 — EasyLEED–ViPErLEED compatibility

Provides the separate external-reference compatibility contract and its own denominator.

- Bundle: [figures/figure_s4_easyleed_viperleed_compatibility](figures/figure_s4_easyleed_viperleed_compatibility/README.md)
- Caption/source display: [open](figures/figure_s4_easyleed_viperleed_compatibility/README.md)
- Provenance: [manifest](figures/figure_s4_easyleed_viperleed_compatibility/manifest.json)
- Denominators: External-reference opportunities remain separate from the common five-method matrix.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n.
- Missing values: Unavailable common fits remain absent from compatibility statistics.
- Palette: Uses distinct external-reference encodings, not a sixth common-matrix method color.

### Figure S5 — Geometry-inconsistent continuation examples

Shows qualitative examples selected by a frozen rule and linked to their source rows.

- Bundle: [figures/figure_s5_geometry_inconsistent_continuation](figures/figure_s5_geometry_inconsistent_continuation/README.md)
- Caption/source display: [open](figures/figure_s5_geometry_inconsistent_continuation/README.md)
- Provenance: [manifest](figures/figure_s5_geometry_inconsistent_continuation/manifest.json)
- Denominators: Qualitative examples remain attached to the accepted-fit denominators reported in Figure 4 and Table S2.
- Rounding: Diagram labels and exact counts are not converted into rounded estimates.
- Missing values: The selected examples preserve the original missing and accepted-fit state.
- Palette: Example annotations identify methods without implying a rank.

### Figure S6 — Cu pair 2–4 missingness diagnostic

Keeps the pair 2-4 amplitude-driven missingness diagnostic outside the main five-method endpoint contract.

- Bundle: [figures/figure_s6_cu111_pair_2_4_missingness](figures/figure_s6_cu111_pair_2_4_missingness/README.md)
- Caption/source display: [open](figures/figure_s6_cu111_pair_2_4_missingness/README.md)
- Provenance: [manifest](figures/figure_s6_cu111_pair_2_4_missingness/manifest.json)
- Denominators: Uses its separate 24-candidate diagnostic denominator and is not pooled into the primary pairs.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places.
- Missing values: Amplitude-only rejections remain explicit missing measurements.
- Palette: Uses the frozen method palette only for proposals inside the diagnostic.

## Supplementary tables

### Table S1 — Software, models, versions, hashes and environments

Freezes software, model, configuration, checkpoint, and environment provenance.

- Bundle: [tables/table_s1_software_environments](tables/table_s1_software_environments/README.md)
- Caption/source display: [open](tables/table_s1_software_environments/output/table_s1_versions_hashes_environments.md)
- Provenance: [manifest](tables/table_s1_software_environments/manifest.json)
- Denominators: Not a numerical-outcome table; provenance records are complete by declared environment role.
- Rounding: Diagram labels and exact counts are not converted into rounded estimates.
- Missing values: Unfrozen run-time fields are labeled by provenance status rather than inferred.
- Palette: Not applicable.

### Table S2 — Gaussian gates, fixed opportunity sets and conditional denominators

Centralizes Gaussian gates, fixed opportunity sets, and every method-specific conditional denominator.

- Bundle: [tables/table_s2_gates_denominators](tables/table_s2_gates_denominators/README.md)
- Caption/source display: [open](tables/table_s2_gates_denominators/output/table_s2_gaussian_gates_validation_denominators.md)
- Provenance: [manifest](tables/table_s2_gates_denominators/manifest.json)
- Denominators: Every fixed outcome and conditional metric has an explicit unit, denominator, and surviving n.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally.
- Missing values: Unavailable proposals and rejected fits remain inside fixed denominators.
- Palette: Not applicable.

### Table S3 — Ag(111) outcomes by scan–pair and leave-one-case-out

Reports scan-pair strata and leave-one-case-out sensitivity without treating four cases as independent acquisitions.

- Bundle: [tables/stratified_uncertainty](tables/stratified_uncertainty/README.md)
- Caption/source display: [open](tables/stratified_uncertainty/output/table_s3_energy_scan_by_case.md)
- Provenance: [manifest](tables/stratified_uncertainty/manifest.json)
- Denominators: Coverage, post-gap branch-consistent recovery, accepted fits, and pair residuals retain their own case- and method-specific denominators.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n.
- Missing values: Conditional residuals are omitted where no jointly branch-consistent pair frame survives.
- Palette: Not applicable.

### Table S4 — Cu(111) pairwise physical endpoints

Provides full-precision pairwise endpoints at frozen thermal-cycle landmarks.

- Bundle: [tables/table_s4_cu111_endpoints](tables/table_s4_cu111_endpoints/README.md)
- Caption/source display: [open](tables/table_s4_cu111_endpoints/output/table_s4_cu111_endpoints.md)
- Provenance: [manifest](tables/table_s4_cu111_endpoints/manifest.json)
- Denominators: Every method-pair endpoint reports valid frames out of 701.
- Rounding: Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally. Counts and denominators are shown as exact integers in numerator/denominator form. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places.
- Missing values: A missing endpoint would remain blank with its valid-frame count.
- Palette: Not applicable.

### Table S5 — Controlled translations by source trial

Retains full per-trial translation coverage, conditional errors, and pair invariance including the oracle control.

- Bundle: [tables/table_s5_controlled_translation](tables/table_s5_controlled_translation/README.md)
- Caption/source display: [open](tables/table_s5_controlled_translation/output/table_s5_controlled_translation.md)
- Provenance: [manifest](tables/table_s5_controlled_translation/manifest.json)
- Denominators: Each trial and method keeps 20 displaced-spot and 10 displaced-pair opportunities; conditional n is attached.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Controlled-translation displacement errors are shown to 4 decimal places. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places.
- Missing values: Proposal and fit failures remain in the fixed denominators.
- Palette: Not applicable.

### Table S6 — Measurement-window and capture-contract sensitivity

Reports all predefined sensitivity variants in a compact reader-facing view while preserving full-precision exports, without selecting a replacement configuration.

- Bundle: [tables/table_s6_measurement_sensitivity](tables/table_s6_measurement_sensitivity/README.md)
- Caption/source display: [open](tables/table_s6_measurement_sensitivity/output/table_s6_measurement_sensitivity.md)
- Provenance: [manifest](tables/table_s6_measurement_sensitivity/manifest.json)
- Denominators: Every variant retains fixed 221 and 24 opportunities plus accepted-fit and pair-frame conditional n.
- Rounding: Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n. Counts and denominators are shown as exact integers in numerator/denominator form. Machine-readable SI retains full source precision; any reader-facing summaries declare their formatting locally.
- Missing values: Undefined conditional values remain absent and are never converted to zero.
- Palette: Not applicable.

### Table S7 — Cross-regime numerical summary

Places the five-method numerical anchors side by side while preserving regime-specific evidence roles and denominators.

- Bundle: [tables/table_s7_cross_regime_summary](tables/table_s7_cross_regime_summary/README.md)
- Caption/source display: [open](tables/table_s7_cross_regime_summary/output/table_s7_cross_regime_summary.md)
- Provenance: [manifest](tables/table_s7_cross_regime_summary/manifest.json)
- Denominators: Each block retains its fixed opportunity set; every conditional error or residual follows coverage and carries its own n.
- Rounding: Counts and denominators are shown as exact integers in numerator/denominator form. Rates derived from exact counts are shown to 2 decimal places. Controlled-translation displacement errors are shown to 4 decimal places. Cu(111) endpoint deviations and pair-invariance percentage-point values are shown to 6 decimal places. Ag(111) conditional pair-distance residuals are shown to 3 decimal places with method-specific n.
- Missing values: Coverage exposes missing proposals and fits before conditional values.
- Palette: No color scale, composite score, or rank is used.
