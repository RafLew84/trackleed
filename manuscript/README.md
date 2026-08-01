# Manuscript workspace

Status: venue-portable scientific revision with the final claim boundary.
Sections 1–7 and the abstract are written as manuscript prose. The experimental
description, author information, submission metadata, and release authorization
are explicitly deferred.

Working title:

> Evaluating Heterogeneous LEED Spot-Tracking Methods under a Common
> Spot-Position Measurement Contract

Target form: venue-portable methodological Research Paper. Journal-specific
formatting and placement decisions are deferred to the submission adapter.

## Scope

The paper evaluates spot tracking as one component of a quantitative LEED
measurement chain. Native tracker outputs are proposals for a common `48 x 48`
measurement ROI; the reported position is the center of the same two-dimensional
Gaussian fit for every method.

TrackLEED is the reference implementation and reproducibility vehicle. The paper is
not a GUI or software-feature article, and the absence of dense manual annotation
is part of the validation protocol rather than the central scientific claim.

## Reader-facing terminology

Use **Ag(111) energy scan** in the manuscript for the experiment type called
`ScanE` in the internal repository glossary. Outside this terminology note,
the string `ScanE` should appear only inside literal archive identifiers,
artifact paths, or filenames.

Use **Scan A**, **Scan B**, and **memory-analysis scan** as the scientific
series labels. Archive identifiers and internal image numbers belong in
provenance manifests; reader-facing prose describes only the scientific series
and evidence role.

## Files

- [`manuscript.md`](manuscript.md) - venue-portable manuscript with drafted
  Sections 1–7 and explicit placeholders for deferred experimental, author,
  submission, and release-authorization inputs.
- [`supplementary-material.md`](supplementary-material.md) - working
  supplementary document containing generated Tables S1–S7.
- [`references.bib`](references.bib) - primary-source bibliography used by the
  manuscript's Pandoc-style citation keys.
- [`claim-evidence-map.md`](claim-evidence-map.md) - permitted claims, numerical
  anchors, evidence sources, and limitations.
- [`figures-and-tables.md`](figures-and-tables.md) - generated final inventory
  of exactly five main figures, two main tables, Figures S1–S6, and Tables
  S1–S7.
- [`publication_inventory/`](publication_inventory/) - cross-regime gate that
  validates titles, roles, source/output hashes, builders, tests, rounding,
  denominators, palette, and missing-value contracts for all final displays.
- [`figures/`](figures/) - deterministic builders, source-data tables, manifests,
  and final publication assets.
- [`tables/table01_data_evidence_roles/`](tables/table01_data_evidence_roles/) -
  deterministic main table separating four data regimes from five evidence
  roles and recording their reporting units.
- [`tables/table02_tracking_methods_contract/`](tables/table02_tracking_methods_contract/) -
  deterministic tracking-method contract with five main strategies, a
  matched-weight SAM2-family control, and a separate whole-pattern reference.
- [`tables/table_s7_cross_regime_summary/`](tables/table_s7_cross_regime_summary/) -
  deterministic three-block supplementary summary of the common five-method
  results across controlled translations, Cu(111), and independent Ag(111)
  energy scans.
- [`tables/table_s1_software_environments/`](tables/table_s1_software_environments/) -
  deterministic supplementary software/model inventory with run-frozen
  hashes and path-free snapshots of eight retained execution environments.
- [`tables/table_s2_gates_denominators/`](tables/table_s2_gates_denominators/) -
  deterministic supplementary contract separating Gaussian gates, fixed
  validation opportunities, and method-dependent conditional sample sizes.
- [`tables/stratified_uncertainty/`](tables/stratified_uncertainty/) -
  finalized per-scan–pair Table S3 with leave-one-case-out sensitivity,
  deterministic builder, and manifest.
- [`tables/table_s4_cu111_endpoints/`](tables/table_s4_cu111_endpoints/) -
  deterministic Cu(111) method-by-pair endpoint table with full-precision
  source CSV, reader-facing supplement block, provenance hash chain, and
  manifest.
- [`tables/table_s5_controlled_translation/`](tables/table_s5_controlled_translation/) -
  controlled-translation outcomes by independently initialized source trial.
- [`tables/table_s6_measurement_sensitivity/`](tables/table_s6_measurement_sensitivity/) -
  full-precision measurement-window and capture-contract sensitivity results.
- [`writing-checklist.md`](writing-checklist.md) - unresolved inputs and checks
  required before submission.
- [`scientific_baseline/`](scientific_baseline/) - immutable input,
  opportunity and measurement-contract package for the principal Ag(111)
  reconstruction, including stored native proposals, `221/24` reference keys
  and read-only fail-closed validation.
- [`experimental_metadata/`](experimental_metadata/) - retained source-linked
  acquisition-metadata record, explicit unavailable fields, TIFF inventory and
  metadata-derived claim constraints; it is not used to infer the deferred
  reader-facing experimental description.
- [`scientific_core/`](scientific_core/) - documented one-command headless CPU
  reconstruction contract, pinned environment, semantic component contract,
  and fail-closed `scientific-baseline-complete` fingerprint.

## Reconstruct and validate the scientific core

Run from the repository root in the pinned CPU environment:

```text
python -m scripts.reconstruct_publication --clean --validate
```

The command rebuilds the central result, sensitivity, source data, all twenty
main/SI display bundles and manifests, then validates the manuscript and the
frozen semantic fingerprint. It never refreshes the fingerprint automatically.

## Sources of truth

- [Final publication concept](../.scratch/trackleed-physics-method-paper/issues/08-wybor-finalnej-koncepcji.md)
- [Resolved publication map](../.scratch/trackleed-physics-method-paper/map.md)
- [Cross-regime evidence matrix](../.scratch/trackleed-physics-method-paper/artifacts/cross-regime-evidence-matrix/README.md)
- [Controlled-translation artifact](../.scratch/trackleed-physics-method-paper/artifacts/controlled-translation-chain/README.md)
- [Independent Ag(111) energy-scan validation](../.scratch/trackleed-physics-method-paper/artifacts/independent-scane-validation/README.md)
- [Five-method Cu(111) run](../.scratch/trackleed-physics-method-paper/artifacts/cu111-tracker-evaluation-subset/runs/five-methods-v1/README.md)
- [Cu(111) pairwise endpoint](../.scratch/trackleed-physics-method-paper/artifacts/cu111-pairwise-delta-r/README.md)
- [EasyLEED/ViPErLEED external reference](../.scratch/trackleed-physics-method-paper/artifacts/domain-baseline-high-energy/README.md)

## Drafting convention

Only the following markers are authorized in the reader-facing package until
the deferred information is supplied:

- `TODO[EXPERIMENT]` - experimental-team description and approval;
- `TODO[AUTHOR]` - author, affiliation, contribution, or acknowledgement input;
- `TODO[SUBMISSION]` - journal-specific submission decision;
- `TODO[RELEASE-AUTHORIZATION]` - legal or repository-release authorization.

Numerical claims should be copied from a validated artifact, not recomputed in the
manuscript. If an artifact changes, update the claim-evidence map before revising
the prose.
