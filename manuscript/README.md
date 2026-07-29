# Manuscript workspace

Status: complete narrative draft with explicit scientific and release
placeholders. Sections 1–8 and the abstract are written as manuscript prose.
Author information, selected acquisition metadata, and final repository,
license, and DOI details remain unresolved.

Working title:

> Evaluating Spot-Tracking Strategies for Quantitative LEED Analysis

Primary target: *Surface Science*.  
Backup target: *Computer Physics Communications* only if the software,
performance, and public-runner evidence is strengthened.

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
series labels. Their archive identifiers are mapped once in Section 2 for
traceability. Internal image numbers belong in manifests; reader-facing prose
describes the corresponding physical state or experimental role.

## Files

- [`manuscript.md`](manuscript.md) - working manuscript with drafted Sections
  1–8 and explicit placeholders for missing author, acquisition, and public
  release information.
- [`supplementary-material.md`](supplementary-material.md) - working
  supplementary document containing generated Tables S1–S4: software and
  environments, measurement gates and denominators, stratified energy-scan
  outcomes, and Cu(111) pairwise physical endpoints.
- [`references.bib`](references.bib) - primary-source bibliography used by the
  manuscript's Pandoc-style citation keys.
- [`claim-evidence-map.md`](claim-evidence-map.md) - permitted claims, numerical
  anchors, evidence sources, and limitations.
- [`figures-and-tables.md`](figures-and-tables.md) - planned main and supplementary
  display items.
- [`figures/`](figures/) - deterministic builders, source-data tables, manifests,
  and final publication assets.
- [`tables/table01_data_evidence_roles/`](tables/table01_data_evidence_roles/) -
  deterministic main table separating four data regimes from five evidence
  roles and recording their reporting units.
- [`tables/table02_tracking_methods_contract/`](tables/table02_tracking_methods_contract/) -
  deterministic tracking-method contract with five main strategies, a
  matched-weight SAM2-family control, and a separate whole-pattern reference.
- [`tables/table03_cross_regime_summary/`](tables/table03_cross_regime_summary/) -
  deterministic three-block summary of the common five-method results across
  controlled translations, Cu(111), and independent Ag(111) energy scans.
- [`tables/table_s1_software_environments/`](tables/table_s1_software_environments/) -
  deterministic supplementary software/model inventory with run-frozen
  hashes and path-free snapshots of eight retained execution environments.
- [`tables/table_s2_gates_denominators/`](tables/table_s2_gates_denominators/) -
  deterministic supplementary contract separating Gaussian gates, fixed
  validation opportunities, and method-dependent conditional sample sizes.
- [`tables/stratified_uncertainty/`](tables/stratified_uncertainty/) -
  finalized per-scan–pair Table S3 with leave-one-case-out sensitivity,
  together with the generated per-trial Table S5 draft, deterministic builder,
  and manifest.
- [`tables/table_s4_cu111_endpoints/`](tables/table_s4_cu111_endpoints/) -
  deterministic Cu(111) method-by-pair endpoint table with full-precision
  source CSV, reader-facing supplement block, provenance hash chain, and
  manifest.
- [`writing-checklist.md`](writing-checklist.md) - unresolved inputs and checks
  required before submission.

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

Use the following markers until the missing information is supplied:

- `TODO[AUTHOR]` - author or affiliation input;
- `TODO[CITATION]` - literature citation required;
- `TODO[DECISION]` - scientific or editorial choice still to be made;
- `TODO[FIGURE]` - display item not yet assembled;
- `TODO[TEXT]` - prose not yet drafted.

Numerical claims should be copied from a validated artifact, not recomputed in the
manuscript. If an artifact changes, update the claim-evidence map before revising
the prose.
