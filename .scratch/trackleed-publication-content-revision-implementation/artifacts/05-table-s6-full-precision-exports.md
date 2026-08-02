# I05 — Table S6 and full-precision exports

Date: 2026-08-02

## Canonical path

- Input: `manuscript/source_data/measurement_sensitivity/sensitivity_results.csv`.
- Builder: `trackleed/publication/scientific_sensitivity_evidence.py`, invoked by
  `python -m manuscript.tables.table_s6_measurement_sensitivity.build_table`.
- Reader view: compact 75-row, seven-column Markdown synchronized into
  `manuscript/supplementary-material.md`.
- Machine-readable outputs: unchanged 375-row canonical CSV, 75-row
  full-precision method-aggregate CSV, 15-row codebook CSV, and a two-sheet
  XLSX with 75 `method_aggregate` plus 300 `case_records` rows.

The canonical CSV SHA-256 remains
`5941410f0f912f856f0f8a826a972009155720aed6b8ca290cd31e79a268257e`.
The artifact manifest binds every source, implementation file and output.

## Reader and workbook contracts

The reader view uses the 15 approved short codes, exact numerator/denominator
counts and three-decimal conditional residuals with exact `n`. A generated
human-readable codebook defines ROI, proposal-to-fit, branch-tolerance, R² and
amplitude variants. The central `R48-P15*` block keeps the non-color asterisk;
the canonical row-style directive and XLSX use fill `#EAF2F8`.

The user explicitly authorized XlsxWriter as the alternative generator because
the session did not expose the spreadsheet skill's artifact-tool runtime.
XlsxWriter 3.2.9 is pinned in the headless environment. Excel read-only QA
confirmed exactly two sheets, 76/301 used rows including headers, 33 columns,
typed numeric `Double` cells, filters, frozen panes, and a distinct central-row
fill. Both sheets passed visual inspection; temporary previews were removed.

## Verification

- Initial RED: two Table S6 contract failures for absent aggregate CSV,
  codebook and XLSX.
- Final targeted Table S6: `5 passed`.
- Declared sensitivity display contracts: `8 passed`.
- Publication inventory: `8 passed`.
- Ruff and compileall: passed.
- Full `trackleed/tests`: `248 passed, 6 failed`; all six failures are the
  pre-existing `PermissionError` baseline for unavailable
  `manuscript/scientific_panel` inputs. No new failure appeared.
- Frozen predecessor: `freeze_scientific_core --validate-only` rejected the
  semantic revision as required; rebaseline remains deferred to I12.
- Final Standards review: 0 findings.
- Final Spec review: 0 findings after resolving the human-readable-codebook and
  canonical-central-style observations.
