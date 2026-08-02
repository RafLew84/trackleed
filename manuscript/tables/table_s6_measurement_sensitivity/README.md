# Table S6 — Measurement-window and capture-contract sensitivity

Table S6 reports the complete 15-variant, five-method Ag(111)
measurement-window and capture-contract sensitivity. The compact
publication-facing Markdown has 75 method-aggregate rows, 15 reader codes and
seven columns. Its counts remain exact while conditional residuals are rounded
to three decimal places for display.

The output bundle keeps the original 375-row full-precision canonical CSV and
adds a 75-row full-precision method-aggregate CSV, a 15-row variant codebook,
and a reader-convenience XLSX. The workbook contains exactly the sheets
`method_aggregate` and `case_records`, with 75 and 300 data rows respectively.
Numeric cells remain typed and retain source precision; number formats control
display rounding only. CSV remains the source of truth.

The generated Markdown includes the human-readable 15-code codebook. Its
`table-s6-row-style` directive carries the approved pale-blue fill for the five
central-contract rows into the later canonical DOCX/PDF builder; the asterisk
remains the non-color cue. The XLSX applies the same fill directly.

Coverage and recovery retain their fixed denominators (`221` and `24`).
Geometry inconsistency retains accepted-fit n, and pair-distance residuals
retain method- and variant-specific jointly consistent pair-frame n. The table
does not select a configuration or define an overall tracker ranking.

Build from the repository root with:

```bash
python -m manuscript.tables.table_s6_measurement_sensitivity.build_table
```

The builder requires XlsxWriter 3.2.9 in the pinned headless reconstruction
environment. It consumes the single authenticated package in
`manuscript/source_data/measurement_sensitivity/`; `manifest.json` binds that
shared source package, implementation, and output hashes.
