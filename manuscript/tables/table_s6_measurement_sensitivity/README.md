# Table S6 — measurement sensitivity

Table S6 reports the complete 15-variant, five-method Ag(111) sensitivity
assessment. The publication-facing Markdown shows method aggregates; the CSV
retains all 375 full-precision method and case records.

Coverage and recovery retain their fixed denominators (`221` and `24`).
Geometry inconsistency retains accepted-fit n, and pair-distance residuals
retain method- and variant-specific jointly consistent pair-frame n. The table
does not select a configuration or define an overall tracker ranking.

Build from the repository root with:

```bash
python -m manuscript.tables.table_s6_measurement_sensitivity.build_table
```

The authenticated sources are in `source_data/`; `manifest.json` binds source,
implementation, and output hashes.
