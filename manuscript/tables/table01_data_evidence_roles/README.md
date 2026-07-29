# Table 1: data and evidence roles

This artifact builds the manuscript's main data-regime table from one frozen,
reader-facing source file. The table distinguishes four experimental regimes
from five evidence roles:

- the controlled translations are derived from four source states in the
  Cu(111) sequence and are not four independent acquisitions;
- the restricted high-energy external reference reuses 15 images from the
  memory-analysis scan and is not an additional dataset or replicate.

This structure prevents the display from inflating the amount of independent
evidence. It also keeps the EasyLEED/ViPErLEED reference separate from the
common five-strategy matrix because it has a different initialization contract
and denominator.

## Files

- `source_data/data_regimes.csv` is the frozen publication-facing table source.
- `source_data/upstream_contracts/` contains byte-identical snapshots of the
  five upstream contracts used to construct the table.
- `source_data/source_provenance.json` records each original path and hash
  alongside the committed snapshot path and hash. Rebuilding validates the
  local snapshots and does not require `.scratch`.
- `build_table.py` validates the evidence-role structure and creates the
  publication outputs.
- `output/table01_data_evidence_roles.csv` is the machine-readable table.
- `output/table01_data_evidence_roles.md` is the exact block copied between the
  generated markers in Section 2.1 of the manuscript.
- `manifest.json` records source, builder, and output hashes.

## Rebuild and verify

From this directory:

```bash
python3 -m unittest test_build_table.py
python3 build_table.py
```

The test also validates all frozen-source hashes, checks that the generated
Markdown and the manuscript copy are identical, and ensures that internal
archive identifiers do not enter the reader-facing table.
