# Supplementary Table S4 — Cu(111) pairwise endpoints

Status: generated and validated.

## Purpose

Table S4 reports the exact method- and pair-specific values behind the three
thermal-cycle landmarks summarized in Figure 5:

- the median of the predefined five-image 800 °C plateau;
- the outcome-selected post-plateau minimum, which occurred in the common
  image acquired at 797 °C for every method and pair;
- the predefined final cooled state acquired at 92 °C.

The reporting unit is one method–pair combination within one Cu(111)
thermal-cycle series. The two primary pairs are strata of that series, not
independent replicates. Values are descriptive and do not establish external
absolute accuracy, an overall tracker ranking, or crystallographic anisotropy.

## Source contract

The `source_data/` directory contains byte-identical committed snapshots of
the validated Figure 5 endpoint summaries, cycle landmarks, and independent
between-method spread calculation. `source_provenance.json` records and
hash-checks both each local snapshot and its committed Figure 5 origin.

The relative endpoint is

`Delta R/R = Delta D/D = D_ij(t) / D_ij,ref - 1`,

where `D_ij,ref` is the median valid pair distance in the first ten images,
with at least eight valid distances required. All ten method–pair records have
ten valid reference images and 701/701 valid endpoints.

Frame identifiers remain in the frozen machine-readable contracts for audit
purposes. The generated reader-facing table uses physical phase and
temperature labels.

## Outputs

- `output/table_s4_cu111_endpoints.csv` retains full-precision endpoint and
  reference values for all ten method–pair records.
- `output/table_s4_cu111_endpoints.md` is the six-decimal reader-facing table
  copied exactly between the Table S4 markers in
  `manuscript/supplementary-material.md`.
- `manifest.json` records source, builder, and output hashes.

## Rebuild

From this directory:

```bash
../../../.venv/bin/python -m unittest test_build_table.py
../../../.venv/bin/python build_table.py
```

The builder refuses altered schemas, malformed or escaping provenance paths,
hash mismatches, incomplete method–pair coverage, changed landmark roles, or
endpoint extrema inconsistent with the independent Figure 5 spread table.
