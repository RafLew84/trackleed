# Table 2: methods mapped to the common measurement contract

This artifact builds the manuscript's tracking-method overview from one
frozen source table. It preserves three distinct roles:

- five methods in the common strategy matrix;
- three SAM2-family methods compared under byte-identical Hiera Base+
  checkpoint weights;
- ViPErLEED as a separate whole-pattern external reference with a different
  initialization contract and denominator.

The publication table shows the interfaces needed to understand the
comparison, including each native availability rule and the common downstream
measurement. Audit-only fields in the source table retain exact upstream
revisions, software versions, checkpoint or plugin hashes, and executed
configuration hashes where a separate hash was recorded. A whole-contract
hash is never presented as a method-configuration hash. Detailed environments
belong in supplementary Table S1.

## Files

- `source_data/methods.csv` is the frozen table source. It contains nine
  publication-facing fields plus the row order and audit-only fields.
- `source_data/upstream_contracts/` contains byte-identical snapshots of the
  five upstream records used to validate the rows.
- `source_data/source_provenance.json` records the original and frozen paths
  and SHA-256 hashes. Rebuilding does not require `.scratch`.
- `build_table.py` validates panel membership, revisions, checkpoint hashes,
  available executed-configuration hashes, common-family weights, and the
  external-reference boundary.
- `output/table02_tracking_methods_contract.csv` is the machine-readable
  publication table.
- `output/table02_tracking_methods_contract.md` is the exact block copied
  between the generated markers in Section 3.3 of the manuscript.
- `manifest.json` records the table contract and source, builder, and output
  hashes.

## Rebuild and verify

From this directory:

```bash
python3 -m unittest test_build_table.py
python3 build_table.py
```

The test also checks deterministic output, exact manuscript synchronization,
the shared controlled-family checkpoint, and exclusion of internal archive
identifiers from the reader-facing table.
