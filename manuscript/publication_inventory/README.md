# Final publication display inventory

This integration artifact freezes the approved display set: five main
figures, two main tables, Figures S1-S6, and Tables S1-S7. It validates each
bundle's full-precision source-data and rendered-output hashes against its
manifest, and also requires a builder, tests, title/role contract, rounding
rules, denominator policy, method-palette policy, and missing-value policy.

Historical artifact directories may remain for provenance, but they are not
final display placements. In particular, there is no main Table 3; the
cross-regime numerical summary is Table S7.

## Files

- `source_data/display_inventory.json` is the human-reviewable source
  inventory and shared rendering contract.
- `build_inventory.py` runs every declared bundle contract test by default,
  validates every source/output path-hash association, writes
  `../figures-and-tables.md`, and records bundle fingerprints that also cover
  builders, captions, and tests.
- `test_build_inventory.py` checks exact membership, fail-closed title and
  placement behavior, hash-chain completeness, and deterministic output.
- `manifest.json` binds the inventory source, validator, final index, and all
  twenty bundle fingerprints.

## Rebuild and verify

From this directory:

```bash
../../.venv/bin/python -m unittest test_build_inventory.py
../../.venv/bin/python build_inventory.py
```

`--skip-contract-tests` is reserved for rebuilding the index after the same
contract tests have already been run separately in the current worktree.
