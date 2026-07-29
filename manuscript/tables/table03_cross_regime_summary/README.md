# Table 3: quantitative summary across evidence regimes

This artifact builds the manuscript's main cross-regime quantitative table
from a frozen five-method evidence matrix. One numbered table is rendered as
three vertically stacked blocks because the regimes answer different
questions and do not share one meaningful performance axis:

- controlled translations report proposal and Gaussian-measurement coverage,
  conditional displacement error, pair coverage, and pair invariance;
- the Cu(111) positive control reports Gaussian-measurement and endpoint
  availability together with agreement against the framewise method median;
- the independent Ag(111) energy scans report measurement coverage, post-gap
  identity recovery, geometry-inconsistent continuation, and the conditional
  pair-distance residual with its surviving pair-frame count.

No composite score, ranking, or rank-like color scale is produced. The caption
also discloses that DAM4SAM uses `sam21pp-L` for Cu(111) and the controlled
translations but `sam21pp-B` for the independent energy scans.

## Files

- `source_data/method_evidence_matrix.csv` is a byte-identical committed
  snapshot of the completed cross-regime evidence matrix.
- `source_data/source_provenance.json` records the original and frozen paths
  and SHA-256 hashes.
- `build_table.py` validates the common denominators and generates the outputs.
- `output/table03_quantitative_summary.csv` provides one machine-readable row
  per method with regime-prefixed columns.
- `output/table03_quantitative_summary.md` contains the exact block copied
  between the generated markers after Section 5.3 of the manuscript.
- `manifest.json` records source, builder, contract, and output hashes.

## Rebuild and verify

From this directory:

```bash
../../../.venv/bin/python -m unittest test_build_table.py
../../../.venv/bin/python build_table.py
```

The tests verify the frozen numerical anchors, source hash, block structure,
conditional denominators, deterministic outputs, and exact manuscript
synchronization.
