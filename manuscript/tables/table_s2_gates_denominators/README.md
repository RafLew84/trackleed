# Table S2: Gaussian gates and validation denominators

This artifact builds supplementary Table S2 as three vertically stacked
blocks:

- the frozen Gaussian-measurement and energy-branch evaluation gates;
- fixed opportunity denominators, including the separate diagnostic and
  frozen visual-audit counts;
- method-dependent sample sizes used only for conditional error summaries.

The separation is deliberate. For example, the 221 baseline-visible
opportunities per method in the independent energy scans remain the coverage
denominator even though the numbers of accepted fits in all visibility
regimes are 272, 269, 272, 206, and 222. Those latter counts are denominators
only for geometry-inconsistency rates. Likewise, pair-distance residuals use
the explicitly reported number of images on which both branches passed the
proposal, Gaussian, and branch-consistency gates.

## Files

- `source_data/gate_records.csv` is the normalized, publication-facing gate
  contract derived from the archived executed sources.
- `source_data/denominator_records.csv` records every fixed, audit, diagnostic,
  and conditional denominator used by the validation.
- `source_data/upstream_contracts/` contains archived snapshots of the
  executed gate implementations and source result records. Text snapshots use
  repository-stable LF line endings; provenance retains both frozen and
  original SHA-256 hashes.
- `source_data/source_provenance.json` links each snapshot to its origin and
  validates frozen SHA-256 hashes. Current committed origins are checked during
  every build; the two original run-workspace paths are optional audit
  locations because the committed snapshots make a clean checkout
  self-contained. The historical `peak_analyzer.py` executed source is pinned
  to its originating commit, so later compatible API changes do not rewrite
  the scientific record. Text origins whose working-tree line endings can
  differ are checked against an explicit LF-normalized hash.
- `build_table.py` validates the numerical contracts, creates three
  machine-readable CSV files and the three-block Markdown table, and
  synchronizes the generated Table S2 block in
  `manuscript/supplementary-material.md`.
- `manifest.json` records all source, builder, and output hashes.

The two archived runner files under `executed_sources/` are retained evidence,
not reusable library modules. Machine-specific paths are absent from the
reader-facing table.

## Rebuild and verify

From this directory:

```bash
python3 -m unittest test_build_table.py
python3 build_table.py
```

Changing a gate, denominator, or archived upstream snapshot changes primary
scientific source data and requires review rather than a formatting-only
rebuild.
