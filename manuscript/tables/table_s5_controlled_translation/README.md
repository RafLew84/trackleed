# Table S5 - controlled translations by source trial

This artifact reports all four independently initialized source trials from
the controlled-translation experiment. It preserves full numeric precision in
CSV and keeps the fixed opportunity denominators separate from conditional
measurement denominators.

The frozen, authenticated source package is shared at
`manuscript/source_data/controlled_translations/`. Run the deterministic
builder from the repository root:

```bash
python -m manuscript.tables.table_s5_controlled_translation.build_table
```

The builder writes the full-precision CSV, reader-facing Markdown, and a
manifest binding source, implementation, and output SHA-256 hashes. It also
synchronizes the generated Markdown into `manuscript/supplementary-material.md`.

Known translations assess measurement-chain equivariance and
common-estimator behavior. They do not establish absolute tracking accuracy
or identity ground truth.
