# Stratified descriptive reporting

This artifact freezes the manuscript's uncertainty-reporting decision. It
does not treat successive images, visible opportunities, or I(V) gaps within
one propagated trajectory as independent replicates.

## Reporting policy

- Results are descriptive; no confidence intervals, p-values, or image-level
  bootstrap intervals are reported.
- Controlled translations are stratified by four separately initialized
  source trials drawn from one Cu(111) sequence; they are not four independent
  experimental acquisitions.
- The independent Ag(111) energy-scan stress test is stratified by two scans
  and two frozen pairs per scan.
- The SAM2-family and Cu(111) experiments each contain one series and two
  pairs and are not assigned population-level uncertainty intervals.
- Nonlinear Gaussian-fit uncertainty is a diagnostic of one accepted image
  measurement, not sampling uncertainty for tracker performance.
- Binomial intervals are not applied to correlated frames or gaps.

## Outputs

- `output/energy_scan_by_case.csv` and
  `output/table_s3_energy_scan_by_case.md` form the finalized Table S3,
  preserving coverage, post-gap identity recovery, geometry inconsistency,
  and conditional residuals for all 20 scan–pair–method cases. The generated
  Markdown is copied exactly between the Table S3 markers in
  `manuscript/supplementary-material.md`.
- `output/energy_scan_leave_one_case_out.csv` omits one of the four frozen
  scan-pair cases at a time and recomputes only count-based outcomes. It is a
  sensitivity analysis, not a confidence interval.
- `output/controlled_translation_by_trial.csv` and
  `output/table_s5_controlled_translation_by_trial.md` report proposal and
  measurement coverage, conditional displacement-error summaries, pair
  coverage, and maximum absolute pair-endpoint change for each source trial.
  They form the finalized Table S5. The generated Markdown is copied exactly
  between the Table S5 markers in `manuscript/supplementary-material.md`.

The leave-one-case-out analysis does not pool or average the conditional
case-median residuals. Their raw per-case values and surviving pair-frame
denominators remain in Table S3.

## Rebuild

From this directory:

```bash
../../../.venv/bin/python -m unittest test_build_tables.py
../../../.venv/bin/python build_tables.py
```

The builder validates the provenance schema, confined paths, and SHA-256
hashes in `source_data/source_provenance.json`, writes all CSV and Markdown
outputs, records the builder and output hashes in `manifest.json`, and
synchronizes Tables S3 and S5 into the supplement. The two original
run-workspace paths are optional audit locations; their committed frozen
tables make a clean checkout self-contained.
