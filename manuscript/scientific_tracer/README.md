# Scan A / pair 1 / LocoTrack reconstruction tracer

This package freezes one narrow tracer through the scientific reconstruction
seam. It starts from the frozen LocoTrack proposals and the 35 linked Scan A
TIFF images, performs the common Gaussian measurement on CPU, evaluates the
central measurement contract, summarizes the pair and gap outcomes, and
validates the result against the historical central record.

The tracer never runs a tracker, opens the GUI, loads a model checkpoint, or
uses a GPU. `frozen/` is immutable input and expected-output evidence. `output/`
is rebuilt deterministically by the tracer command.

The frozen CPU environment is Python 3.12.9, NumPy 2.2.5, SciPy 1.15.2,
scikit-image 0.25.2 and tifffile 2025.3.30. The command checks these versions
before fitting because optimizer changes outside this fingerprint can exceed
the declared continuous-value tolerance.

Run:

```text
python -m scripts.run_publication_tracer --clean --validate
```

The candidate layer contains source identity, crop geometry, solver outcome,
fit parameters and errors, geometry residuals and signed gate margins. The
evaluation layer references candidate keys and contains classifications only;
threshold-only changes therefore do not copy or recompute fits.

Validation is fail-closed and does not refresh the frozen record. It checks
input and output hashes, schemas, record keys, statuses, classifications,
counts, continuous-value tolerances and reader-facing rounding.

The generated `output/` contains separate candidate and evaluation CSV files,
pair and gap records, a central and reader-facing summary, provenance with the
CPU runtime fingerprint, an output hash manifest and the targeted validation
report. Running with `--validate` and without `--clean` validates an existing
output without rewriting it.
