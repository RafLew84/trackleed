# Scientific tracer architecture

The narrow Scan A / pair 1 / LocoTrack tracer keeps one publication-facing
orchestration seam while assigning each internal concern to one module:

- `scientific_tracer_contract` authenticates the frozen package and validates
  the execution and input contracts;
- `scientific_tracer_records` owns typed record schemas and CSV codecs;
- `scientific_tracer_reconstruction` owns Gaussian reconstruction and physical
  pair, gap and summary artifacts;
- `scientific_tracer_evaluation` owns threshold-only classification;
- `scientific_tracer_validation` compares reconstructed artifacts with frozen
  evidence and enforces fail-closed output integrity;
- `trackleed.publication.scientific_tracer` owns reconstruction orchestration
  and publication policy.

Normal reconstruction and validation never update frozen evidence. An
intentional implementation-only refreeze has a separate command:

```text
python -m scripts.refreeze_publication_tracer --approve-refreeze
```

The refreeze path first authenticates the existing package, stages output with
the current typed schemas, and requires every field retained from the older
schema to match the frozen scientific record. It then refreshes expected
records, typed schemas, counts, the package manifest and the external
authentication record together. It is tested against a temporary package
copy; changing the repository's immutable package still requires explicit
author approval.
