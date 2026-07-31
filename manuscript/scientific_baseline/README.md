# Frozen scientific-baseline contract

This package freezes the inputs, opportunity keys and measurement contract for
the principal Ag(111) publication stress test. It is the starting contract for
reconstruction, not a completed sensitivity result and not the
`scientific-baseline-complete` fingerprint.

## Frozen scope

- four cases: Scan A/B × pair 1/2;
- five methods: EasyLEED, LocoTrack, OSTrack, DAM4SAM and Lucas–Kanade;
- 81 unique raw images;
- 221 tracker-independent baseline-visible opportunities;
- 24 evaluable natural I(V) gaps;
- one central measurement contract, 15 mandatory sensitivity variants and
  five conditional width contracts.

The package copies stored native proposals and case-level physics-baseline
records into `frozen_inputs/`. Their original archive paths remain as
provenance but may be unavailable after freezing. Raw images are not copied:
they remain linked by repository-relative path, format, byte size and SHA-256
and therefore must be accessible for validation and later refitting.

## Validation

Run:

```text
python scripts/freeze_publication_baseline.py --validate-only
```

Validation is read-only and fail-closed. It checks package hashes, source
hashes and schemas, source-inventory bindings, the exact 221/24 keys, the
canonical contract and the package summary. It never refreshes hashes or
rewrites the baseline. `--build` is only for creating a new package from a
directory that contains `source_inventory.json` and no frozen outputs; it
refuses to replace an existing package.

The publication width gate is inclusive `0.60–13 px` and is evaluated after
fitting. It is distinct from the Gaussian optimizer's hard sigma bounds
(`0.5 px` to the ROI axis length), which are not varied by width sensitivity.
