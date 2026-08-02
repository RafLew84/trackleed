# I02 publication line-ending contract — implementation evidence

Date: 2026-08-02
Environment: `C:\Users\rlewa\anaconda3\envs\netcat\python.exe`

## Outcome

Hash-bound repository text is canonical LF on checkout and write. Authenticated
source packages remain byte-preserved. The predecessor scientific-core
fingerprint was not edited or rebaselined.

## RED

- The initial checkout-policy guard failed on a hash-bound path without
  effective `eol=lf`.
- The writer guard found 20 `Path.write_text` calls without an explicit
  newline contract.
- Regression tests reproduced direct and transitive SHA-256 collisions between
  normalized text and authenticated byte-preserved evidence.

## GREEN implementation

- Root `.gitattributes`: `* text=auto eol=lf`.
- Nested attributes preserve authenticated source bytes and explicitly restore
  LF for generated CSV outputs.
- Reconstruction writers use `newline="\n"` (CSV writers already use explicit
  LF terminators).
- `scripts/migrate_publication_line_endings.py` performs a fixed-point,
  reviewable migration while keeping the predecessor fingerprint immutable.
- Table S1 declares its frozen `source_data/**` boundary explicitly and keeps
  frozen hash constants semantically stable while its builder is normalized.

## Verification

- `pytest trackleed/tests/test_publication_line_endings.py`: 7 passed.
- Line-ending guard plus Table S1 contract: 19 passed.
- Full `trackleed/tests`: 245 passed; the six remaining failures are the known
  pre-existing permission failures for user-deleted/inaccessible
  `manuscript/scientific_panel` inputs.
- Existing focused bundle suites for Tables 1, 2, S1, S7, Figure S2, Table S6
  and publication inventory passed during implementation.
- Clean Windows snapshot was checked out with `core.autocrlf=true`; required
  local inputs were copied as independent files.
- Final migration report: 14 controlled hash-graph changes, three fixed-point
  iterations, `fingerprint_replaced: false`.
- `manuscript/publication_inventory/build_inventory.py` passed all 20 bundle
  contracts in the clean snapshot.
- `python -m scripts.reconstruct_publication --clean --validate` regenerated
  and validated the scientific artifacts without a line-ending mismatch. Its
  only terminal error was the required fail-closed result:
  `Scientific-core fingerprint differs from the frozen baseline`.
- Two-axis code review converged with 0 remaining Standards findings and
  0 Spec findings.

This terminal error is expected until the separately authorized controlled
rebaseline ticket; accepting or replacing the fingerprint is outside I02.
