# Experimental metadata and metadata-derived claim constraints

This package records the experimental metadata that can be supported by
authoritative local sources for the TrackLEED publication scientific core. It
also records explicit gaps and the claim constraints caused by those gaps.
These constraints are inputs to later claim adjudication; they are not the
final scientific claim boundary.

The statuses have strict meanings:

- `recorded`: an authoritative source records the value and is linked by path,
  locator and SHA-256;
- `not recorded`: the author confirms that the value was not recorded;
- `not available`: no authoritative value is available in the audited sources,
  without claiming that no external laboratory record ever existed.

No value is inferred solely from a filename. In particular, the headerless
Scan A and memory-analysis sidecars remain uninterpreted because no local
column schema was found.

`tiff_metadata_inventory.json` binds the embedded TIFF geometry, encoding and
writer fields to all 851 publication acquisition images. Validation checks the
package hashes, provenance-source hashes, TIFF source hashes and embedded
metadata, required fields, statuses, constraints and author checkpoint without
rewriting any record.

Run the package validation with:

```text
python scripts/validate_experimental_metadata.py
```

The frozen package includes the author's HITL confirmation of every recovered
value, explicit gap and absence of unsupported inference. The scientific-core
gate uses:

```text
python scripts/validate_experimental_metadata.py --require-author-confirmation
```
