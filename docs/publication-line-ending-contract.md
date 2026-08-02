# Publication line-ending contract

Status: active
Decision date: 2026-08-02

## Canonical policy

All repository text that participates in the publication scientific-core
fingerprint or reader-facing semantic inventory is checked out and committed
with LF (`\n`) line endings. The root `.gitattributes` enforces
`text=auto eol=lf`, including on Windows with `core.autocrlf=true`.

Every reconstruction writer that emits text uses an explicit LF newline
contract. `Path.write_text` calls declare `newline="\n"`; CSV writers declare
`lineterminator="\n"` and use an explicit newline mode. Generated outputs are
updated only by their builders.

## Byte-preserved source exceptions

Some authenticated upstream packages contain text-shaped files whose original
bytes are evidence. Nested `.gitattributes` files classify those paths as
`binary`/`-text`, so Git performs no checkout conversion. They are explicit
byte-preserved inputs, not generated repository text:

- frozen scientific-baseline inputs;
- frozen controlled-translation sources;
- frozen source-data packages for Figure S4, Tables 1/2/S3/S7;
- frozen software, runtime, configuration and provenance records for Table S1.

Generated CSV outputs in Tables 1, 2 and S7 override their package-level
binary rule with `text eol=lf`.

The fail-closed guard in
`trackleed/tests/test_publication_line_endings.py` derives hash-bound paths from
the scientific-core fingerprint contract and publication semantic inventory.
It rejects an undeclared byte-preserved exception, a text path without
effective `eol=lf`, or a reconstruction `write_text` call without an explicit
newline.

## Fingerprint boundary

This contract changes builders and semantic manifests but does not replace
`manuscript/scientific_core/fingerprint.json`. The predecessor fingerprint
remains immutable and is expected to reject these reviewed component changes
until the separately authorized controlled rebaseline.
