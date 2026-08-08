# Constrained cross-contract Supplementary Information

This package contains the short, venue-portable Supplementary Information
source for the focused cross-contract manuscript. It is limited to supporting
validation of the frozen measurement contracts, the restricted calibration
pilot, and the predeclared sensitivity grid. It does not duplicate the full
per-frame evidence, image sequences, execution environment, hashes, or
exhaustive configurations already retained in manifested repository artifacts.

The reader-facing source is `supplement.md`. Two figures and two compact tables
are designed for an 88 mm single-column width and use a declared minimum type
size of 7 pt. Ticket 09 owns DOCX/PDF integration and Page-level render
acceptance; this package freezes the scientific SI content and standalone
display assets upstream of that editorial step.

From the repository root:

```text
python scripts/freeze_cross_contract_supplement.py --freeze
python scripts/freeze_cross_contract_supplement.py --validate-only
pytest trackleed/tests/test_cross_contract_supplement.py -q
```

Freeze mode builds in staging, validates the complete package before
publication, and refuses to replace an existing frozen supplement.
