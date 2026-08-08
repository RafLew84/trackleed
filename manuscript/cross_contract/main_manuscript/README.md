# Focused cross-contract main manuscript

This package contains the venue-portable scientific draft for the focused
cross-contract article. `manuscript.md` is the reader-facing source. It binds
the author-approved claim boundary, frozen principal-scan outcomes, and the
three compact main figures without treating either measurement contract as
ground truth.

The draft deliberately remains upstream of journal styling and DOCX/PDF page
assembly. Ticket 09 owns canonical editable-document generation, page-level
rendering, and final package integration. This package instead freezes and
validates scientific prose, section structure, word budgets, display inventory,
terminology, denominators, and Authorized deferred publication inputs.

From the repository root:

```text
python scripts/freeze_cross_contract_manuscript.py --freeze
python scripts/freeze_cross_contract_manuscript.py --validate-only
pytest trackleed/tests/test_cross_contract_manuscript.py -q
```

Freeze mode is staged, validates before publication, and refuses to replace an
existing manifested manuscript package.
