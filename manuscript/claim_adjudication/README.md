# Scientific claim-boundary adjudication

This artifact combines the author-confirmed metadata constraints with the
predeclared measurement-window and capture-contract sensitivity.
`adjudication_contract.json` records a decision and final bounded wording for
every claim C1–C9. The generated output contains the sensitivity anchors, the
complete boundary, and a boundary-bound human checkpoint.

The proposed decisions are four preserved, five narrowed, and zero removed.
No claim may be strengthened by a favorable perturbation, and no perturbation
changes the frozen central measurement contract or selects a configuration.

Build from the repository root with:

```bash
python -m manuscript.claim_adjudication.build_adjudication
```

`output/author_checkpoint.json` was explicitly confirmed by the publication
author on 2026-08-01. A confirmed checkpoint is preserved only while its
boundary SHA-256 still matches. A new approval can be recorded after building
with `--confirm`, `--confirmed-by`, and `--confirmed-date`.
