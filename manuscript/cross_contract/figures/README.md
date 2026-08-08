# Compact cross-contract main figures

This package contains the three author-bound main-text figures for the focused
cross-contract manuscript:

1. a full-width conceptual framework for the two operational paths and their
   shared interpretation layer;
2. a single-column view of the four measurability outcomes and contract-
   specific versus shared bounded-gap components;
3. a single-column view of accepted out-of-branch event classification.

The package contains no image montage and no selected crop. Quantitative panels
are rebuilt from the frozen Ticket 04 outputs, while terminology and caption
limits are bound to the Ticket 05 author-approved decision. Outputs are fixed
at 178 x 78 mm for Figure 1, 88 x 98 mm for Figure 2, and 88 x 85 mm for
Figure 3. PNGs are rendered at 300 dpi and PDFs retain those exact page sizes.

Build or validate from the repository root:

```text
python scripts/build_cross_contract_figures.py --build
python scripts/build_cross_contract_figures.py --validate-only
```

Validation authenticates all upstream sources, regenerates machine-readable
panel data and all six image outputs, checks final-size dimensions and minimum
typography, and rejects any byte drift.
