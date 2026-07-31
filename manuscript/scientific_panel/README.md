# Complete Ag(111) central-panel reconstruction

This immutable package freezes the historical record-by-record result for the
five-method, four-case Ag(111) panel and the expected outputs reconstructed
from the separately frozen scientific-baseline inputs.

The runtime path is headless and CPU-only. It does not run a tracker, load a
checkpoint, open the GUI, select a configuration or update the scientific
baseline. Candidate and evaluation records retain proposal, ROI, solver and
gate outcomes separately. Coverage and recovery use the fixed 221/24 key sets;
geometry inconsistency and pair residuals report their conditional per-method
denominators.

Run:

```text
python -m scripts.run_publication_panel --clean --validate
```

Validation authenticates this package, the scientific-baseline package and
the existing publication source-data anchors before comparing every discrete
and continuous record under the predeclared tolerances.
