# Figure S6 — Cu pair 2–4 missingness diagnostic

Status: generated and validated.

## Scientific role

This supplementary figure isolates an earlier saved-YOLO proposal trajectory
that is outside the five-method Cu(111) positive-control run. The proposal is
present in all `26/26` displayed images, while `24/24` consecutive rejected
Gaussian fits fail only the common amplitude threshold of `0.05`.

## Caption

**Figure S6. A saved Cu(111) pair 2-4 proposal can remain available while the
common quantitative position measurement is rejected.** Gaussian amplitude
and the fixed acceptance threshold are shown across the 26-image context
around the 800 °C plateau; accepted endpoints bracket 24 consecutive rejected
fits. All rejected candidates pass the `R^2`, center, and width gates and fail
only the uncalibrated amplitude gate. The grey dotted curve gives temperature
context. This diagnostic uses an earlier saved YOLO proposal trajectory
outside the frozen five-method run. It supports only the distinction between
proposal availability and quantitative-measurement availability; it does not
show identity loss, compare the five methods on spots 2 and 4, or define a
physical LEED I(V) observable.

## Rebuild and validate

```bash
python freeze_source_data.py
python freeze_source_data.py --validate-only
python build_figure.py
python build_figure.py --validate-only
python test_build_figure.py
```

Final PNG and PDF files are written to `output/`; `manifest.json`
authenticates the contract, source data, builder and outputs.
