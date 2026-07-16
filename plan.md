Plan: directional, hybrid tracking improvements

- [x] 1) Map current flow: read `analysis/tracking.py`, `logic/analysis_coordinator.py`, and `gui/tracking_dialog.py` to document how Hungarian/YOLO/OF are currently invoked, which params are passed, and where SpotROI/Track are created and persisted.
- [x] 2) Define config defaults: extend tracking params structure to include center point (auto = image center), radial direction mode (inward/outward/auto), lateral angle tolerance, ring bin size, dynamic max distance toggle, max_misses, YOLO refresh interval, and hybrid mode flag.
- [x] 3) Add geometry helpers: compute pattern center, radial unit vectors, angular offsets, ring/bin index from radius, and median step magnitude; expose a small util module or functions inside `analysis/tracking.py`.
- [x] 4) Upgrade Hungarian cost: replace pure Euclidean cost with combined radial + lateral cost, include penalties for ring changes and wrong direction, and clamp matches by dynamic max_distance (median step ± margin). Keep unmatched detection → new track and allow births any frame.
- [x] 5) Track lifecycle robustness: add `max_misses` handling (tracks survive pulsation), keep last predicted position during gaps, drop only after threshold, and ensure min_track_length is respected when closing tracks.
- [x] 6) Global direction flip: estimate median radial velocity across active tracks; when sign flips for majority, switch radial sign in the cost model (or keep dual predictions and pick cheaper assignment).
- [-] 7) Hybrid YOLO + OF: run YOLO detections every N frames; between refreshes propagate spots with LK and then associate to next YOLO detections with the directional Hungarian. Allow user-selected subset of detections to seed tracks.
- [x] 8) UI wiring: add controls in tracking dialog for center selection (manual/auto), direction mode, lateral tolerance, ring bin, dynamic max distance, max_misses, and YOLO refresh interval; plumb through coordinator to tracking functions with sensible defaults.
- [x] 9) Validation: craft a short script/notebook to simulate radial-inward/outward sequences with new-appearing spots, run all modes, and log jump/ID switches; adjust defaults based on results. Document the mode and defaults in `README.md`/dialog help text.

## Plan: export FWHM from positions analysis

### Export contract

- [x] 1) Preserve the existing CSV columns and append fit metadata: `position_source`, `fit_ok`, `quality_status`, `quality_flags`, `sigma_x`, `sigma_y`, `fwhm_x`, `fwhm_y`, `fwhm_x_error`, `fwhm_y_error`, `theta`, and `r_squared`.
- [x] 2) Keep FWHM and sigma values in pixels, keep `theta` in radians, and document that `fwhm_x`/`fwhm_y` describe the principal axes of the rotated Gaussian fit rather than fixed image axes.
- [x] 3) Export numerical FWHM values for fits with `quality_status` equal to `ok` or `warning`. For `fit_ok=false`, keep FWHM and sigma cells empty while retaining diagnostic fields such as `r_squared`, `quality_status`, and `quality_flags`.
- [x] 4) Distinguish fitted positions from fallback tracked positions using `position_source=gaussian_fit` or `position_source=tracked_center`. Preserve the current behavior for missing spots and group filtering.
- [x] 5) Treat `positions_analysis_results.fit_points_by_frame` as the source of truth. Export must flatten stored fit results and must not reload TIFF files or rerun Gaussian fitting.

### Implementation workflow (`implement` + TDD)

- [x] 6) Record the current commit SHA (`98cbbd20d06b06e29d604fb12b88663ca4a414a4`) as the fixed point for the final review and use this approved plan as the implementation specification.
- [x] 7) Add one tracer-bullet test showing that a valid stored fit writes `fwhm_x` and `fwhm_y` to the exported CSV; run it first and confirm the RED state.
- [x] 8) Implement the minimum change in `PositionsAnalysisDialog._build_export_point_rows()` and `_export_positions_results_csv()` needed to make the tracer-bullet test GREEN.
- [x] 9) Continue in vertical RED-GREEN cycles for: warning fits, failed fits, fallback tracked positions, missing spots, group filtering, and empty cells instead of `nan` for unavailable FWHM values.
- [x] 10) Once all behavioral tests are GREEN, refactor fit-payload flattening into one focused helper, for example `_build_fit_export_fields()`, and rerun the tests after the refactor.
- [x] 11) Update `README.md` with the new CSV columns, units, quality semantics, and the distinction between fitted and fallback positions.

### Verification and delivery

- [x] 12) During implementation, regularly run the new focused test file, the existing positions-analysis/coordinator tests, and the repository's configured typechecker. If no mypy or pyright configuration exists, use `py_compile` as the minimum syntax/import check.
- [x] 13) Run the full test suite once after implementation and refactoring are complete.
- [x] 14) Verify the export against `data/24i02_R-33_N20_350`: expect 2270 exported position rows, 2258 rows with accepted FWHM values, 12 present-but-failed fits with empty FWHM values, and 70 absent spots omitted as before. Compare representative exported values with the saved `.positions.json`, including frame 19 / track 1 (`fwhm_x=18.633149...`, `fwhm_y=20.694953...`).
- [x] 15) Run the `code-review` workflow against the SHA recorded before implementation, reviewing Standards and Spec separately. Address all actionable findings and rerun the affected tests.
- [x] 16) Create a focused commit on the current branch containing the exporter, tests, and documentation changes.
