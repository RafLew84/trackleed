# H6 synthetic benchmark (stage 2)

This entry point creates a self-contained tracking benchmark alongside the existing
segmentation-training generator. It supports stationary and energy-scanned H6 scenes, optional common drift,
one constant background, one deterministic realization and multiple observation schedules.
It requires NumPy, tifffile and Pillow; generation and verification import no GUI,
torch, trackers or model weights.

## Generate and verify

Run from the repository in its Python environment:

```sh
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h6.json --validate-only
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h6.json --output runs/h6-static-example
python scripts/generate_tracking_benchmark.py --verify runs/h6-static-example
```

The output parent must already exist. The target must not exist, even as an empty
directory. Use a new name for every repetition. This repository's local environment
is available in WSL as `.venv/bin/python`; substitute that for `python` there.

The supplied example has three 1000 × 1000 frames at 0, 50 and 100 ms, six persistent
IDs and 18 reference records. Positions and pixels are identical across observations.
Its known center is (499.5, 499.5); lattice scale is 180 px, circular sigma 4 px,
peak amplitude above background 0.5, background 0.1, and fixed PNG bounds [0, 1].

Validation prints the fully expanded configuration and counts without creating
output files or rendering a sequence. Normal generation prints the plan to stderr
before writing. Success and verification reports are JSON on stdout; an error
returns a nonzero exit code with the failing field or file.

## Open in TrackLEED

1. Generate the package and run the verification command above.
2. Start TrackLEED with `python -m trackleed`.
3. Select **File → Open Directory...** and choose the package's **images** folder.
4. Move through frames 0, 1 and 2. The same six reflections keep their positions and brightness.
5. Inspect `frames.csv` and `reference.csv`: three observations, six IDs per observation.

Only the `images` folder is intended for GUI loading. It contains PNG8 frames only.
TrackLEED's existing generic filename parser supplies numeric frame IDs, but does
not ingest benchmark time, energy or reference tables into its session model;
the authoritative observation time and 100 eV energy remain in `frames.csv`.
Manual GUI inspection is optional; the integration test uses the real parser
and image reader on generated PNG files without opening widgets.

## Configuration version 1

The root object requires `schema_version: 1` and objects `scene`, `condition`
and `schedule`. Empty objects select the defaults below. `realization_id`
defaults to `deterministic`. IDs start with a letter and contain at most 64
ASCII letters, digits, underscores or hyphens.

| Object | Fields and defaults |
| --- | --- |
| scene | id: static-h6; pattern: H6; image_size: [1000,1000] (width,height); center_px: [499.5,499.5]; lattice_scale_px: 180; sigma_px: 4; amplitude: 0.5; energy_ev: 100 |
| condition | id: constant; background: 0.1; png_lower: 0; png_upper: 1 |
| schedule | id: static; times_ns: [0,50000000,100000000] |

Widths, lattice scale and non-null energy must be positive; amplitude and background
are nonnegative. All real parameters are finite. Image dimensions are positive
integers. Times are strictly increasing, nonnegative int64 nanoseconds; booleans
are not numbers. Energy may be null for an unlabelled stationary scene.
PNG bounds must have a positive finite difference.

Unknown fields, duplicate JSON keys and unsupported versions/options are errors.
Version 1 rejects motion; version 2 adds the energy and sampling features below.
Both versions currently reject H18/H54, noise, imported backgrounds, apparatus masks
and optional instance-mask export. The entire 4-sigma support must fit inside
the frame's pixel-center rectangle; clipping is introduced in stage 5.
A profile with no sampled support (W0 = 0) is an error, not a disappearance.
Rendering also rejects values which cannot be represented as finite float32.

## Energy scans and sampling (configuration version 2)

Version 2 adds one fixed H6 scene with multiple schedules. Version 1 inputs and
published packages remain readable with their original schema, IDs and flat image
paths. Schema versions describe data; the generator version is recorded separately.
The version 2 root uses `schedules` (a nonempty array) instead of `schedule`.
The scene and condition defaults above still apply. No random motion is introduced.

Ready 1000 × 1000 configurations:

| File in configs/synthetic-benchmark | Expected result |
| --- | --- |
| energy_scan_h6.json | 80–200 eV over 2 s; schedules every 20/50/100 ms; 101/41/21 frames and 978 reference records |
| stationary_control_h6.json | 100 eV; 0, 1, 2 s; exactly identical positions and pixels |
| common_drift_h6.json | 100 eV; 0, 1, 2 s; common drift (6,0) px/s, ending 12 px right |

With an existing output parent, use a different target name for each run:

```sh
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/energy_scan_h6.json --validate-only
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/energy_scan_h6.json --output runs/h6-energy-example
python scripts/generate_tracking_benchmark.py --verify runs/h6-energy-example
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/stationary_control_h6.json --output runs/h6-stationary-control
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/common_drift_h6.json --output runs/h6-drift-control
```

In version 2 the image folders are `sequences/<schedule_id>/images` and
`sequences/<schedule_id>/rendered`. The root CSV tables contain all schedules;
the manifest records `schedule_ids`. All sequences are verified and published
as one package: a failure in a later schedule publishes none of them.

A timed scan declares the following scene field (omit `energy_ev`, or set it to null):

```json
"energy_scan": {
  "model": "linear_time",
  "reference_energy_ev": 100,
  "energies_ev": [80, 200],
  "times_ns": [0, 2000000000],
  "valid_energy_range_ev": [80, 200],
  "projection_radius_px": 600,
  "domain_justification": "Ideal normal-incidence projection; included H6 beams propagate."
}
```

The two energies correspond to the two times. Ascending, descending and constant
scans are supported; times must increase strictly. Observations outside this
declared interval are rejected. The scene can also be evaluated between scheduled
observations without changing its state or depending on call order.

All centers follow `c0 + d(t) + sqrt(E0/E(t)) * (p0-c0)`. The common drift is
`scene.drift_px_per_s * (time_ns / 1e9)`, measured from synthetic time zero,
and defaults to [0,0]. Amplitude and sigma stay constant; the background stays
fixed in image coordinates. The brightest sampled pixel can vary with subpixel
position even though the continuous peak amplitude is constant.

Without `energy_scan`, `energy_ev` is a fixed optional label and no reference
energy is required. With zero drift this gives exact stationary control.
Drift can be combined with a timed scan or a fixed energy.

The scan requires positive E0, a positive ordered applicability range, positive
projection scale R0, and a nonempty justification. Both the requested energies
and E0 must lie within the declared range. For all included H6 beams the validator
checks `(r0/R0) * sqrt(E0/E_min) <= 1` across that range.
R0 is an ideal projection parameter, **not** the screen-mask radius or measured
apparatus calibration. This check enforces the declared model assumption; it does
not infer real propagation from positive energies or pixel positions.
A conservative bound using maximum radius and drift extrema must fit the full
4-sigma support inside the image; clipping remains a later-stage feature.

A schedule uses either an explicit strictly increasing `times_ns` array or:

```json
{"id": "every-20ms", "sampling": {
  "start_ns": 0, "end_ns": 2000000000,
  "interval_ns": 20000000, "phase_ns": 0
}}
```

Its first observation is start + phase, then integer interval steps up to and
including end when aligned. Phase must satisfy 0 <= phase < interval.
The expanded scene preserves these parameters and the full integer time list.
Unstructured time lists have `sampling: null`; the list directly specifies every
observation. Generated regular schedules are limited to 1,000,000 observations
to reject accidental excessive expansion. IDs are unique ignoring case; version 2
schedule IDs also reject Windows device names.

Timed observation IDs are `t<nanoseconds>`: at 100 ms the ID is `t100000000`
in every schedule, despite frame indices 5, 2 and 1. Integer differences are taken
before converting the scan fraction to float, preserving nanosecond keys even
above 2^53. PNG and TIFF pixels and reflection states at a shared observation
are identical. Positional deltas and previous IDs can differ because each schedule
has a different predecessor. First deltas and the first predecessor are empty.

For an energy list without known times, set `model: "energy_list"`,
`times_ns: null`, and e.g. `energies_ev: [200, 120, 120, 80]`.
Keep the same required E0/domain/projection/justification fields.
A schedule selects `observation_indices: [0, 1, 2, 3]`; indices refer to the
ordered scene list, must increase strictly, and may select a subset.
Uneven energy steps, descending order and repeated values are preserved.
IDs `e0`, `e1`, etc. identify separate occurrences, so the two 120 eV entries
retain different observation IDs. Equal energy does not mean the same observation.
Both time and delta time remain empty; delta energy and displacement are real
differences along the selected sequence. Time-dependent drift is rejected for
this axis. Explicit time and energy-index schedules cannot be mixed.

### Inspect the moving example

1. Generate and verify the package with the commands above.
2. Open `sequences/every-20ms/images` in TrackLEED and step through frames:
   all six reflections contract together about (499.5,499.5).
3. Open the 50 ms and 100 ms folders separately. Compare 100 ms at indices 5, 2
   and 1 using `frames.csv` observation ID `t100000000`; their pixels match.
4. Open the stationary control's `sequences/control/images`: all three frames
   are identical. The drift control shifts the pattern 0, 6 and 12 px right.
5. Inspect energy, time and predecessor deltas in `frames.csv`, then positions
   and displacements in `reference.csv`. The existing GUI does not import these
   CSV metadata automatically.

These are instantaneous observations. Sampling interval is not exposure duration;
no motion blur, attenuation, noise, H18/H54, masks, or tracker ranking is introduced.

## Package and numerical contract

- `manifest.json`: type `trackleed.synthetic-benchmark`, schema version 1 or 2,
  complete status, scene/condition/realization/schedule IDs, counts, generator
  provenance and hashes of every other file.
- `input.json`: the actual input configuration; `scene.json`: its expanded values.
- `schema.json`: versioned column types/units, coordinates, rendering and encoding rules.
- `frames.csv`: observation IDs, int64 times, optional energy, predecessor and
  time/energy deltas, image paths, file/pixel hashes and clipping/quantization counts.
- `reference.csv`: every frame × reflection, including zero-amplitude reflections.
- `rendered/frame_XXXXXX.tiff`: unnormalized, unclipped float32 (inside each sequence in v2).
- `images/frame_XXXXXX.png`: uint8 grayscale with one fixed photometric mapping (inside each sequence in v2).
- `.trackleed-synthetic-benchmark`: permanent type marker, also protecting corrupt
  packages from the legacy exporter.

Coordinates are float64 in pixels: x right, y down, top-left pixel center (0,0).
The six primary reflections are indexed by (h,k) satisfying
max(|h|,|k|,|h+k|)=1, using b1=(r,0), b2=(r/2,sqrt(3)*r/2).
The central reflection is excluded. IDs derive from those indices, e.g. h+1_k+0.

At pixel centers, intensity is A exp(-q²/2) for q ≤ 4, exactly zero beyond.
The omitted continuous Gaussian mass is exp(-8); it is not redistributed.
A is the continuous peak parameter, not the brightest sampled pixel.
The nominal contour q ≤ 3 is distinct from rendering support.
Overlap lists describe intersections of analytic closed disks; they are not masks
or a detectability result. No instance masks or boxes are exported at this stage.

The renderer adds components in float64 and then casts the composed image to float32.
PNG uses the stored TIFF array, converted to float64 for the fixed mapping:
floor(255 * clip((I-L)/(U-L), 0, 1) + 0.5). There is no per-frame min/max scaling.
Counterfactual images are recomposed with the same background and other reflections,
omitting one reflection, then converted to float32 and PNG by the same rules.

The reference records amplitude extinction separately from positive pre-quantization
signal, float64/float32 signal pixels, changed float32 image pixels and changed PNG
pixels. Thus weak signal, floating-point loss, saturation and extinction stay distinct.
W0 is the unit-amplitude sampled support sum. Since clipping and apparatus masks
are unsupported in stages 1–2, W1=W2=W3=W0, the retained fraction is 1 and geometric
loss fractions are 0, also for A=0.

Every ID persists in every frame. The first predecessor and all first deltas are
missing, represented by empty CSV fields; subsequent deltas are differences from
the preceding observation of the same schedule (zero for stationary positions).
Booleans use true/false, lists use JSON, CSV is UTF-8 with comma separators and
decimal points. NaN/Infinity are never missing-value encodings.

Pixel fingerprints hash a sorted compact JSON header (dtype with little-endian
byte order, shape, order=C), a zero byte and the contiguous canonical pixel bytes.
File hashes and pixel hashes are distinct. Reproduction is promised within the
recorded code/library environment, not across arbitrary numerical libraries.
The source digest hashes a sorted compact JSON mapping of generator paths to
SHA-256 file hashes. Git state covers tracked files and generator sources; unrelated
untracked files are outside that flag's scope. If Git is unavailable, Git metadata
is null and the source digest still identifies the generator content.

## Publication, integrity and compatibility

A unique sibling directory named `.trackleed-incomplete-*` holds the working
result. The entire package is read back and checked before atomic publication.
Publication never replaces an existing target, including a late competing empty
directory. Windows uses no-replace directory rename; Linux uses renameat2 with
RENAME_NOREPLACE. Other platforms fail explicitly instead of weakening this guarantee.

WSL Windows-mounted drives may reject renameat2 flags. In that case publication
uses the native Windows [Directory.Move](https://learn.microsoft.com/en-us/dotnet/api/system.io.directory.move)
operation through PowerShell interoperability. Source and target must be sibling
Windows drive paths; paths travel as JSON data on stdin and are never shell code.
Missing interoperability causes a clear failure and retains the unpublished stage.
A regression test exercises successful publication and a late empty-directory
conflict on the repository's actual filesystem.

Failure leaves an unpublished staging directory, with a failure report when
possible. There is no automatic overwrite, resume or cleanup of previous results.
A staging directory is rejected even if it already contains a complete manifest.
Symlinks inside packages and staging aliases are rejected.

Verification reads actual TIFF/PNG/CSV/JSON, checks inventory, file and decoded
pixel hashes, IDs/counts/schema/provenance, complete frame-by-ID records, and
reconstructs the supported scene to verify image and reference contents.
Checksums detect corruption; they do not authenticate a maliciously rewritten package.

The old generator, training masks, YOLO labels, flow and PNG16 conversion retain
their previous meaning. The legacy PNG converter rejects a new benchmark root or
any descendant before creating its destination, including recognized packages with
unsupported/corrupt manifests. Already prepared benchmark PNG8 needs no conversion.
Detached anonymous TIFF files without the package marker/manifest cannot reliably
be identified as benchmark data.

Generator references and prepared images are separate from later tracker outputs.
PNG8 does not establish identical model-internal pixels after resize, preprocessing
or intermediate JPEG encoding. This stage does not run or rank tracking models.

## Automated checks

```sh
python -m unittest trackleed.tests.test_synthetic_config trackleed.tests.test_synthetic_scene trackleed.tests.test_synthetic_package trackleed.tests.test_synthetic_cli trackleed.tests.test_synthetic_motion trackleed.tests.test_synthetic_sampling -v
python -m mypy --follow-imports=silent trackleed/synthetic scripts/generate_tracking_benchmark.py scripts/export_segmentation_to_png.py
python scripts/validate_current_workflow.py
```

The repository validator includes the code-size guardrail and configured mandatory
tests; acquisition-dependent publication tests remain outside its CI suite.
Small fixtures check independent analytic positions/pixels, subpixel centers,
support boundaries, quantization, preserved identities, real file round trips,
legacy compatibility and publication failures. The supplied 1000 × 1000 example
and the new scan/drift/stationary controls are the user demonstrations.
