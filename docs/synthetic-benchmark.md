# Synthetic tracking benchmark (stage 4)

This entry point creates a self-contained tracking benchmark alongside the existing
segmentation-training generator. It supports H6/H18/H54 scenes with circular or
elliptical profiles, optional instance masks, per-ID attenuation, event coverage, energy scans, optional common drift,
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
Versions 1 and 2 reject H18/H54 and instance-mask export; version 3 adds these below.
All versions still reject noise, imported backgrounds and apparatus masks.
The entire 4-sigma support must fit inside
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
no motion blur, attenuation, noise or tracker ranking is introduced by this feature.

## Patterns, profiles and instances (configuration version 3)

Version 3 keeps the version 2 `schedules` format and observation rules. It adds
`scene.pattern: "H18"` or `"H54"`, plus a root `export_masks` boolean (default false).
H6 is also supported. Versions 1 and 2 retain their existing configurations,
reference columns and historical package readability.

The lattice basis is b1=(r,0), b2=(r/2,sqrt(3)*r/2), for
`r = scene.lattice_scale_px`. With n=max(|h|,|k|,|h+k|), H6 uses n=1 and H18
uses 1<=n<=2. Neither includes (0,0). H54 adds six satellites around each of
the six n=1 primary reflections: q*(cos(j*pi/3),sin(j*pi/3)), j=0..5.
`scene.satellite_offset_px` defaults to 18 and must be positive; the close
example uses 9. These vectors are shared by all parents, with no random offsets.
Primary IDs stay `h+1_k+0`, etc.; satellite IDs append `_s0` through `_s5`.
Their `parent_id` names the primary, their role is `satellite`, and h/k record
the parent's lattice indices. IDs and relationships persist in every observation.

`scene.amplitude` sets primary amplitude; `scene.satellite_amplitude` defaults
to 0.15. Each must be nonnegative. Both are continuous peak parameters, not
discrete image maxima. Zero amplitude retains positions and ellipse geometry.
Time-dependent attenuation is introduced in a later stage.

The circular shorthand `scene.sigma_px` remains available. Alternatively specify
one shared shape for all reflections:

```json
"shape": {"sigma_u_px": 6, "sigma_v_px": 3, "theta_rad": 0.5235987755982988}
```

Both widths must be positive and all parameters finite. The angle rotates u
from image x towards image y (downwards). Omitted shape fields default to
4, 4 and 0. An explicit shape and `sigma_px` together are rejected. The expanded
configuration records the selected representation; reference rows always carry
both actual widths and the angle. The same shape is used by every primary and
satellite. Energy changes their positions and satellite offsets, while widths,
angles and amplitudes stay fixed. Propagation and frame-support checks include
all primary and satellite positions, including the outer H18 shell.

When `export_masks` is true, each frame and ID gets two independent grayscale
PNG files with values 0 and 255, under
`sequences/<schedule>/masks/frame_XXXXXX/`. Masks are never stored in `images`.
The geometry mask samples the closed 3-sigma ellipse at pixel centers. The
signal mask is the same for positive amplitude and empty for zero amplitude.
Weak positive amplitudes retain that mask even when PNG quantization removes
their image contribution. Geometry and signal masks from different IDs may
overlap. They are separate layers, not an exclusive-label segmentation map.

Version 3 reference rows add `geometry_mask_path`, `signal_mask_path`,
`bbox_geometry` and `bbox_signal`. Bounding boxes are tight raster bounds
`[xmin,ymin,xmax_exclusive,ymax_exclusive]`, encoded as JSON arrays inside CSV.
An empty mask has an empty CSV box field (null), while its all-zero file still
exists. With export disabled all four optional fields are empty. No mask export
changes images or the remaining reference fields. The analytic ellipse remains
defined by x/y, sigma_u/v and theta even if its nominal raster is empty.
An empty 3-sigma mask does not imply absence of the rendered 4-sigma tail.

Overlap lists describe analytic closed contours/support, including for zero
amplitude. They account for the shared ellipse's axes and rotation and do not
assert detectability. File inventory, mask pixels and bounding boxes are verified
against the scene after real readback, alongside the existing image/reference checks.

### Stage 3 demonstrations

All examples contain three stationary 1000 x 1000 images at 0, 50 and 100 ms:

| Configuration | IDs / reference rows | Shape and output |
| --- | --- | --- |
| static_h18.json | 18 / 54 | Circular sigma=4; no masks |
| static_h54.json | 54 / 162 | Circular sigma=4, q=18; no masks |
| static_h54_close_masks.json | 54 / 162 | Elliptical 6/3, theta=pi/6, q=9; 324 mask files |

Use a new output name on each run, with an existing parent directory:

```sh
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h18.json --validate-only
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h18.json --output runs/h18-static-example
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/h18-static-example
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h54.json --output runs/h54-static-example
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/h54-static-example
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/static_h54_close_masks.json --output runs/h54-close-example
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/h54-close-example
```

1. Generate and verify all three packages with the commands above.
2. In TrackLEED select **File -> Open Directory...**, then a package's
   `sequences/static/images` folder. Move through the three unchanged images.
3. Compare H18 with H54: the six inner primaries gain six satellites each;
   the twelve outer primaries retain their positions. Count IDs in `reference.csv`,
   since overlapping signals need not show separate image maxima.
4. Open the close example's images. The profiles are elliptical and neighboring
   signals overlap. The known pattern center remains (499.5,499.5).
5. Inspect the separate `h+1_k+0_geometry.png` and `h+1_k+0_s0_geometry.png`
   masks in `sequences/static/masks/frame_000000/`: they share pixels and retain
   separate IDs. Their corresponding rows provide the parent relation and boxes.

The existing GUI loads the PNG images; reference CSV and instance masks remain
separate benchmark artifacts. Technical acceptance uses the real TrackLEED
parser and image reader without requiring manual widget interaction.

## Package and numerical contract

- `manifest.json`: type `trackleed.synthetic-benchmark`, schema version 1, 2 or 3,
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
In H6 the six primary reflections are indexed by (h,k) satisfying
max(|h|,|k|,|h+k|)=1, using b1=(r,0), b2=(r/2,sqrt(3)*r/2).
The central reflection is excluded. IDs derive from those indices, e.g. h+1_k+0.

At pixel centers, intensity is A exp(-q²/2) for q ≤ 4, exactly zero beyond.
The omitted continuous Gaussian mass is exp(-8); it is not redistributed.
A is the continuous peak parameter, not the brightest sampled pixel.
The nominal contour q ≤ 3 is distinct from rendering support.
Versions 1/2 use circular profiles and analytic closed disks for overlap.
Version 3 additionally supports the shared ellipse and optional masks/boxes above.
Overlap is not a detectability result.

The renderer adds components in float64 and then casts the composed image to float32.
PNG uses the stored TIFF array, converted to float64 for the fixed mapping:
floor(255 * clip((I-L)/(U-L), 0, 1) + 0.5). There is no per-frame min/max scaling.
Counterfactual images are recomposed with the same background and other reflections,
omitting one reflection, then converted to float32 and PNG by the same rules.

The reference records amplitude extinction separately from positive pre-quantization
signal, float64/float32 signal pixels, changed float32 image pixels and changed PNG
pixels. Thus weak signal, floating-point loss, saturation and extinction stay distinct.
W0 is the unit-amplitude sampled support sum. Since clipping and apparatus masks
are unsupported in stages 1–4, W1=W2=W3=W0, the retained fraction is 1 and geometric
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
python -X utf8 -m unittest trackleed.tests.test_synthetic_patterns trackleed.tests.test_synthetic_instances trackleed.tests.test_synthetic_frozen -v
python -X utf8 scripts/validate_current_workflow.py
```

The [shared local/CI validator](workflow-validation.md) checks changed text,
generator types, code size, and all configured mandatory tests, including frozen
historical v1/v2/v3 packages. Acquisition-dependent publication tests remain outside
its CI suite.
Small fixtures check independent analytic positions/pixels, subpixel centers,
support boundaries, quantization, preserved identities, real file round trips,
legacy compatibility and publication failures. The supplied 1000 × 1000 example
and the new scan/drift/stationary controls are the user demonstrations.


## Attenuation and event coverage (configuration version 4)

Version 4 adds `scene.attenuations` (default `[]`). Versions 1–3 preserve their
original configuration, schemas, pixels and references and reject this new field.
An example declaration is:

```json
{"id":"long-gap","reflection_id":"h+1_k+0","axis":"time_ns",
 "knots":[700000000,700000000,1100000000,1100000000],"minimum_fraction":0}
```

Each event requires exactly these five fields. The knots are
`[fade_start, minimum_start, minimum_end, return_end]`, ordered as
`a <= b < c <= d`. Transitions are linear; equal transition endpoints make a
step. A step minimum is half-open `[b,c)`. With a nonzero return duration, the
first return point `c` still has minimum amplitude; `d` has baseline amplitude.
For knots `[500,700,1100,1300]` ms and zero minimum, the 1100 ms sample is zero.
The corresponding step has already returned at 1100 ms. Time knots use exact
nonnegative int64 nanoseconds, including values above 2^53.

The minimum is a fraction in `[0,1)` of that reflection's own base amplitude:
0 extinguishes it and 0.05 retains 5% (.025 for a .5 primary, .0075 for a .15
satellite). The base amplitude must be positive for an attenuation event;
unchanging zero-amplitude reflections remain supported without an event.
Positive attenuated amplitudes that underflow to zero are rejected rather than
silently reclassified as extinction. Values lost later in float32/PNG remain
separate reference quantities.

Supported axes are `time_ns` and `energy_ev`. Energy knots are positive finite
values on an increasing energy axis. The profile A(E) is evaluated in the
declared acquisition order, even when E decreases or repeats. There is no
invented time or interpolation between entries of an energy list. Time profiles
require known times; energy profiles require known E. Different reflections may
use different axes. Event IDs are portable and unique; target IDs must exist.
This version supports one event per reflection and rejects overlapping or
multiple definitions for the same ID, including mixtures of axes.

All IDs, positions, parent relations, geometric masks and displacements persist
through A=0. Only the signal mask becomes empty (with a null signal bbox).
At positive A it retains the nominal geometry even if PNG contains no detectable
change. Attenuation does not change the common energy scaling or the profile
shape. Fixed L/U encoding is applied to all frames.

Every v4 package contains `coverage.json`, also when its event list is empty.
It belongs to the manifest inventory and hashes; verification reconstructs its
contents from the scene and actual ordered observations. Changing a count and
updating its hash is still rejected. A failed report write prevents publication.

Each event × schedule record contains its definition, counts and observation IDs:

| Field | Meaning |
| --- | --- |
| `minimum_sample_count` / `minimum_observation_ids` | Samples in the declared minimum, including the first point of a linear return |
| `attenuated_sample_count` / `attenuation_observed` | Samples in fade, minimum or return phases; baseline endpoints are excluded |
| `zero_amplitude_sample_count` / `captured_zero_gap` | Samples with A=0; the boolean never represents tracker recovery |
| `zero_signal_sample_count` | Attenuated samples with zero float64 component; separate from latent A=0 |
| `positive_signal_lost_in_float32_count` | Positive float64 components causing no changed float32 image pixel |
| `positive_float32_signal_lost_in_png_count` | Positive changed float32 contributions causing no changed PNG code; may include saturation |
| `sampling_status` | `captured` when the minimum was sampled, otherwise `missed`; partial ramp observations are counted separately |
| `before_observation_id` / `after_observation_id` | Nearest baseline samples before/after the entire event, in acquisition order |
| `context_status` | `bracketed`, `truncated`, `outside_observations`, or `nonmonotonic` |
| `truncated_before` / `truncated_after` | Missing baseline context; read together with `context_status`, which identifies events outside the observations |
| `sampled_runs` | Contiguous attenuated observations with local baseline neighbors and truncation flags |

For a nonmonotonic energy sequence, global before/after and truncation values are
null: use the separately recorded runs in acquisition order. These are observed
groups, not inferred continuous trajectories. Repeated energies count as distinct
observations. A sampled 5% minimum is captured attenuation, never a zero gap.
A ramp-only sequence can have `attenuation_observed=true`, `sampling_status=missed`
and `captured_zero_gap=false` at once. Saturation is also visible through separate
frame photometry; the PNG-loss count alone does not identify its cause per ID.

At 20/50/100 ms sampling from zero, `[700,1100)` ms gives 20/8/4 minimum samples;
`[730,770)` gives 2/1/0; `[790,830)` gives 2/1/1. Linear ramps at 500–700 and
1100–1300 ms give 21/9/5 zeros, because 1100 ms is included. Counts are obtained
from actual observations, not by dividing duration by interval.

### Stage 4 demonstrations

All examples are 1000 × 1000, use E(t)=80+60t eV with t in seconds, and keep L/U=0/1.
The selected preview windows are shorter than the final two-second benchmark.

| Config in `configs/synthetic-benchmark/` | Frames | Expected result |
| --- | ---: | --- |
| `attenuation_h18_long.json` | 7 | 600–1200 ms every 100 ms; four zero observations, baseline on both sides |
| `attenuation_h18_short_missed.json` | 3 | 700/800/900 ms; `[730,770)` is entirely missed |
| `attenuation_h18_short_captured.json` | 3 | Same observations; `[790,830)` has one zero at 800 ms |
| `attenuation_h18_ramp_weak.json` | 7 | Two independent IDs: a zero ramp and a 5% ramp; both include minimum at 1100 ms |
| `attenuation_h54_satellite.json` | 3 | `h+1_k+0_s0` disappears at 800 ms beside its bright parent; all 54 IDs and optional masks persist |

Run the following with a new output name (the `runs` parent must exist), substituting
any config from the table:

```sh
python scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/attenuation_h18_long.json --output runs/h18-long-gap
python scripts/generate_tracking_benchmark.py --verify runs/h18-long-gap
python -X utf8 -m unittest trackleed.tests.test_synthetic_attenuation trackleed.tests.test_synthetic_attenuation_package trackleed.tests.test_synthetic_frozen -v
```

1. Generate and verify one example; read `coverage.json` for its counts and context.
2. In TrackLEED open `sequences/<schedule>/images` with File → Open Directory.
3. Compare frames around the gap using the times in `frames.csv`.
4. Compare the two short examples: the missed one contains no zero observation;
   the captured one contains a single zero at 800 ms.
5. For H54 inspect the satellite ID and its parent in `reference.csv`; the parent
   remains bright and the satellite's geometry mask persists while its signal
   mask is empty. Its position during the gap is a reference, not an image measurement.


## Screen, occlusion and frame limits (version 5)

Stage 05 adds fixed screen/occluder masks, off-frame reference positions,
separate W0-W3 signal losses and verified apparatus rasters. See the
[apparatus guide](synthetic-apparatus.md) for configuration, four 1000 x 1000
demonstrations, commands and expected results. Versions 1-4 preserve their
existing behavior.
