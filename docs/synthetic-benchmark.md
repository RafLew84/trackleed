# Static H6 synthetic benchmark (stage 1)

This entry point creates a self-contained tracking benchmark alongside the existing
segmentation-training generator. It currently supports one stationary H6 scene,
one constant background, one deterministic realization and one observation schedule.
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
This stage rejects H18/H54, motion, noise, imported backgrounds, apparatus masks
and optional instance-mask export. The entire 4-sigma support must fit inside
the frame's pixel-center rectangle; clipping is introduced in stage 5.
A profile with no sampled support (W0 = 0) is an error, not a disappearance.
Rendering also rejects values which cannot be represented as finite float32.

## Package and numerical contract

- `manifest.json`: type `trackleed.synthetic-benchmark`, schema version 1,
  complete status, scene/condition/realization/schedule IDs, counts, generator
  provenance and hashes of every other file.
- `input.json`: the actual input configuration; `scene.json`: its expanded values.
- `schema.json`: versioned column types/units, coordinates, rendering and encoding rules.
- `frames.csv`: observation IDs, int64 times, optional energy, predecessor and
  time/energy deltas, image paths, file/pixel hashes and clipping/quantization counts.
- `reference.csv`: every frame × reflection, including zero-amplitude reflections.
- `rendered/frame_XXXXXX.tiff`: unnormalized, unclipped float32.
- `images/frame_XXXXXX.png`: uint8 grayscale with one fixed photometric mapping.
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
are unsupported in stage 1, W1=W2=W3=W0, the retained fraction is 1 and geometric
loss fractions are 0, also for A=0.

Every ID persists in every frame. The first predecessor and all first deltas are
missing, represented by empty CSV fields; subsequent positional deltas are zero.
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
python -m unittest trackleed.tests.test_synthetic_config trackleed.tests.test_synthetic_scene trackleed.tests.test_synthetic_package trackleed.tests.test_synthetic_cli -v
python -m mypy --follow-imports=silent trackleed/synthetic scripts/generate_tracking_benchmark.py scripts/export_segmentation_to_png.py
python scripts/validate_current_workflow.py
```

The repository validator includes the code-size guardrail and configured mandatory
tests; acquisition-dependent publication tests remain outside its CI suite.
Small fixtures check independent analytic positions/pixels, subpixel centers,
support boundaries, quantization, preserved identities, real file round trips,
legacy compatibility and publication failures. The supplied 1000 × 1000 example
is the user demonstration.
