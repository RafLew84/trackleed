# Static backgrounds and camera-arm masks (schema v6)

Stage 06 adds static procedural backgrounds and prepared grayscale/RGB imports.
It also adds convex polygon occluders for the camera arm. Existing v1-v5 packages
retain their original schema and decoded pixels. No additional temporal noise,
automatic reflection removal, apparatus calibration or tracker ranking is added.

## Configuration

The optional `condition.background_model` requires `schema_version: 6`.
Omitting it keeps the constant `condition.background` and procedural apparatus
appearance from v5. All coordinates below use zero-based pixel centers, x right,
y down. The scene and procedural apparatus are declared in output coordinates;
they do not move or rescale implicitly when an imported image is transformed.

```json
{
  "kind": "procedural",
  "gradient": [0.10, -0.04],
  "glow": {"center_px": [560, 470], "sigma_px": [330, 300], "amplitude": 0.12},
  "texture_amplitude": 0.008,
  "texture_seed": 1729
}
```

Inside the allowed signal region the background is the base level plus
`gx*(x/max(width-1,1)-0.5) + gy*(y/max(height-1,1)-0.5)`, plus an optional
elliptical Gaussian glow and uniform spatial texture. Gradient coefficients
are full-image intensity differences. Glow widths are positive standard
deviations in pixels; its amplitude and texture amplitude are nonnegative.
Generated apparatus uses the declared outside-screen and occluder levels.
Texture is drawn once with a separate `background_texture` PCG64 stream and
an explicit uint64 seed. Its uniform values are in `[-amplitude, amplitude)`.
It remains fixed across observations and schedules and never changes geometry.

```json
{
  "kind": "import",
  "path": "assets/controlled-background-rgb.png",
  "gain": 0.00392156862745098,
  "offset": 0,
  "screen_mask": "assets/controlled-screen.png",
  "occluder_mask": "assets/controlled-occluders.png"
}
```

Paths are relative to the configuration's directory, not the process directory.
Use single-image numeric grayscale or RGB PNG/TIFF. Alpha, palette and animated
images are rejected. TIFF supports MINISBLACK grayscale with one sample per pixel
or contiguous RGB with exactly three samples. MINISWHITE and other photometric
interpretations are rejected explicitly. RGB is converted in float64 using exactly
`0.299*R + 0.587*G + 0.114*B` in the stored channel units, then `gray*gain+offset`.
Gain is required and positive; offset defaults to zero. There is no min-max
normalization, automatic division by the dtype maximum, clipping, or claim of
radiometric calibration. For uint8 a gain of `1/255` is a convenient explicit
choice. Floating-point TIFF may retain negative values. The prepared background
must be finite and representable in float32.

Optional source masks must be grayscale uint8 with values exactly 0 or 255 and
the same source dimensions as the background. An imported screen intersects
the procedural screen; imported occluders join the procedural occluder union.
These masks restrict added reflections without changing imported background
pixels, including the appearance and fixed noise of the apparatus.

## Explicit crop, uniform scaling and translation

Without `transform`, source dimensions must equal the output image dimensions.
An explicit transform may crop, scale uniformly and translate the input:

```json
"transform": {
  "crop_xywh": [100, 100, 600, 600],
  "scale": 1.25,
  "offset_px": [125, 125],
  "fill": 0.02
}
```

Crop uses integer x/y and positive width/height within the source. Null crop
selects the whole source. Scale is one positive scalar, preserving proportions;
offset is an output translation in pixels. Defaults are scale 1, offset `[0,0]`,
fill 0. For each axis the exact pixel-center mapping is
`output + 0.5 = offset + scale*(source - crop_origin + 0.5)`.
The crop occupies half-open pixel-cell bounds. Background interpolation is
bilinear with edge clamping inside the crop. Masks use nearest-neighbor sampling
with half ties toward the larger index. Output pixels outside transformed input
coverage receive `fill` in final image units and cannot receive reflection signal.
The same crop/scale/offset applies to both imported masks. No separate, hidden
mask alignment is performed. Set the ideal scene's center/lattice scale and any
procedural mask coordinates explicitly in the resulting output coordinate system.

## Camera arm

An apparatus occluder may have this v6 definition:

```json
{"kind":"polygon","vertices_px":[[307,0],[357,0],[507,445],[445,465]]}
```

Polygons must have 3-64 distinct vertices, be simple and strictly convex, with
coordinates within +/-1e9 px. Either winding is accepted, and boundary pixel
centers are included. Use a union of several polygons/circles or an imported
binary mask for an irregular outline. The output remains two semantic masks:
`screen.png` and the union `occluders.png`; their count does not limit the number
of apparatus elements.

The three stage 06 presets share a circular screen, central disk, slanted arm
and a small lateral attachment. Their manually selected geometry is inspired by
`data/24c17_Ann_Gr1_177_ud900C/BG/Ru0001-GrG_E0177_T0700.tiff`; it is an approximate
example, not a calibrated segmentation. The RGB preset uses entirely controlled
synthetic pixels, not that measurement. T0700/T0900 are never default inputs.
The blackened apparatus area in the ScanE files differs from the visible physical
outline in these color measurements; a mask must be matched to its own source.

## Retained inputs and verification

Every v6 package includes `background/prepared.tiff` (float64 used for composition),
`background.json` (conversion, transform, interpolation and random-stream rules),
and the screen/occluder masks. Rendered observations remain float32 TIFF and
fixed-scale uint8 PNG. Original imports are retained byte-for-byte as
`inputs/background.source`, `inputs/screen_mask.source` and/or
`inputs/occluder_mask.source`. They preserve their original PNG/TIFF encoding;
the `.source` suffix separates them from frame folders. Input metadata includes
file hashes, decoded pixel hashes, dtype, shape, format and measured range.

`input.json` preserves the user's configuration and original path declarations.
Expanded `scene.json` uses package-relative retained inputs. Verification reads
only those retained sources, recomputes the prepared background and masks, and
checks the actual frame/reference files. Moving the package or removing its
external sources therefore does not invalidate it. A changed prepared background
is rejected even if its file and pixel hashes have been replaced in the manifest.
The sources are captured once before publication and reused across schedules.

Successful import checks technical integrity, not scientific suitability. Review
any experimental input for residual reflections, labels and suitable masks before
using it in an experiment. Its existing texture and recorded noise remain fixed.

## Generate and inspect the demonstration

From the repository root, with a new destination for each command:

```sh
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/background_constant_h18.json --validate-only
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/background_constant_h18.json --output /path/to/new/constant-h18
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/background_glow_h18.json --output /path/to/new/glow-h18
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/background_import_h18.json --output /path/to/new/import-h18
python -X utf8 scripts/generate_tracking_benchmark.py --verify /path/to/new/import-h18
```

Each preset has five 1000 x 1000 observations of the same H18 translation, with
18 persistent reflection IDs. `h+1_k+0` stays at y=250 and crosses the arm at
x=340, 370, 405, 450 and 480 px: clear, partial, full, partial and clear.
Its amplitude stays 0.5. Other reflections can meet the screen independently.

1. Open `sequences/preview/images` from one generated package in TrackLEED.
2. Step through the five frames; follow the reflection at the coordinates above.
3. Compare the same frames from the other two backgrounds. Positions are unchanged.
4. Inspect `apparatus/occluders.png` and the target rows in `reference.csv`:
   full occlusion has zero retained signal and an empty signal mask, while ID,
   position and the nominal geometry mask remain defined.
5. Move a copy of the package and run `--verify` there; original assets are not needed.

Small checks (no GUI, model downloads, or experimental input dependency):

```sh
python -X utf8 -m unittest trackleed.tests.test_synthetic_backgrounds trackleed.tests.test_synthetic_background_package trackleed.tests.test_synthetic_frozen -v
python -m mypy --config-file mypy-generator.ini trackleed/synthetic scripts/generate_tracking_benchmark.py scripts/export_segmentation_to_png.py
python -X utf8 scripts/validate_current_workflow.py
```

The frozen v5 archive was captured before stage 06 edits. Its provenance is in
`trackleed/tests/fixtures/synthetic/benchmark-v5.md`; older fixtures are unchanged.
