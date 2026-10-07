# Screen, occlusion and frame limits (configuration version 5)

Version 5 permits reference centers and finite rendering support outside the
image. It adds optional `condition.apparatus`; omitted or null means the full
frame with no occluders and the existing constant background. Versions 1-4
keep their schemas, pixels and references, require support inside the frame,
and reject the new apparatus field, including null.

```json
"apparatus": {
  "screen": {"kind": "circle", "center_px": [499.5,499.5], "radius_px": 450},
  "occluders": [{"kind": "circle", "center_px": [499.5,499.5], "radius_px": 40}],
  "outside_screen_level": 0.02,
  "occluder_level": 0.04
}
```

Screen shapes are `full_frame` and `circle`. Occluders are circles or
`{"kind":"vertical_band","x_range_px":[650,690]}`. Coordinates and radii
are finite pixel units; radii must be positive. Circle boundaries and both band
endpoints are included at pixel centers. A zero-width band at an integer x
blocks that column. Multiple occluders form a union, so overlapping occluders
do not count losses twice. Masks stay fixed when the pattern drifts.

In a non-null apparatus object, the default screen is full-frame and the default
occluder list is empty. The explicit 450 px screen / 40 px central circle above
is the standard apparatus example from the specification; the demonstrations
below change it deliberately to exercise each limit.

These are procedural static shapes. File import and richer backgrounds belong
to stage 06. Unknown fields, moving masks, invalid geometry and unsupported
schema versions fail explicitly.

The background is `condition.background` inside the usable screen, defaults to
0.02 outside the screen and 0.04 inside a screened occluder. Both appearance
levels can be set to any finite nonnegative value. Outside-screen appearance
takes precedence where an occluder extends beyond the screen. The masks limit
only added reflection signal; occlusion does not zero the composed image.
PNG uses the same fixed lower/upper bounds in every frame. TIFF remains float32
without clipping; overflow of the composed float32 image is rejected.

## References and masks

Every observation still records every reflection and its latent position, ID,
amplitude, shape, parent and displacement. A visible fragment never replaces
the specified center. `center_in_frame` uses the closed pixel-center rectangle
`0 <= x <= width-1`, `0 <= y <= height-1`; this flag alone says nothing about
remaining signal. Full nominal geometry and analytic overlap remain defined
outside the image and through zero amplitude.

The same finite, unit-amplitude 4-sigma profile supplies four sample sums:

| Field | Included samples |
| --- | --- |
| W0 | Complete support, including outside the image |
| W1 | Inside the rectangular frame |
| W2 | Inside the frame and screen |
| W3 | Inside the frame and screen, not occluded |

The separate frame, screen and occluder loss fractions are `(W0-W1)/W0`,
`(W1-W2)/W0` and `(W2-W3)/W0`; the retained fraction is `W3/W0`. They sum to
one within numerical precision and remain defined when amplitude is zero.
W0=0 is unsupported spatial sampling and causes an error, including for a
reflection completely outside the frame. Positive amplitude with W3=0 is
geometric signal loss, not amplitude extinction. Float64 component pixels,
float32 component pixels, changed float32 image pixels and changed PNG pixels
remain separate diagnostics. Overlapping reflections add signal; they do not
occlude each other.

Optional 3-sigma geometry masks are limited only by the frame. Signal masks
add screen/occluder restrictions and become empty at zero amplitude. Boxes
bound actual nonzero mask pixels with exclusive upper endpoints; an empty mask
has a null box. A center at x=-14 with sigma=4 has no nominal mask inside the
frame but still has a positive rendered tail. Empty masks are not evidence of
complete loss of signal.

Every v5 package saves `apparatus/screen.png` and `apparatus/occluders.png`,
even when reflection-mask export is off. These are independent grayscale binary
rasters (0/255) with the image dimensions. The occluder mask includes the full
in-frame occluder region, independent of the screen. The input and expanded
configuration preserve the procedural definitions and appearance levels.
The manifest inventories and hashes the masks and their decoded pixels.
Verification reads them and checks the expected geometry, as well as all
images, optional instance masks, boxes and reference fields. Updating a hash
does not conceal a mask or reference that disagrees with the scene.

## Stage 05 demonstrations

All four configurations in `configs/synthetic-benchmark/` use 1000 x 1000 images,
fixed L/U=0/1, no added noise and separate instance masks. The three scans use
E(t)=80+60t eV, E0=100 eV and declared R0=600 px over 80-200 eV.
The preview schedules select a few key observations from this two-second scan.

| Configuration | Frames | Expected behavior |
| --- | ---: | --- |
| apparatus_screen_h18.json | 7 | H18 outer spots enter a circular screen of radius 340 px |
| apparatus_band_h6.json | 7 | H6 right spot crosses the inclusive 650-690 px band: partial at 0 ms, fully blocked at 500 ms, returned at 2000 ms |
| apparatus_frame_h6.json | 5 | H6 with 500 px lattice scale has two centers initially outside the frame; both enter as energy rises |
| apparatus_tail_h6.json | 1 | Left H6 spot at x=-14 has an empty 3-sigma mask and a nonzero 4-sigma tail |

From the repository root, use a new output directory with an existing parent:

```sh
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/apparatus_screen_h18.json --validate-only
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/apparatus_screen_h18.json --output runs/stage05-screen
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/stage05-screen
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/apparatus_band_h6.json --output runs/stage05-band
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/stage05-band
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/apparatus_frame_h6.json --output runs/stage05-frame
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/stage05-frame
python -X utf8 scripts/generate_tracking_benchmark.py --config configs/synthetic-benchmark/apparatus_tail_h6.json --output runs/stage05-tail
python -X utf8 scripts/generate_tracking_benchmark.py --verify runs/stage05-tail
```

1. Generate and verify the examples above, or use the supplied demonstration packages.
2. In TrackLEED choose **File -> Open Directory...** and open one package's
   `sequences/preview/images`. Only frame PNGs belong in that directory.
3. In the band example follow the right reflection before, during and after the
   gap. Find `h+1_k+0` in `reference.csv`: its position and ID persist while
   W3 and changed-pixel counts reach zero. The background strip stays visible.
4. Compare the screen and frame examples. Read the separate loss fractions;
   the initially off-frame centers remain outside coordinates in the reference.
5. For the tail example inspect `h-1_k+0` and its separate geometry/signal mask
   files. They are empty despite W3>0 and positive TIFF signal near the left edge;
   this faint tail can disappear after PNG quantization. Masks, TIFFs and CSVs
   are benchmark artifacts; the existing GUI opens the prepared frame PNGs.

## Validation

```sh
python -X utf8 -m unittest trackleed.tests.test_synthetic_apparatus trackleed.tests.test_synthetic_apparatus_package trackleed.tests.test_synthetic_frozen -v
python -X utf8 -m unittest discover -s trackleed/tests -p "test_synthetic_*.py" -v
python -X utf8 scripts/validate_current_workflow.py
```

Follow the shared [acceptance order](workflow-validation.md): small tests,
historical package checks, review and fixes, full validation, then demonstrations.
Historical v1-v4 archives are immutable. Stage 05 does not simulate apparatus
deformation, moving occluders, imported backgrounds or tracker performance.
