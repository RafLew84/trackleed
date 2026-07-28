# Figure 1 - representative measurement-chain case

Status: representative case frozen; final figure not yet rendered.

This directory freezes the real LEED image, one canonical spot, the five
archived proposal coordinates, and the five common Gaussian measurements that
will anchor the reader-facing measurement-chain schematic.

## Selection

The case was selected without inspecting tracker outputs:

1. take the predeclared diagnostic probe whose reason is
   `evaluation_range_end`;
2. take the lowest canonical core spot ID.

This gives the final image of the 701-image Cu(111) sequence, acquired at
92 °C and 56 eV, and spot 1. The frame is already a metadata-selected
diagnostic anchor, and spot 1 is selected by identity order rather than by
tracker behavior.

All five methods process the same 701-image sequence and the same selected
spot. They start at image 0 from the same Gaussian-refined seed center
(`x = 140.7950 px`, `y = 712.6157 px`). Point methods receive that center;
box- and mask-based methods receive the shared `48 x 48 px` box centered on
it. The seed is an initialization prompt, not ground truth, and correction or
reinitialization is forbidden.

After selection, the completeness check found:

- five of five native proposals available;
- five of five accepted common Gaussian fits;
- no manual annotation or reinitialization;
- maximum pairwise separation of the reduced native proposal coordinates:
  `5.599510 px`;
- maximum pairwise separation of the common Gaussian centers: `0.002306 px`.

These separations describe the selected case; they were not used to select it
and do not constitute a qualitative method ranking.

## Native-geometry limitation

The publication run retained the native point for LocoTrack, the accepted
Lucas-Kanade point, and the EasyLEED Kalman position. For OSTrack it retained
the bbox center after the frozen `x+w/2, y+h/2` reduction, but not the bbox
width and height. For DAM4SAM it retained the mask centroid, but not the mask
pixels.

Consequently, the current files support an empirical comparison of proposal
coordinates, ROIs, and Gaussian centers. They do **not** support drawing the
actual OSTrack bbox or DAM4SAM mask for this frame. The final renderer must
either:

- consume a targeted native-geometry export for those two methods; or
- mark the bbox and mask glyphs explicitly as schematic.

It must not reconstruct and present invented geometry as a native model output.

## Files

- `contract.json` - frozen selection, completeness, descriptive geometry, claim
  limit, and archive gap.
- `freeze_case.py` - deterministic source-case extractor and validator.
- `test_freeze_case.py` - selection, completeness, and archive-gap tests.
- `source_data/representative_initialization.json` - shared sequence, spot,
  seed center, prompt box, and no-reinitialization contract.
- `source_data/representative_method_outputs.csv` - one row per method with
  proposal coordinate, ROI, fit center, and fit diagnostics.
- `source_data/representative_spot_crop.png` - lossless 128 x 128 crop from the
  raw TIFF.
- `source_data/representative_spot_crop.json` - crop coordinates and image
  provenance.
- `source_data/source_provenance.json` - hashes of all upstream artifacts.
- `manifest.json` - code, contract, and committed source-data hashes.

## Rebuild

When the frozen local publication artifacts are available:

```bash
python freeze_case.py
```

Validate the committed case without rewriting it:

```bash
python freeze_case.py --validate-only
```

This full validation checks recorded upstream hashes and compares the committed
files byte-for-byte with a fresh deterministic rebuild. When the upstream
artifacts are unavailable, only the integrity of the committed figure package
can be checked:

```bash
python freeze_case.py --offline-integrity-only
```

Run the contract tests:

```bash
python test_freeze_case.py
```
