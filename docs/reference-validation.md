# Local reference validation

The regular fail-closed gate is shared by local runs and CI; see
[workflow validation](workflow-validation.md) for setup and included checks:

```powershell
python -X utf8 scripts/validate_current_workflow.py
```

Real acquisition data is accepted only through an explicit local manifest and an
explicit report path under the repository's ignored `.scratch` directory:

```powershell
python -X utf8 scripts/validate_current_workflow.py `
  --reference-manifest .scratch/local-reference-manifest.json `
  --reference-report .scratch/local-reference-report.json
```

The command never downloads data and has no synthetic fallback. Missing inputs,
dependencies, approved frames, or either CLI path make the run fail.

## Manifest schema

Paths are relative to the declared `reference_root`, which is itself relative to
the manifest. Inputs must contain two benchmark sessions with exactly 4 and 20
tracks and enough source frames for the canonical 120-frame sample. The combined
golden selection must contain 25–35 distinct selections, and its count must match
the manual approval record.

```json
{
  "schema_version": 1,
  "reference_root": "../local-data",
  "golden_approval": {
    "status": "manually_approved",
    "frame_count": 30,
    "criteria": {
      "minimum_method_available_frames": 1,
      "minimum_temporal_pairs": 1,
      "max_center_error_px": 25.0,
      "max_temporal_jump_px": 15.0,
      "max_spot_removal_shift_px": 5.0,
      "max_ransac_seed_spread_px": 1.0
    }
  },
  "benchmark": {
    "four_track_session": "benchmark/four.lts.json",
    "twenty_track_session": "benchmark/twenty.lts.json"
  },
  "series": [
    {
      "name": "approved-series-a",
      "session": "series-a/analyzed_session.lts.json",
      "analysis_manifest": "series-a/analysis_manifest.json",
      "golden_frame_ids": [1, 20, 40]
    }
  ]
}
```

The local report includes code and input hashes, environment metadata, real-series
self-consistency, golden fit and center errors, all center-method metrics, spot
removal, explicit refusals, multi-seed RANSAC sensitivity, recovery outcomes, and
the canonical benchmark's raw repetitions, medians, MAD, and blocking status.
The manually approved criteria are mandatory and blocking; a metric with no
comparable observations cannot pass by producing an empty report field.
Neither manifests, acquisition data, golden frames, nor reports belong in Git or
GitHub Actions artifacts.
