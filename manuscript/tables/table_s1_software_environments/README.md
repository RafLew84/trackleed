# Table S1: software, model, and environment records

This artifact builds supplementary Table S1 as three vertically stacked
blocks:

- software and model identity;
- configuration-record provenance, configuration hashes, and implementation
  hashes;
- retained execution environments.

The distinction between run-frozen and post-run records is explicit. The
Cu(111)/translation contracts contain run-frozen method-configuration hashes.
For the independent energy-scan variants of LocoTrack, OSTrack and
Lucas–Kanade, the controlled SAM2 family, and the additional EasyLEED regimes,
where no separate method-level configuration file was frozen during execution,
the artifact contains normalized configuration records derived from archived
copies of the sources that were actually executed. Table S1 labels those
records as post-run normalized rather than run-frozen. Full package
inventories were captured from the retained local environments on 2026-07-29,
after the runs. They document those preserved runtimes but do not support a
claim of bitwise reproducibility for GPU inference.

## Files

- `source_data/component_records.csv` is the frozen component-level source.
- `source_data/configuration_records/` contains the explicitly labelled
  post-run normalized configuration records.
- `source_data/environment_roles.csv` maps eight environments to their study
  roles and execution modes.
- `source_data/environment_snapshots/` contains path-free inventories for
  seven Python Conda environments and the ImageJ/Java runtime. Python records
  include all installed distributions, Conda package/build records, the
  Conda-history hash, and the detected PyTorch CUDA build.
- `source_data/system_record.json` records the retained host OS, GPU, driver,
  and VRAM.
- `source_data/upstream_contracts/` contains byte-identical method-contract
  snapshots, executed-source snapshots, the EasyLEED outputs supporting the
  normalized runtime settings, and a publication-safe extract of the domain
  installation record; `source_data/source_provenance.json` records and
  validates both the frozen and origin hashes where applicable.
- `capture_environments.py` recreates Python environment snapshots from
  retained Conda directories without storing personal or absolute paths.
- `build_table.py` validates the source contracts and creates two
  machine-readable CSV files and the three-block Markdown table.
- `output/table_s1_versions_hashes_environments.md` is copied exactly between
  the generated markers in `manuscript/supplementary-material.md`.
- `manifest.json` records source, capture-script, builder, environment, and
  output hashes.

## Rebuild and verify

From this directory:

```bash
python3 -m unittest test_build_table.py
python3 build_table.py
```

To recapture retained Python environments deliberately:

```bash
python3 capture_environments.py \
  --conda-root /path/to/anaconda3 \
  --output-dir source_data/environment_snapshots \
  --capture-date YYYY-MM-DD
```

Recapturing changes primary source data and therefore requires scientific
review; it is not part of the normal table rebuild.
