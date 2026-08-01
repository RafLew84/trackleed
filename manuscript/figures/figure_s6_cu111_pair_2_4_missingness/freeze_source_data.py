#!/usr/bin/env python3
"""Freeze and validate the Figure S6 source-data package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Sequence

from build_figure import (
    FIGURE_ROOT,
    PROVENANCE_PATH,
    SOURCE_DATA_PATH,
    load_and_validate_rows,
    sha256_file,
)


REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
UPSTREAM_PATH = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "cu111-pairwise-delta-r/pair_2_4_fit_diagnostics.csv"
)
BOOLEAN_FIELDS = (
    "exported_fit_present",
    "refit_returned_parameters",
    "r_squared_gate",
    "amplitude_gate",
    "center_gate",
    "sigma_gate",
    "refit_accepted",
)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return "" if value is None else value


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fields})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_upstream() -> list[dict[str, object]]:
    with UPSTREAM_PATH.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        {
            "pair": "2-4",
            "tracking_source": "earlier saved YOLO proposal trajectory",
            "inside_five_method_run": False,
            "frame_id": int(row["frame_id"]),
            "temperature_c": float(row["temperature_c"]),
            "energy_ev": float(row["energy_ev"]),
            "tracking_proposal_present": _as_bool(row["tracking_roi_present"]),
            **{field: _as_bool(row[field]) for field in BOOLEAN_FIELDS},
            "rejection_reason": row["rejection_reason"],
            "r_squared": float(row["r_squared"]),
            "amplitude": float(row["amplitude"]),
            "amplitude_threshold": 0.05,
        }
        for row in source_rows
    ]
    if [row["frame_id"] for row in rows] != list(range(356, 382)):
        raise ValueError("Figure S6 upstream frame set changed")
    return rows


def build(output_root: Path = FIGURE_ROOT) -> None:
    source_data_path = output_root / "source_data/pair_2_4_missingness_diagnostic.csv"
    provenance_path = output_root / "source_data/source_provenance.json"
    _write_csv(source_data_path, load_upstream())
    _write_json(
        provenance_path,
        {
            "schema_version": "1.0",
            "figure": "Figure S6",
            "source": {
                "evidence_role": "separate saved-proposal diagnostic",
                "origin_path": UPSTREAM_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "origin_sha256": sha256_file(UPSTREAM_PATH),
                "frozen_source_sha256": sha256_file(source_data_path),
            },
        },
    )


def validate(*, check_upstream: bool = True) -> None:
    load_and_validate_rows()
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    source = provenance["source"]
    if source["frozen_source_sha256"] != sha256_file(SOURCE_DATA_PATH):
        raise ValueError("Figure S6 frozen source hash mismatch")
    if check_upstream:
        origin = REPOSITORY_ROOT / source["origin_path"]
        if not origin.is_file() or sha256_file(origin) != source["origin_sha256"]:
            raise ValueError("Figure S6 upstream source hash mismatch")
        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure-s6-source-"
        ) as temporary:
            rebuilt = Path(temporary)
            build(rebuilt)
            for relative in (
                Path("source_data/pair_2_4_missingness_diagnostic.csv"),
                Path("source_data/source_provenance.json"),
            ):
                if sha256_file(FIGURE_ROOT / relative) != sha256_file(
                    rebuilt / relative
                ):
                    raise ValueError(f"Non-deterministic Figure S6 source: {relative}")
    print(
        "Figure S6 source validation passed: saved-proposal diagnostic is "
        "separate from the frozen five-method Cu(111) run."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--offline-integrity-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.validate_only:
        build()
    validate(check_upstream=not arguments.offline_integrity_only)


if __name__ == "__main__":
    main()
