#!/usr/bin/env python3
"""Freeze the four-case source package for supplementary Figure S3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Iterable


FIGURE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
UPSTREAM_ROOT = (
    REPOSITORY_ROOT / ".scratch/trackleed-physics-method-paper/artifacts/"
    "independent-scane-validation"
)

CASES = (
    ("scan_a_pair_1", "26a21", "pair1", "Scan A", "pair 1", 35),
    ("scan_a_pair_2", "26a21", "pair2", "Scan A", "pair 2", 35),
    ("scan_b_pair_1", "26a21b", "pair1", "Scan B", "pair 1", 46),
    ("scan_b_pair_2", "26a21b", "pair2", "Scan B", "pair 2", 46),
)
SOURCE_FILES = (
    "case_contract.csv",
    "pair_measurements.csv",
    "visibility_regimes.csv",
    "source_provenance.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError(f"Inconsistent CSV schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _upstream_paths(series_id: str, pair_id: str) -> tuple[Path, Path]:
    case_root = UPSTREAM_ROOT / series_id / pair_id
    return case_root / "pair_measurements.csv", case_root / "scan_measurements.csv"


def build(output_root: Path = FIGURE_ROOT) -> None:
    pair_rows: list[dict[str, object]] = []
    visibility_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    provenance_sources: dict[str, dict[str, str]] = {}

    for case_id, series_id, pair_id, series, pair, frame_count in CASES:
        pair_path, visibility_path = _upstream_paths(series_id, pair_id)
        source_pair_rows = read_csv(pair_path)
        source_visibility_rows = read_csv(visibility_path)
        case_rows.append(
            {
                "case_id": case_id,
                "series": series,
                "pair": pair,
                "frame_count": frame_count,
                "valid_pair_frames": len(source_pair_rows),
                "visibility_branch_frames": len(source_visibility_rows),
            }
        )
        for row in source_pair_rows:
            pair_rows.append(
                {
                    "case_id": case_id,
                    "series": series,
                    "pair": pair,
                    "frame": row["frame"],
                    "energy_ev": row["energy_ev"],
                    "pair_distance_px": row["pair_distance_px"],
                    "predicted_distance_px": row["predicted_distance_px"],
                    "distance_residual_px": row["distance_residual_px"],
                    "midpoint_x": row["midpoint_x"],
                    "midpoint_y": row["midpoint_y"],
                    "midpoint_dx_px": row["midpoint_dx_px"],
                    "midpoint_dy_px": row["midpoint_dy_px"],
                    "midpoint_drift_px": row["midpoint_drift_px"],
                }
            )
        for row in source_visibility_rows:
            visibility_rows.append(
                {
                    "case_id": case_id,
                    "series": series,
                    "pair": pair,
                    "frame": row["frame"],
                    "energy_ev": row["energy_ev"],
                    "branch": row["branch"],
                    "status": row["status"],
                    "fit_valid": row["fit_valid"].lower(),
                }
            )
        for role, path in (
            ("physics_pair_measurements", pair_path),
            ("frozen_visibility_regimes", visibility_path),
        ):
            provenance_sources[f"{case_id}:{role}"] = {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }

    output = output_root / "source_data"
    write_csv(output / "case_contract.csv", case_rows)
    write_csv(output / "pair_measurements.csv", pair_rows)
    write_csv(output / "visibility_regimes.csv", visibility_rows)
    write_json(
        output / "source_provenance.json",
        {
            "figure": "Figure S3",
            "schema_version": "1.0",
            "sources": provenance_sources,
        },
    )


def validate_source_package(
    source_directory: Path = SOURCE_DATA_DIRECTORY,
    *,
    check_upstream: bool = True,
) -> None:
    cases = read_csv(source_directory / "case_contract.csv")
    pairs = read_csv(source_directory / "pair_measurements.csv")
    visibility = read_csv(source_directory / "visibility_regimes.csv")
    if [row["case_id"] for row in cases] != [case[0] for case in CASES]:
        raise ValueError("Figure S3 frozen case order changed")
    if [int(row["valid_pair_frames"]) for row in cases] != [17, 20, 25, 27]:
        raise ValueError("Figure S3 valid-pair denominators changed")
    if len(pairs) != 89 or len(visibility) != 324:
        raise ValueError("Figure S3 source cardinality changed")
    if {row["status"] for row in visibility} != {
        "visible_fit",
        "iv_visibility_gap",
        "beam_onset_or_termination",
        "apparatus_absence",
    }:
        raise ValueError("Figure S3 visibility regime set changed")
    for row in pairs:
        values = [
            float(row[field])
            for field in (
                "pair_distance_px",
                "predicted_distance_px",
                "distance_residual_px",
                "midpoint_dx_px",
                "midpoint_dy_px",
                "midpoint_drift_px",
            )
        ]
        if not all(value == value for value in values):
            raise ValueError("Figure S3 contains a non-finite plotted value")
        if float(row["midpoint_drift_px"]) < 0:
            raise ValueError("Figure S3 midpoint drift became negative")

    provenance = json.loads(
        (source_directory / "source_provenance.json").read_text(encoding="utf-8")
    )
    if check_upstream:
        for name, source in provenance["sources"].items():
            path = (REPOSITORY_ROOT / source["path"]).resolve()
            if REPOSITORY_ROOT.resolve() not in path.parents:
                raise ValueError(f"Figure S3 provenance escapes repository: {name}")
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"Figure S3 upstream hash mismatch: {name}")


def validate() -> None:
    validate_source_package()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s3-source-") as tmp:
        rebuilt_root = Path(tmp) / "figure_s3_ag111_midpoint_drift"
        build(rebuilt_root)
        for name in SOURCE_FILES:
            if sha256_file(SOURCE_DATA_DIRECTORY / name) != sha256_file(
                rebuilt_root / "source_data" / name
            ):
                raise ValueError(f"Non-deterministic Figure S3 source: {name}")
    print("Figure S3 source validation passed: 4 cases and 89 valid pair frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.validate_only:
        build()
    validate()


if __name__ == "__main__":
    main()
