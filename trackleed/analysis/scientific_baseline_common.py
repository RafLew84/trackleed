"""Shared types and file/schema operations for the scientific baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from trackleed.analysis.scientific_baseline_contract import (
    EXPECTED_CASE_MATRIX,
    EXPECTED_CASE_COUNT,
    EXPECTED_EVALUABLE_GAPS,
    EXPECTED_VISIBLE_OPPORTUNITIES,
    METHODS,
    SCHEMA_VERSION,
    STUDY,
)

OPPORTUNITY_FIELDS = [
    "opportunity_key",
    "image_artifact_id",
    "image_source_path",
    "image_sha256",
    "case_id",
    "series_id",
    "pair_id",
    "frame",
    "energy_ev",
    "branch",
    "baseline_status",
]
GAP_FIELDS = [
    "gap_key",
    "case_id",
    "series_id",
    "pair_id",
    "branch",
    "direction",
    "start_energy_ev",
    "end_energy_ev",
    "frame_count",
    "reference_energy_ev",
]
GENERATED_NAMES = {
    "frozen_inputs",
    "gaps.csv",
    "input_manifest.json",
    "measurement_contract.json",
    "opportunities.csv",
    "package_manifest.json",
    "README.md",
}
README_TEMPLATE = Path(__file__).with_name("scientific_baseline_README.md")


@dataclass(frozen=True)
class PackageReport:
    case_count: int
    method_count: int
    raw_image_count: int
    visible_opportunity_count: int
    evaluable_gap_count: int


class BaselineValidationError(ValueError):
    """The frozen baseline package or one of its linked sources changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys_sha256(keys: list[str]) -> str:
    canonical = "".join(f"{key}\n" for key in sorted(keys))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _source_location(workspace_root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError(f"Source path must be relative: {relative_path}")
    workspace = workspace_root.resolve()
    resolved = (workspace / requested).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Source path escapes the workspace: {relative_path}")
    return resolved


def _resolve_source(workspace_root: Path, relative_path: str) -> Path:
    resolved = _source_location(workspace_root, relative_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _schema(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        columns, rows = _read_csv(path)
        return {
            "format": "csv",
            "columns": columns,
            "row_count": len(rows),
        }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "format": "json",
            "top_level_type": type(payload).__name__,
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
        }
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            return {
                "format": "npz",
                "arrays": {
                    name: {
                        "dtype": str(arrays[name].dtype),
                        "shape": list(arrays[name].shape),
                    }
                    for name in sorted(arrays.files)
                },
            }
    if suffix in {".tif", ".tiff"}:
        return {"format": "tiff"}
    return {"format": suffix.removeprefix(".") or "binary"}


def _artifact_record(
    *,
    artifact_id: str,
    role: str,
    source: Path,
    workspace_root: Path,
    package_path: str | None = None,
    case_id: str | None = None,
    case_ids: list[str] | None = None,
    method: str | None = None,
    pair_id: str | None = None,
    series_id: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "artifact_id": artifact_id,
        "role": role,
        "source_path": source.relative_to(workspace_root.resolve()).as_posix(),
        "source_sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "schema": _schema(source),
    }
    optional = {
        "package_path": package_path,
        "case_id": case_id,
        "case_ids": case_ids,
        "method": method,
        "pair_id": pair_id,
        "series_id": series_id,
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    return record


def _copy_frozen_input(source: Path, package_dir: Path, relative_path: str) -> None:
    destination = package_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _load_inventory(package_dir: Path) -> dict[str, Any]:
    inventory_path = package_dir / "source_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported source-inventory schema version")
    if payload.get("study") != STUDY:
        raise ValueError(f"Expected study {STUDY!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CASE_COUNT} cases")
    case_matrix = {
        (
            str(case.get("case_id", "")),
            str(case.get("series_id", "")),
            str(case.get("pair_id", "")),
        )
        for case in cases
        if isinstance(case, dict)
    }
    if case_matrix != EXPECTED_CASE_MATRIX:
        raise ValueError(
            "Cases must match the exact frozen case matrix: "
            "Scan A/B × pair 1/2"
        )
    return payload


def _gap_rows(case: dict[str, Any], summary_path: Path) -> list[dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference_energy = float(summary["reference_energy_ev"])
    rows = []
    for gap in summary.get("visibility_gaps", []):
        if gap.get("classification") != "iv_visibility_gap":
            continue
        start = float(gap["start_energy_ev"])
        end = float(gap["end_energy_ev"])
        if end < reference_energy:
            direction = "low"
        elif start > reference_energy:
            direction = "high"
        else:
            raise ValueError(
                f"Gap spans the reference energy in {case['case_id']}: {start}–{end}"
            )
        rows.append(
            {
                "gap_key": (
                    f"{case['case_id']}|{gap['branch']}|"
                    f"{start:.10g}|{end:.10g}"
                ),
                "case_id": case["case_id"],
                "series_id": case["series_id"],
                "pair_id": case["pair_id"],
                "branch": gap["branch"],
                "direction": direction,
                "start_energy_ev": format(start, ".10g"),
                "end_energy_ev": format(end, ".10g"),
                "frame_count": int(gap["frame_count"]),
                "reference_energy_ev": format(reference_energy, ".10g"),
            }
        )
    return rows


def _opportunity_rows(
    case: dict[str, Any],
    baseline_path: Path,
    image_records_by_frame: list[dict[str, object]],
) -> list[dict[str, object]]:
    columns, rows = _read_csv(baseline_path)
    required = {"frame", "energy_ev", "branch", "status"}
    if not required.issubset(columns):
        raise ValueError(
            f"{case['case_id']} baseline is missing columns: {sorted(required - set(columns))}"
        )
    opportunities = []
    for row in rows:
        if row["status"] != "visible_fit":
            continue
        frame = int(row["frame"])
        if frame < 0 or frame >= len(image_records_by_frame):
            raise ValueError(
                f"{case['case_id']} frame {frame} has no frozen raw-image identity"
            )
        image_record = image_records_by_frame[frame]
        branch = row["branch"]
        opportunities.append(
            {
                "opportunity_key": f"{case['case_id']}|frame={frame}|branch={branch}",
                "image_artifact_id": image_record["artifact_id"],
                "image_source_path": image_record["source_path"],
                "image_sha256": image_record["source_sha256"],
                "case_id": case["case_id"],
                "series_id": case["series_id"],
                "pair_id": case["pair_id"],
                "frame": frame,
                "energy_ev": row["energy_ev"],
                "branch": branch,
                "baseline_status": "visible_fit",
            }
        )
    return opportunities
