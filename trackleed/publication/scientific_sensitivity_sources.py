"""Freeze and authenticate publication source data from sensitivity output."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from typing import Any, TypedDict

from trackleed.analysis.scientific_panel_common import METHODS


SCHEMA_VERSION = 1
SOURCE_FILE_MAP = {
    "sensitivity_results.csv": "sensitivity_results.csv",
    "rejection_cause_counts.csv": "rejection_cause_counts.csv",
    "gate_margin_audit.csv": "gate_margin_audit.csv",
    "summary.json": "sensitivity_summary.json",
    "provenance.json": "sensitivity_provenance.json",
    "output_manifest.json": "sensitivity_output_manifest.json",
    "validation_report.json": "sensitivity_validation_report.json",
}
DERIVED_GATE_MARGIN_FILE = "gate_margin_summary.csv"
DERIVATION_INPUTS = ("evaluation_records.csv",)
GATE_MARGIN_SPECS = (
    ("proposal_to_fit_margin_px", "proposal-to-fit", "px", 15.0),
    ("r_squared_margin", "R-squared", "dimensionless", 0.15),
    ("amplitude_margin", "amplitude", "fit amplitude", 0.012),
    ("sigma_x_lower_margin_px", "sigma-x lower", "px", 12.4),
    ("sigma_x_upper_margin_px", "sigma-x upper", "px", 12.4),
    ("sigma_y_lower_margin_px", "sigma-y lower", "px", 12.4),
    ("sigma_y_upper_margin_px", "sigma-y upper", "px", 12.4),
    ("branch_margin_px", "branch", "px", 18.0),
)


class GateMarginSummaryRecord(TypedDict):
    method: str
    margin_id: str
    display_label: str
    unit: str
    normalizer: float
    finite_margin_n: int
    negative_count: int
    minimum: float
    p05: float
    median: float
    p95: float
    maximum: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read sensitivity source JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Sensitivity source JSON is not an object: {path}")
    return value


def _validate_upstream_sensitivity_output(output_directory: Path) -> str:
    manifest_path = output_directory / "output_manifest.json"
    manifest = read_json(manifest_path)
    hashes = manifest.get("files_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(
        hashes, dict
    ):
        raise ValueError("Sensitivity output manifest changed")
    for source_name in (*SOURCE_FILE_MAP, *DERIVATION_INPUTS):
        path = output_directory / source_name
        if not path.is_file():
            raise ValueError(f"Sensitivity source is missing: {source_name}")
        if source_name in hashes and sha256_file(path) != hashes[source_name]:
            raise ValueError(f"Sensitivity source hash changed: {source_name}")
    validation = read_json(output_directory / "validation_report.json")
    if (
        validation.get("fail_closed") is not True
        or validation.get("output_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Sensitivity validation report changed")
    return str(manifest["sensitivity_id"])


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _gate_margin_summary_bytes(output_directory: Path) -> bytes:
    evaluation_path = output_directory / "evaluation_records.csv"
    with evaluation_path.open(newline="", encoding="utf-8") as handle:
        evaluations = list(csv.DictReader(handle))
    rows: list[GateMarginSummaryRecord] = []
    for method in METHODS:
        selected = [
            row
            for row in evaluations
            if row["variant_id"] == "roi_48_proposal_15"
            and row["method"] == method
        ]
        for margin_id, label, unit, normalizer in GATE_MARGIN_SPECS:
            values = [
                float(row[margin_id])
                for row in selected
                if row[margin_id] and math.isfinite(float(row[margin_id]))
            ]
            if not values:
                raise ValueError(
                    f"No finite central gate margins: {method}/{margin_id}"
                )
            rows.append(
                GateMarginSummaryRecord(
                    method=method,
                    margin_id=margin_id,
                    display_label=label,
                    unit=unit,
                    normalizer=normalizer,
                    finite_margin_n=len(values),
                    negative_count=sum(value < 0.0 for value in values),
                    minimum=min(values),
                    p05=_quantile(values, 0.05),
                    median=_quantile(values, 0.50),
                    p95=_quantile(values, 0.95),
                    maximum=max(values),
                )
            )
    if len(rows) != len(METHODS) * len(GATE_MARGIN_SPECS):
        raise ValueError("Gate-margin summary matrix changed")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(GateMarginSummaryRecord.__annotations__),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def freeze_sensitivity_source_data(
    output_directory: Path,
    destination_directory: Path,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Copy validated sensitivity results into a publication source package."""

    output_directory = output_directory.resolve()
    destination_directory = destination_directory.resolve()
    workspace_root = workspace_root.resolve()
    sensitivity_id = _validate_upstream_sensitivity_output(output_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    for source_name, frozen_name in SOURCE_FILE_MAP.items():
        shutil.copyfile(
            output_directory / source_name,
            destination_directory / frozen_name,
        )
    gate_margin_path = destination_directory / DERIVED_GATE_MARGIN_FILE
    gate_margin_path.write_bytes(_gate_margin_summary_bytes(output_directory))
    origin_directory = output_directory.relative_to(workspace_root).as_posix()
    sources = {
        frozen_name: {
            "frozen_sha256": sha256_file(destination_directory / frozen_name),
            "origin_path": f"{origin_directory}/{source_name}",
            "origin_sha256": sha256_file(output_directory / source_name),
        }
        for source_name, frozen_name in SOURCE_FILE_MAP.items()
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "sensitivity_id": sensitivity_id,
        "origin_required_at_build": False,
        "sources": sources,
        "derived_sources": {
            DERIVED_GATE_MARGIN_FILE: {
                "frozen_sha256": sha256_file(gate_margin_path),
                "derived_from": {
                    name: sha256_file(output_directory / name)
                    for name in DERIVATION_INPUTS
                },
                "builder": {
                    "path": "trackleed/publication/scientific_sensitivity_sources.py",
                    "sha256": sha256_file(
                        workspace_root
                        / "trackleed"
                        / "publication"
                        / "scientific_sensitivity_sources.py"
                    ),
                },
            }
        },
    }
    (destination_directory / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def validate_sensitivity_source_data(
    source_directory: Path,
    *,
    origin_directory: Path | None = None,
) -> dict[str, Any]:
    """Validate a frozen package and, when supplied, its authenticated origin."""

    source_directory = source_directory.resolve()
    provenance = read_json(source_directory / "source_provenance.json")
    sources = provenance.get("sources")
    derived_sources = provenance.get("derived_sources")
    expected_names = set(SOURCE_FILE_MAP.values())
    actual_names = {path.name for path in source_directory.iterdir() if path.is_file()}
    if (
        provenance.get("schema_version") != SCHEMA_VERSION
        or provenance.get("origin_required_at_build") is not False
        or not isinstance(provenance.get("sensitivity_id"), str)
        or not isinstance(sources, dict)
        or not isinstance(derived_sources, dict)
        or set(sources) != expected_names
        or set(derived_sources) != {DERIVED_GATE_MARGIN_FILE}
        or actual_names
        != expected_names
        | {DERIVED_GATE_MARGIN_FILE, "source_provenance.json"}
    ):
        raise ValueError("Sensitivity source provenance changed")
    for frozen_name, record in sources.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"frozen_sha256", "origin_path", "origin_sha256"}
        ):
            raise ValueError(f"Sensitivity source record changed: {frozen_name}")
        path = source_directory / frozen_name
        if sha256_file(path) != record["frozen_sha256"]:
            raise ValueError(f"Frozen sensitivity source changed: {frozen_name}")
        for hash_field in ("frozen_sha256", "origin_sha256"):
            digest = str(record[hash_field])
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Malformed sensitivity source hash: {frozen_name}")

    derived_record = derived_sources[DERIVED_GATE_MARGIN_FILE]
    if (
        not isinstance(derived_record, dict)
        or set(derived_record) != {"frozen_sha256", "derived_from", "builder"}
        or derived_record["frozen_sha256"]
        != sha256_file(source_directory / DERIVED_GATE_MARGIN_FILE)
        or not isinstance(derived_record["derived_from"], dict)
        or set(derived_record["derived_from"]) != set(DERIVATION_INPUTS)
        or not isinstance(derived_record["builder"], dict)
        or set(derived_record["builder"]) != {"path", "sha256"}
    ):
        raise ValueError("Derived gate-margin source changed")

    upstream_manifest = read_json(
        source_directory / "sensitivity_output_manifest.json"
    )
    upstream_hashes = upstream_manifest.get("files_sha256")
    if (
        upstream_manifest.get("sensitivity_id") != provenance["sensitivity_id"]
        or not isinstance(upstream_hashes, dict)
    ):
        raise ValueError("Frozen sensitivity output manifest changed")
    for source_name, frozen_name in SOURCE_FILE_MAP.items():
        if source_name in upstream_hashes and (
            sha256_file(source_directory / frozen_name) != upstream_hashes[source_name]
        ):
            raise ValueError(f"Frozen sensitivity source is not manifest-linked: {source_name}")
    validation = read_json(source_directory / "sensitivity_validation_report.json")
    if (
        validation.get("fail_closed") is not True
        or validation.get("output_manifest_sha256")
        != sha256_file(source_directory / "sensitivity_output_manifest.json")
    ):
        raise ValueError("Frozen sensitivity validation report changed")
    if origin_directory is not None:
        origin_directory = origin_directory.resolve()
        origin_sensitivity_id = _validate_upstream_sensitivity_output(
            origin_directory
        )
        if origin_sensitivity_id != provenance["sensitivity_id"]:
            raise ValueError("Frozen sensitivity origin identity changed")
        for source_name, frozen_name in SOURCE_FILE_MAP.items():
            origin_sha256 = sha256_file(origin_directory / source_name)
            record = sources[frozen_name]
            if (
                record["origin_sha256"] != origin_sha256
                or record["frozen_sha256"] != origin_sha256
            ):
                raise ValueError(
                    f"Frozen sensitivity source diverges from origin: {source_name}"
                )
        for source_name in DERIVATION_INPUTS:
            if (
                derived_record["derived_from"][source_name]
                != sha256_file(origin_directory / source_name)
            ):
                raise ValueError(
                    f"Gate-margin derivation input changed: {source_name}"
                )
        if (
            (source_directory / DERIVED_GATE_MARGIN_FILE).read_bytes()
            != _gate_margin_summary_bytes(origin_directory)
        ):
            raise ValueError("Derived gate-margin summary diverges from origin")
    return provenance


__all__ = [
    "SOURCE_FILE_MAP",
    "DERIVED_GATE_MARGIN_FILE",
    "GATE_MARGIN_SPECS",
    "GateMarginSummaryRecord",
    "freeze_sensitivity_source_data",
    "sha256_file",
    "validate_sensitivity_source_data",
]
