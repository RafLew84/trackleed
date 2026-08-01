"""Freeze and authenticate publication source data from sensitivity output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


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
    for source_name in SOURCE_FILE_MAP:
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
    }
    (destination_directory / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def validate_sensitivity_source_data(source_directory: Path) -> dict[str, Any]:
    """Validate a self-contained frozen sensitivity source package."""

    source_directory = source_directory.resolve()
    provenance = read_json(source_directory / "source_provenance.json")
    sources = provenance.get("sources")
    expected_names = set(SOURCE_FILE_MAP.values())
    actual_names = {path.name for path in source_directory.iterdir() if path.is_file()}
    if (
        provenance.get("schema_version") != SCHEMA_VERSION
        or provenance.get("origin_required_at_build") is not False
        or not isinstance(provenance.get("sensitivity_id"), str)
        or not isinstance(sources, dict)
        or set(sources) != expected_names
        or actual_names != expected_names | {"source_provenance.json"}
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
    return provenance


__all__ = [
    "SOURCE_FILE_MAP",
    "freeze_sensitivity_source_data",
    "sha256_file",
    "validate_sensitivity_source_data",
]
