"""Freeze and authenticate the controlled-translation source package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, TypedDict

from trackleed.analysis.controlled_translation_evidence import (
    METHOD_ORDER,
    PAIR_IDS,
    TRIAL_ORDER,
)

SOURCE_FILE_MAP = {
    "contract.json": "contract.json",
    "equivariance_results.csv": "equivariance_results.csv",
    "pair_invariance.csv": "pair_invariance.csv",
    "method_summary.csv": "method_summary.csv",
    "manifest.json": "upstream_manifest.json",
}


class ControlledTranslationSourceReport(TypedDict):
    spot_record_count: int
    pair_record_count: int
    trial_count: int
    method_count: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read controlled-translation JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Controlled-translation JSON is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_source_matrix(
    spot_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
) -> None:
    if len(spot_rows) != 576 or len(pair_rows) != 288:
        raise ValueError("Controlled-translation source cardinality changed")
    if {row["method"] for row in spot_rows} != set(METHOD_ORDER):
        raise ValueError("Controlled-translation method inventory changed")
    if {
        int(row["source_frame_order"]) for row in spot_rows
    } != set(TRIAL_ORDER):
        raise ValueError("Controlled-translation trial inventory changed")
    if {row["pair"] for row in pair_rows} != set(PAIR_IDS):
        raise ValueError("Controlled-translation pair inventory changed")


def _validate_upstream_output(origin_directory: Path) -> None:
    manifest = read_json(origin_directory / "manifest.json")
    output_hashes = manifest.get("output_files_sha256")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "completed"
        or not isinstance(output_hashes, dict)
    ):
        raise ValueError("Controlled-translation upstream manifest changed")
    for name in (
        "equivariance_results.csv",
        "pair_invariance.csv",
        "method_summary.csv",
    ):
        path = origin_directory / name
        if not path.is_file() or output_hashes.get(name) != sha256_file(path):
            raise ValueError(f"Controlled-translation upstream hash changed: {name}")
    contract_path = origin_directory / "contract.json"
    if manifest.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("Controlled-translation upstream contract changed")
    if read_json(contract_path).get("status") != "frozen_ready_for_runner":
        raise ValueError("Controlled-translation contract is not frozen")


def freeze_controlled_translation_source_data(
    origin_directory: Path,
    destination_directory: Path,
    *,
    workspace_root: Path,
) -> None:
    """Freeze the authenticated controlled-translation tables exactly once."""

    origin_directory = origin_directory.resolve()
    destination_directory = destination_directory.resolve()
    workspace_root = workspace_root.resolve()
    _validate_upstream_output(origin_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, str]] = {}
    origin_root = origin_directory.relative_to(workspace_root).as_posix()
    for origin_name, frozen_name in SOURCE_FILE_MAP.items():
        origin_path = origin_directory / origin_name
        frozen_path = destination_directory / frozen_name
        shutil.copyfile(origin_path, frozen_path)
        digest = sha256_file(origin_path)
        sources[frozen_name] = {
            "frozen_sha256": sha256_file(frozen_path),
            "origin_path": f"{origin_root}/{origin_name}",
            "origin_sha256": digest,
        }
    (destination_directory / "source_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_required_at_build": False,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_controlled_translation_source_data(
    source_directory: Path,
    *,
    origin_directory: Path | None = None,
) -> ControlledTranslationSourceReport:
    """Validate frozen source hashes and optionally their authenticated origin."""

    source_directory = source_directory.resolve()
    provenance = read_json(source_directory / "source_provenance.json")
    sources = provenance.get("sources")
    expected_names = set(SOURCE_FILE_MAP.values())
    actual_names = {
        path.name
        for path in source_directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    if (
        provenance.get("schema_version") != 1
        or provenance.get("origin_required_at_build") is not False
        or not isinstance(sources, dict)
        or set(sources) != expected_names
        or actual_names != expected_names | {"source_provenance.json"}
    ):
        raise ValueError("Controlled-translation source provenance changed")
    for frozen_name, record in sources.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"frozen_sha256", "origin_path", "origin_sha256"}
        ):
            raise ValueError(
                f"Controlled-translation source record changed: {frozen_name}"
            )
        frozen_hash = sha256_file(source_directory / frozen_name)
        if record["frozen_sha256"] != frozen_hash:
            raise ValueError(
                f"Frozen controlled-translation source changed: {frozen_name}"
            )
        for field in ("frozen_sha256", "origin_sha256"):
            digest = str(record[field])
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"Malformed controlled-translation hash: {frozen_name}"
                )

    upstream_manifest = read_json(source_directory / "upstream_manifest.json")
    upstream_hashes = upstream_manifest.get("output_files_sha256")
    if not isinstance(upstream_hashes, dict):
        raise ValueError("Frozen controlled-translation manifest changed")
    for name in (
        "equivariance_results.csv",
        "pair_invariance.csv",
        "method_summary.csv",
    ):
        if upstream_hashes.get(name) != sha256_file(source_directory / name):
            raise ValueError(
                f"Controlled-translation source is not manifest-linked: {name}"
            )
    if upstream_manifest.get("contract_sha256") != sha256_file(
        source_directory / "contract.json"
    ):
        raise ValueError("Frozen controlled-translation contract is not linked")

    spot_rows = _read_csv(source_directory / "equivariance_results.csv")
    pair_rows = _read_csv(source_directory / "pair_invariance.csv")
    _validate_source_matrix(spot_rows, pair_rows)
    if origin_directory is not None:
        origin_directory = origin_directory.resolve()
        _validate_upstream_output(origin_directory)
        for origin_name, frozen_name in SOURCE_FILE_MAP.items():
            origin_hash = sha256_file(origin_directory / origin_name)
            record = sources[frozen_name]
            if (
                record["origin_sha256"] != origin_hash
                or record["frozen_sha256"] != origin_hash
            ):
                raise ValueError(
                    f"Controlled-translation source diverges from origin: "
                    f"{origin_name}"
                )
    return ControlledTranslationSourceReport(
        spot_record_count=len(spot_rows),
        pair_record_count=len(pair_rows),
        trial_count=len(TRIAL_ORDER),
        method_count=len(METHOD_ORDER),
    )


__all__ = [
    "freeze_controlled_translation_source_data",
    "sha256_file",
    "validate_controlled_translation_source_data",
]
