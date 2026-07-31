"""Authentication and input contract for the complete Ag(111) panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trackleed.analysis.scientific_baseline_validation import validate_package
from trackleed.analysis.scientific_panel_common import (
    BASELINE_MANIFEST_PATH,
    BASELINE_PACKAGE,
    CASE_SPECS,
    LEGACY_PUBLICATION_ALIASES,
    METHODS,
    PANEL_AUTHENTICATION_PATH,
    PANEL_ID,
    PUBLISHED_AGGREGATE_RESULTS_PATH,
    PUBLISHED_CASE_RESULTS_PATH,
    SCHEMA_VERSION,
    TracerValidationError,
    panel_hash_manifest,
    read_json,
    relative_file,
    sha256,
)
from trackleed.analysis.scientific_panel_records import (
    AGGREGATE_REPORT_FIELDS,
    CASE_REPORT_FIELDS,
    PANEL_GAP_FIELDS,
    PANEL_PAIR_FIELDS,
)
from trackleed.analysis.scientific_tracer_contract import runtime_fingerprint
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
    read_raw_csv,
)


@dataclass(frozen=True)
class PreparedPanel:
    workspace_root: Path
    panel_dir: Path
    output_dir: Path
    contract: dict[str, Any]
    runtime: dict[str, str]


PANEL_SCHEMAS = {
    "candidate_records.csv": CANDIDATE_FIELDS,
    "evaluation_records.csv": EVALUATION_FIELDS,
    "pair_records.csv": PANEL_PAIR_FIELDS,
    "gap_records.csv": PANEL_GAP_FIELDS,
    "case_results.csv": CASE_REPORT_FIELDS,
    "aggregate_method_results.csv": AGGREGATE_REPORT_FIELDS,
}


def panel_package_manifest(panel_dir: Path) -> dict[str, object]:
    files = {
        path.relative_to(panel_dir).as_posix(): sha256(path)
        for path in sorted(panel_dir.rglob("*"))
        if path.is_file()
        and path.name != "package_manifest.json"
        and "output" not in path.relative_to(panel_dir).parts
    }
    return panel_hash_manifest(files, immutable=True)


def validate_panel_package(
    workspace_root: Path,
    panel_dir: Path,
    *,
    authentication_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the immutable expected records and their declared scope."""

    manifest_path = panel_dir / "package_manifest.json"
    authentication = read_json(
        authentication_path or workspace_root / PANEL_AUTHENTICATION_PATH
    )
    if (
        authentication.get("schema_version") != SCHEMA_VERSION
        or authentication.get("tracer_id") != PANEL_ID
        or authentication.get("package_manifest_sha256") != sha256(manifest_path)
    ):
        raise TracerValidationError("Frozen panel manifest contract changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != PANEL_ID
        or manifest.get("immutable") is not True
    ):
        raise TracerValidationError("Frozen panel package identity changed")
    expected_hashes = manifest.get("files_sha256")
    if not isinstance(expected_hashes, dict):
        raise TracerValidationError("Frozen panel manifest has no file hash map")
    actual = {
        path.relative_to(panel_dir).as_posix(): path
        for path in sorted(panel_dir.rglob("*"))
        if path.is_file()
        and path.name != "package_manifest.json"
        and "output" not in path.relative_to(panel_dir).parts
    }
    if set(actual) != set(expected_hashes):
        raise TracerValidationError("Frozen panel package inventory changed")
    for relative, path in actual.items():
        if sha256(path) != expected_hashes[relative]:
            raise TracerValidationError(
                f"Frozen panel package hash mismatch: {relative}"
            )

    contract = read_json(panel_dir / "frozen" / "panel_contract.json")
    expected_cases = [case.case_id for case in CASE_SPECS]
    scope = contract.get("scope")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("tracer_id") != PANEL_ID
        or not isinstance(scope, dict)
        or scope.get("methods") != list(METHODS)
        or scope.get("cases") != expected_cases
    ):
        raise TracerValidationError("Frozen panel scope changed")
    schemas = contract.get("expected_output_schemas")
    expected_files = contract.get("expected_files")
    if (
        contract.get("legacy_publication_aliases") != LEGACY_PUBLICATION_ALIASES
        or not isinstance(schemas, dict)
        or not isinstance(expected_files, dict)
    ):
        raise TracerValidationError("Frozen panel output contract is missing")
    for output_name, fields in PANEL_SCHEMAS.items():
        expected_name = output_name.removesuffix(".csv")
        expected_path = panel_dir / str(expected_files.get(expected_name, ""))
        frozen_fields, _ = read_raw_csv(expected_path)
        if schemas.get(output_name) != list(fields) or frozen_fields != list(fields):
            raise TracerValidationError(f"Frozen panel schema changed: {output_name}")

    counts = contract.get("expected_counts")
    required_counts = {
        "aggregate_method_results",
        "candidate_records",
        "case_results",
        "evaluation_records",
        "gap_records",
        "historical_central_records",
        "pair_records",
    }
    if (
        not isinstance(counts, dict)
        or set(counts) != required_counts
        or any(not isinstance(value, int) or value <= 0 for value in counts.values())
    ):
        raise TracerValidationError("Frozen panel expected counts are invalid")
    return contract


def validate_panel_inputs(
    workspace_root: Path,
    contract: Mapping[str, Any],
) -> None:
    expected_paths = {
        "scientific_baseline_manifest": BASELINE_MANIFEST_PATH.as_posix(),
        "published_case_results": PUBLISHED_CASE_RESULTS_PATH.as_posix(),
        "published_aggregate_results": PUBLISHED_AGGREGATE_RESULTS_PATH.as_posix(),
    }
    inputs = contract.get("frozen_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_paths):
        raise TracerValidationError("Frozen panel input set changed")
    for name, expected_path in expected_paths.items():
        record = inputs[name]
        if not isinstance(record, dict) or record.get("path") != expected_path:
            raise TracerValidationError(f"Frozen panel input path changed: {name}")
        path = relative_file(workspace_root, expected_path)
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise TracerValidationError(f"Frozen panel input hash mismatch: {name}")


def validate_panel_runtime(contract: Mapping[str, Any]) -> dict[str, str]:
    runtime = runtime_fingerprint()
    execution = contract.get("execution_contract")
    expected = execution.get("runtime") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or execution.get("device") != "cpu"
        or runtime != expected
    ):
        raise TracerValidationError(
            f"Panel runtime differs from frozen CPU environment: {runtime!r}"
        )
    return runtime


def prepare_panel(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
    *,
    require_absent_output: bool = False,
) -> PreparedPanel:
    workspace_root = workspace_root.resolve()
    panel_dir = panel_dir.resolve()
    output_dir = output_dir.resolve()
    if require_absent_output and output_dir.exists():
        raise FileExistsError(f"Refusing to replace panel output: {output_dir}")
    validate_package(workspace_root, workspace_root / BASELINE_PACKAGE)
    contract = validate_panel_package(workspace_root, panel_dir)
    validate_panel_inputs(workspace_root, contract)
    runtime = validate_panel_runtime(contract)
    return PreparedPanel(
        workspace_root=workspace_root,
        panel_dir=panel_dir,
        output_dir=output_dir,
        contract=contract,
        runtime=runtime,
    )


__all__ = [
    "PANEL_SCHEMAS",
    "PreparedPanel",
    "panel_package_manifest",
    "prepare_panel",
    "validate_panel_inputs",
    "validate_panel_package",
    "validate_panel_runtime",
]
