"""Explicit authenticated refreeze path for the narrow scientific tracer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trackleed.analysis.scientific_tracer_common import (
    SCHEMA_VERSION,
    SchemaCompatibility,
    TRACER_ID,
    TracerValidationError,
    hash_manifest,
    sha256,
    write_json,
)
from trackleed.analysis.scientific_tracer_contract import (
    validate_input_hashes,
    validate_tracer_package,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
)
from trackleed.analysis.scientific_tracer_validation import (
    load_output_artifacts,
    validate_central_result,
    validate_output_manifest,
)


def _package_path(tracer_dir: Path, relative: str) -> Path:
    path = (tracer_dir / relative).resolve()
    if not path.is_relative_to(tracer_dir.resolve()):
        raise TracerValidationError(f"Frozen tracer path escapes package: {relative}")
    return path


def _package_manifest(tracer_dir: Path) -> dict[str, object]:
    files = {
        path.relative_to(tracer_dir).as_posix(): sha256(path)
        for path in sorted(tracer_dir.rglob("*"))
        if path.is_file()
        and path.name != "package_manifest.json"
        and "output" not in path.relative_to(tracer_dir).parts
    }
    return hash_manifest(files, immutable=True)


def refreeze_tracer_package(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
    authentication_path: Path,
) -> dict[str, Any]:
    """Refresh every dependent frozen record and hash in one explicit action.

    The proposed output must first match the currently frozen scientific record.
    Refreezing can therefore record an implementation/schema refresh, but cannot
    silently select a different scientific result.
    """

    workspace_root = workspace_root.resolve()
    tracer_dir = tracer_dir.resolve()
    output_dir = output_dir.resolve()
    authentication_path = authentication_path.resolve()
    contract_path = tracer_dir / "frozen" / "tracer_contract.json"
    contract = validate_tracer_package(
        workspace_root,
        tracer_dir,
        authentication_path=authentication_path,
        schema_compatibility=SchemaCompatibility.RETAINED_FIELDS,
    )
    validate_input_hashes(workspace_root, tracer_dir, contract)

    validate_output_manifest(output_dir)
    artifacts, _ = load_output_artifacts(output_dir)
    validate_central_result(
        tracer_dir,
        contract,
        artifacts,
        workspace_root,
        schema_compatibility=SchemaCompatibility.RETAINED_FIELDS,
    )

    output_to_expected = {
        "candidate_records.csv": "candidate_records",
        "evaluation_records.csv": "evaluation_records",
        "pair_records.csv": "pair_records",
        "gap_records.csv": "gap_records",
    }
    for output_name, expected_name in output_to_expected.items():
        expected_path = _package_path(
            tracer_dir, contract["expected_files"][expected_name]
        )
        expected_path.write_bytes((output_dir / output_name).read_bytes())
    expected_summary = _package_path(
        tracer_dir, contract["expected_files"]["summary"]
    )
    write_json(expected_summary, artifacts.summary)

    summary = artifacts.summary
    contract["expected_output_schemas"] = {
        "candidate_records.csv": list(CANDIDATE_FIELDS),
        "evaluation_records.csv": list(EVALUATION_FIELDS),
    }
    contract["expected_counts"] = {
        "branch_consistent_measurements": int(
            summary["branch_consistent_measurements"]
        ),
        "candidate_records": len(artifacts.candidates),
        "evaluated_gaps": len(artifacts.gaps),
        "evaluation_records": len(artifacts.evaluations),
        "geometry_inconsistent_accepted_fits": int(
            summary["geometry_inconsistent_accepted_fits"]
        ),
        "jointly_consistent_pair_frames": len(artifacts.pairs),
        "recovered_gaps": int(
            summary["post_gap_recovery"]["recovered_gap_count"]
        ),
    }
    contract["historical_sources"] = {
        "central_records_sha256": sha256(
            _package_path(
                tracer_dir, contract["expected_files"]["central_records"]
            )
        ),
        "gap_records_sha256": sha256(
            _package_path(tracer_dir, contract["expected_files"]["gap_records"])
        ),
        "pair_records_sha256": sha256(
            _package_path(tracer_dir, contract["expected_files"]["pair_records"])
        ),
        "summary_sha256": sha256(expected_summary),
    }
    write_json(contract_path, contract)

    manifest_path = tracer_dir / "package_manifest.json"
    write_json(manifest_path, _package_manifest(tracer_dir))
    authentication = {
        "package_manifest_sha256": sha256(manifest_path),
        "schema_version": SCHEMA_VERSION,
        "tracer_id": TRACER_ID,
    }
    authentication_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(authentication_path, authentication)
    return authentication


__all__ = ["refreeze_tracer_package"]
