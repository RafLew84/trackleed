"""Fail-closed validation of reconstructed scientific tracer artifacts."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from trackleed.analysis.scientific_tracer_common import (
    PUBLISHED_CASE_RESULTS_PATH,
    SCHEMA_VERSION,
    SchemaCompatibility,
    TRACER_ID,
    TracerReport,
    TracerValidationError,
    hash_manifest,
    implementation_sources,
    read_json,
    sha256,
)
from trackleed.analysis.scientific_tracer_reconstruction import (
    TracerArtifacts,
    reader_facing,
    record_key,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
    GAP_FIELDS,
    PAIR_FIELDS,
    CandidateRecord,
    EvaluationRecord,
    GapRecord,
    PairRecord,
    parse_bool,
    read_csv,
    read_raw_csv,
)


VALIDATION_CHECKS = {
    "complete_candidate_records": "passed",
    "complete_evaluation_records": "passed",
    "continuous_value_tolerances": "passed",
    "frozen_input_hashes": "passed",
    "implementation_source_hashes": "passed",
    "reader_facing_rounding": "passed",
    "record_keys_and_counts": "passed",
}
OUTPUT_ARTIFACT_NAMES = (
    "candidate_records.csv",
    "evaluation_records.csv",
    "pair_records.csv",
    "gap_records.csv",
    "summary.json",
    "provenance.json",
)


def output_manifest(output_dir: Path) -> dict[str, object]:
    return hash_manifest(
        {
            name: sha256(output_dir / name) for name in OUTPUT_ARTIFACT_NAMES
        }
    )


def validate_output_manifest(output_dir: Path) -> None:
    manifest = read_json(output_dir / "output_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != TRACER_ID
    ):
        raise TracerValidationError("Tracer output manifest identity changed")
    expected_hashes = manifest.get("files_sha256")
    if not isinstance(expected_hashes, dict):
        raise TracerValidationError("Tracer output manifest has no file hash map")
    if set(expected_hashes) != set(OUTPUT_ARTIFACT_NAMES):
        raise TracerValidationError("Tracer output manifest inventory changed")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_ARTIFACT_NAMES) | {
        "output_manifest.json",
        "validation_report.json",
    }:
        raise TracerValidationError("Tracer output file inventory changed")
    for name, expected_hash in expected_hashes.items():
        if sha256(output_dir / name) != expected_hash:
            raise TracerValidationError(f"Tracer output hash mismatch: {name}")


def _float_matches(actual: object, expected: str, atol: float, rtol: float) -> bool:
    if expected == "":
        return actual is None or actual == ""
    expected_value = float(expected)
    if math.isnan(expected_value):
        try:
            return math.isnan(float(actual))
        except (TypeError, ValueError):
            return False
    try:
        return math.isclose(
            float(actual), expected_value, abs_tol=atol, rel_tol=rtol
        )
    except (TypeError, ValueError):
        return False


def _validate_expected_rows(
    expected_path: Path,
    actual_rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    atol: float,
    rtol: float,
    *,
    record_name: str | None = None,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> None:
    expected_fields, expected_rows = read_raw_csv(expected_path)
    schema_matches = expected_fields == list(fields)
    if schema_compatibility.compares_retained_fields:
        schema_matches = (
            bool(expected_fields)
            and len(expected_fields) == len(set(expected_fields))
            and set(expected_fields).issubset(fields)
        )
    if not schema_matches or len(expected_rows) != len(actual_rows):
        label = record_name or expected_path.name
        raise TracerValidationError(f"Frozen {label} record set changed")
    label = record_name or expected_path.name
    for row_index, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
        for field in expected_fields:
            expected_value = expected[field]
            actual_value = actual[field]
            if expected_value == "":
                matches = actual_value is None or actual_value == ""
            elif expected_value in {"True", "False"}:
                matches = bool(actual_value) == parse_bool(expected_value)
            else:
                try:
                    float(expected_value)
                except ValueError:
                    matches = str(actual_value) == expected_value
                else:
                    matches = _float_matches(actual_value, expected_value, atol, rtol)
            if not matches:
                raise TracerValidationError(
                    f"Frozen {label} record changed: row {row_index}/{field}"
                )


def _compare_nested(
    actual: object,
    expected: object,
    location: str,
    atol: float,
    rtol: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise TracerValidationError(f"Central {location} keys changed")
        for key, expected_value in expected.items():
            _compare_nested(
                actual[key], expected_value, f"{location}.{key}", atol, rtol
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise TracerValidationError(f"Central {location} changed")
        return
    if isinstance(expected, (int, float)):
        if isinstance(expected, int):
            matches = actual == expected
        else:
            matches = math.isclose(
                float(actual), expected, abs_tol=atol, rel_tol=rtol
            )
        if not matches:
            raise TracerValidationError(f"Central {location} changed")
        return
    raise TracerValidationError(f"Unsupported frozen value at {location}")


def _validate_published_case(
    workspace_root: Path,
    summary: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> None:
    _, rows = read_raw_csv(workspace_root / PUBLISHED_CASE_RESULTS_PATH)
    expected = next(
        row
        for row in rows
        if row["series"] == "26a21"
        and row["pair"] == "pair1"
        and row["method"] == "locotrack"
    )
    exact = {
        "visible_opportunities": "physics_baseline_visible_opportunities",
        "branch_consistent_measurements": "branch_consistent_measurements",
        "accepted_measurements_all_frames": "accepted_measurements_all_frames",
        "geometry_inconsistent_accepted_fits": (
            "geometry_inconsistent_accepted_fits"
        ),
        "jointly_consistent_pair_frames": "jointly_consistent_pair_frames",
    }
    for source_field, summary_field in exact.items():
        if int(expected[source_field]) != int(summary[summary_field]):
            raise TracerValidationError(
                f"Published case result changed: {source_field}"
            )
    median = summary["absolute_pair_distance_residual_px"]["median"]
    if not math.isclose(
        float(median),
        float(expected["case_median_abs_distance_residual_px"]),
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise TracerValidationError("Published case residual changed")


def validate_central_result(
    tracer_dir: Path,
    contract: Mapping[str, Any],
    artifacts: TracerArtifacts,
    workspace_root: Path,
    *,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> None:
    """Compare reconstructed records with the authenticated frozen evidence."""

    candidates = artifacts.candidates
    evaluations = artifacts.evaluations
    pairs = artifacts.pairs
    gaps = artifacts.gaps
    summary = artifacts.summary
    tolerance = contract["comparison_tolerances"]["continuous_values"]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])
    counts = contract["expected_counts"]
    if (
        len(candidates) != int(counts["candidate_records"])
        or len(evaluations) != int(counts["evaluation_records"])
    ):
        raise TracerValidationError("Central candidate or evaluation count changed")
    candidate_by_key = {row["record_key"]: row for row in candidates}
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    if (
        len(candidate_by_key) != len(candidates)
        or len(evaluation_by_key) != len(evaluations)
        or set(candidate_by_key) != set(evaluation_by_key)
    ):
        raise TracerValidationError("Central candidate or evaluation keys changed")
    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["candidate_records"],
        candidates,
        CANDIDATE_FIELDS,
        atol,
        rtol,
        record_name="candidate",
        schema_compatibility=schema_compatibility,
    )
    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["evaluation_records"],
        evaluations,
        EVALUATION_FIELDS,
        atol,
        rtol,
        record_name="evaluation",
        schema_compatibility=schema_compatibility,
    )
    _, expected_rows = read_raw_csv(
        tracer_dir / contract["expected_files"]["central_records"]
    )
    if len(expected_rows) != len(candidates):
        raise TracerValidationError("Central candidate record keys changed")
    continuous = (
        "energy_ev",
        "native_x",
        "native_y",
        "native_mask_area",
        "native_mask_score",
        "predicted_x",
        "predicted_y",
        "center_x",
        "center_y",
        "proposal_offset_px",
        "r_squared",
        "amplitude",
        "volume",
        "sigma_x",
        "sigma_y",
        "branch_error_px",
        "radial_residual_px",
        "tangential_residual_px",
    )
    for expected in expected_rows:
        key = record_key(int(expected["frame"]), expected["branch"])
        candidate = candidate_by_key.get(key)
        evaluation = evaluation_by_key.get(key)
        if candidate is None or evaluation is None:
            raise TracerValidationError(f"Central record key changed: {key}")
        discrete_pairs = {
            "method": expected["tracker"],
            "frame": int(expected["frame"]),
            "branch": expected["branch"],
            "physics_baseline_status": expected["physics_baseline_status"],
            "proposal_available": parse_bool(expected["native_visible"]),
        }
        for field, expected_value in discrete_pairs.items():
            if candidate[field] != expected_value:
                raise TracerValidationError(
                    f"Central discrete value changed: {key}/{field}"
                )
        if bool(evaluation["accepted_fit"]) != parse_bool(expected["fit_valid"]):
            raise TracerValidationError(f"Central fit status changed: {key}")
        if evaluation["legacy_fit_reason"] != expected["fit_reason"]:
            raise TracerValidationError(
                f"Central fit classification changed: {key}"
            )
        if bool(evaluation["measurement_valid"]) != parse_bool(
            expected["measurement_valid"]
        ):
            raise TracerValidationError(f"Central measurement status changed: {key}")
        if evaluation["measurement_status"] != expected["measurement_status"]:
            raise TracerValidationError(
                f"Central measurement classification changed: {key}"
            )
        for field in continuous:
            if expected[field] == "":
                continue
            if not _float_matches(candidate.get(field), expected[field], atol, rtol):
                raise TracerValidationError(
                    f"Central continuous value changed: {key}/{field}"
                )

    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["pair_records"],
        pairs,
        PAIR_FIELDS,
        atol,
        rtol,
    )
    _validate_expected_rows(
        tracer_dir / contract["expected_files"]["gap_records"],
        gaps,
        GAP_FIELDS,
        atol,
        rtol,
    )
    expected_summary = read_json(tracer_dir / contract["expected_files"]["summary"])
    _compare_nested(summary, expected_summary, "summary", atol, rtol)
    _validate_published_case(workspace_root, summary, atol, rtol)


def load_output_artifacts(output_dir: Path) -> tuple[TracerArtifacts, dict[str, Any]]:
    """Decode all output records through their typed schemas."""

    try:
        _, candidates = read_csv(
            output_dir / "candidate_records.csv", CandidateRecord
        )
        _, evaluations = read_csv(
            output_dir / "evaluation_records.csv", EvaluationRecord
        )
        _, pairs = read_csv(output_dir / "pair_records.csv", PairRecord)
        _, gaps = read_csv(output_dir / "gap_records.csv", GapRecord)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TracerValidationError(f"Cannot decode tracer output: {exc}") from exc
    summary_document = read_json(output_dir / "summary.json")
    if set(summary_document) != {"central_result", "reader_facing"}:
        raise TracerValidationError("Tracer summary schema changed")
    summary = summary_document["central_result"]
    if not isinstance(summary, dict):
        raise TracerValidationError("Tracer central summary is not an object")
    return (
        TracerArtifacts(
            candidates=candidates,
            evaluations=evaluations,
            pairs=pairs,
            gaps=gaps,
            summary=summary,
        ),
        summary_document,
    )


def tracer_report(artifacts: TracerArtifacts) -> TracerReport:
    summary = artifacts.summary
    return TracerReport(
        candidate_count=len(artifacts.candidates),
        evaluation_count=len(artifacts.evaluations),
        branch_consistent_count=int(summary["branch_consistent_measurements"]),
        evaluated_gap_count=int(summary["post_gap_recovery"]["evaluated_gap_count"]),
        recovered_gap_count=int(summary["post_gap_recovery"]["recovered_gap_count"]),
        matches_frozen_central_record=True,
    )


def validate_complete_output(
    workspace_root: Path,
    tracer_dir: Path,
    output_dir: Path,
    contract: Mapping[str, Any],
    runtime: Mapping[str, str],
    *,
    schema_compatibility: SchemaCompatibility = SchemaCompatibility.EXACT,
) -> TracerReport:
    """Validate all output artifacts and publication policy without rewriting."""

    validate_output_manifest(output_dir)
    artifacts, summary_document = load_output_artifacts(output_dir)
    validate_central_result(
        tracer_dir,
        contract,
        artifacts,
        workspace_root,
        schema_compatibility=schema_compatibility,
    )
    if summary_document["reader_facing"] != reader_facing(
        artifacts.summary, contract["reader_facing_rounding"]
    ):
        raise TracerValidationError("Tracer reader-facing rounding changed")
    provenance = read_json(output_dir / "provenance.json")
    if (
        provenance.get("tracer_id") != TRACER_ID
        or provenance.get("execution_device") != "cpu"
        or provenance.get("runtime") != runtime
        or provenance.get("inputs") != contract["frozen_inputs"]
        or provenance.get("estimator_compatibility")
        != contract["estimator_compatibility"]
        or provenance.get("implementation_sources")
        != implementation_sources(workspace_root)
    ):
        raise TracerValidationError("Tracer provenance changed")
    validation_report = read_json(output_dir / "validation_report.json")
    if (
        validation_report.get("tracer_id") != TRACER_ID
        or validation_report.get("fail_closed") is not True
        or validation_report.get("checks") != VALIDATION_CHECKS
        or validation_report.get("output_manifest_sha256")
        != sha256(output_dir / "output_manifest.json")
    ):
        raise TracerValidationError("Tracer validation report changed")
    report = tracer_report(artifacts)
    if any(
        validation_report.get(key) != value
        for key, value in asdict(report).items()
    ):
        raise TracerValidationError("Tracer validation report counts changed")
    return report


__all__ = [
    "OUTPUT_ARTIFACT_NAMES",
    "VALIDATION_CHECKS",
    "load_output_artifacts",
    "output_manifest",
    "tracer_report",
    "validate_central_result",
    "validate_complete_output",
    "validate_output_manifest",
]
