"""Fail-closed validation for the complete Ag(111) central panel."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from trackleed.analysis.scientific_panel_common import (
    GAPS_PATH,
    METHODS,
    PANEL_ID,
    PUBLISHED_AGGREGATE_RESULTS_PATH,
    PUBLISHED_CASE_RESULTS_PATH,
    SCHEMA_VERSION,
    TracerValidationError,
    PanelReport,
    implementation_sources,
    panel_hash_manifest,
    read_json,
    sha256,
)
from trackleed.analysis.scientific_panel_reconstruction import (
    PanelArtifacts,
    panel_record_key,
    reader_facing_panel,
)
from trackleed.analysis.scientific_panel_records import (
    AGGREGATE_REPORT_FIELDS,
    CASE_REPORT_FIELDS,
    HISTORICAL_CENTRAL_FIELDS,
    PANEL_GAP_FIELDS,
    PANEL_PAIR_FIELDS,
    AggregateReportRecord,
    CaseReportRecord,
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_tracer_records import (
    CANDIDATE_FIELDS,
    EVALUATION_FIELDS,
    CandidateRecord,
    EvaluationRecord,
    parse_bool,
    read_csv,
    read_raw_csv,
)


VALIDATION_CHECKS = {
    "aggregate_and_case_publication_anchors": "passed",
    "complete_candidate_records": "passed",
    "complete_evaluation_records": "passed",
    "continuous_value_tolerances": "passed",
    "fixed_221_opportunity_keys_per_method": "passed",
    "fixed_24_gap_keys_per_method": "passed",
    "frozen_input_hashes": "passed",
    "historical_central_records": "passed",
    "implementation_source_hashes": "passed",
    "method_specific_conditional_denominators": "passed",
    "reader_facing_rounding": "passed",
}
OUTPUT_ARTIFACT_NAMES = (
    "candidate_records.csv",
    "evaluation_records.csv",
    "pair_records.csv",
    "gap_records.csv",
    "case_results.csv",
    "aggregate_method_results.csv",
    "summary.json",
    "provenance.json",
)


def panel_output_manifest(output_dir: Path) -> dict[str, object]:
    return panel_hash_manifest(
        {name: sha256(output_dir / name) for name in OUTPUT_ARTIFACT_NAMES}
    )


def validate_panel_output_manifest(output_dir: Path) -> None:
    manifest = read_json(output_dir / "output_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("tracer_id") != PANEL_ID
    ):
        raise TracerValidationError("Panel output manifest identity changed")
    expected_hashes = manifest.get("files_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        OUTPUT_ARTIFACT_NAMES
    ):
        raise TracerValidationError("Panel output manifest inventory changed")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_ARTIFACT_NAMES) | {
        "output_manifest.json",
        "validation_report.json",
    }:
        raise TracerValidationError("Panel output file inventory changed")
    for name, expected_hash in expected_hashes.items():
        if sha256(output_dir / name) != expected_hash:
            raise TracerValidationError(f"Panel output hash mismatch: {name}")


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


def _validate_rows(
    expected_path: Path,
    actual_rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    atol: float,
    rtol: float,
    label: str,
) -> None:
    expected_fields, expected_rows = read_raw_csv(expected_path)
    if expected_fields != list(fields) or len(expected_rows) != len(actual_rows):
        raise TracerValidationError(f"Frozen panel {label} record set changed")
    for row_index, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
        for field in fields:
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
                    f"Frozen panel {label} changed: row {row_index}/{field}"
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
            raise TracerValidationError(f"Panel {location} keys changed")
        for key, expected_value in expected.items():
            _compare_nested(
                actual[key], expected_value, f"{location}.{key}", atol, rtol
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise TracerValidationError(f"Panel {location} changed")
        return
    if isinstance(expected, int):
        if actual != expected:
            raise TracerValidationError(f"Panel {location} changed")
        return
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, abs_tol=atol, rel_tol=rtol):
            raise TracerValidationError(f"Panel {location} changed")
        return
    raise TracerValidationError(f"Unsupported frozen panel value at {location}")


def _validate_historical_records(
    panel_dir: Path,
    contract: Mapping[str, Any],
    artifacts: PanelArtifacts,
    atol: float,
    rtol: float,
) -> None:
    path = panel_dir / contract["expected_files"]["historical_central_records"]
    fields, rows = read_raw_csv(path)
    if fields != list(HISTORICAL_CENTRAL_FIELDS) or len(rows) != len(
        artifacts.candidates
    ):
        raise TracerValidationError("Frozen historical central record set changed")
    candidate_by_key = {row["record_key"]: row for row in artifacts.candidates}
    evaluation_by_key = {row["record_key"]: row for row in artifacts.evaluations}
    status_translation = {"tracker_not_visible": "proposal_unavailable"}
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
    for expected in rows:
        key = panel_record_key(
            expected["case_id"],
            expected["tracker"],
            int(expected["frame"]),
            expected["branch"],
            48,
        )
        candidate = candidate_by_key.get(key)
        evaluation = evaluation_by_key.get(key)
        if candidate is None or evaluation is None:
            raise TracerValidationError(f"Historical central key changed: {key}")
        discrete = {
            "case_id": expected["case_id"],
            "series_id": expected["series_id"],
            "pair_id": expected["pair_id"],
            "method": expected["tracker"],
            "frame": int(expected["frame"]),
            "branch": expected["branch"],
            "physics_baseline_status": expected["physics_baseline_status"],
            "proposal_available": parse_bool(expected["native_visible"]),
        }
        for field, expected_value in discrete.items():
            if candidate[field] != expected_value:
                raise TracerValidationError(
                    f"Historical central discrete value changed: {key}/{field}"
                )
        expected_fit_valid = (
            parse_bool(expected["fit_valid"]) if expected["fit_valid"] else False
        )
        if bool(evaluation["accepted_fit"]) != expected_fit_valid:
            raise TracerValidationError(f"Historical fit status changed: {key}")
        if evaluation["legacy_fit_reason"] != expected["fit_reason"]:
            raise TracerValidationError(
                f"Historical fit classification changed: {key}"
            )
        expected_measurement_valid = (
            parse_bool(expected["measurement_valid"])
            if expected["measurement_valid"]
            else False
        )
        if bool(evaluation["measurement_valid"]) != expected_measurement_valid:
            raise TracerValidationError(
                f"Historical measurement validity changed: {key}"
            )
        if expected["fit_reason"] == "roi_outside_image":
            expected_status = "roi_invalid"
        else:
            expected_status = status_translation.get(
                expected["measurement_status"], expected["measurement_status"]
            )
        if evaluation["measurement_status"] != expected_status:
            raise TracerValidationError(
                f"Historical measurement classification changed: {key}"
            )
        for field in continuous:
            if expected[field] and not _float_matches(
                candidate.get(field), expected[field], atol, rtol
            ):
                raise TracerValidationError(
                    f"Historical central continuous value changed: {key}/{field}"
                )


def validate_panel_result(
    workspace_root: Path,
    panel_dir: Path,
    contract: Mapping[str, Any],
    artifacts: PanelArtifacts,
) -> None:
    tolerance = contract["comparison_tolerances"]["continuous_values"]
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])
    counts = contract["expected_counts"]
    named_rows = {
        "candidate_records": (artifacts.candidates, CANDIDATE_FIELDS),
        "evaluation_records": (artifacts.evaluations, EVALUATION_FIELDS),
        "pair_records": (artifacts.pairs, PANEL_PAIR_FIELDS),
        "gap_records": (artifacts.gaps, PANEL_GAP_FIELDS),
        "case_results": (artifacts.case_reports, CASE_REPORT_FIELDS),
        "aggregate_method_results": (
            artifacts.aggregate_reports,
            AGGREGATE_REPORT_FIELDS,
        ),
    }
    for name, (rows, fields) in named_rows.items():
        if len(rows) != int(counts[name]):
            raise TracerValidationError(f"Panel {name} count changed")
        _validate_rows(
            panel_dir / contract["expected_files"][name],
            rows,
            fields,
            atol,
            rtol,
            name,
        )
    candidate_keys = {row["record_key"] for row in artifacts.candidates}
    evaluation_keys = {row["record_key"] for row in artifacts.evaluations}
    if (
        len(candidate_keys) != len(artifacts.candidates)
        or candidate_keys != evaluation_keys
    ):
        raise TracerValidationError("Panel candidate/evaluation keys changed")

    _, opportunity_rows = read_raw_csv(
        workspace_root / "manuscript/scientific_baseline/opportunities.csv"
    )
    opportunity_keys = {row["opportunity_key"] for row in opportunity_rows}
    _, gap_rows = read_raw_csv(workspace_root / GAPS_PATH)
    gap_keys = {row["gap_key"] for row in gap_rows}
    for method in METHODS:
        method_opportunities = [
            row["opportunity_key"]
            for row in artifacts.candidates
            if row["method"] == method and row["opportunity_key"] is not None
        ]
        if (
            len(method_opportunities) != 221
            or set(method_opportunities) != opportunity_keys
        ):
            raise TracerValidationError(
                f"Fixed opportunity set changed for method: {method}"
            )
        method_gaps = [
            row["gap_key"] for row in artifacts.gaps if row["method"] == method
        ]
        if len(method_gaps) != 24 or set(method_gaps) != gap_keys:
            raise TracerValidationError(f"Fixed gap set changed for method: {method}")
        method_summary = artifacts.summary["methods"][method]
        if (
            method_summary["geometry_inconsistency_denominator"]
            != method_summary["accepted_measurements_all_frames"]
            or method_summary["pair_residual_denominator"]
            != method_summary["jointly_consistent_pair_frames"]
        ):
            raise TracerValidationError(
                f"Conditional denominator changed for method: {method}"
            )

    _validate_historical_records(panel_dir, contract, artifacts, atol, rtol)
    _validate_rows(
        workspace_root / PUBLISHED_CASE_RESULTS_PATH,
        artifacts.case_reports,
        CASE_REPORT_FIELDS,
        atol,
        rtol,
        "published case result",
    )
    _validate_rows(
        workspace_root / PUBLISHED_AGGREGATE_RESULTS_PATH,
        artifacts.aggregate_reports,
        AGGREGATE_REPORT_FIELDS,
        atol,
        rtol,
        "published aggregate result",
    )
    expected_summary = read_json(
        panel_dir / contract["expected_files"]["summary"]
    )
    _compare_nested(artifacts.summary, expected_summary, "summary", atol, rtol)


def load_panel_output(
    output_dir: Path,
) -> tuple[PanelArtifacts, dict[str, Any]]:
    try:
        _, candidates = read_csv(
            output_dir / "candidate_records.csv", CandidateRecord
        )
        _, evaluations = read_csv(
            output_dir / "evaluation_records.csv", EvaluationRecord
        )
        _, pairs = read_csv(output_dir / "pair_records.csv", PanelPairRecord)
        _, gaps = read_csv(output_dir / "gap_records.csv", PanelGapRecord)
        _, case_reports = read_csv(
            output_dir / "case_results.csv", CaseReportRecord
        )
        _, aggregate_reports = read_csv(
            output_dir / "aggregate_method_results.csv", AggregateReportRecord
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TracerValidationError(f"Cannot decode panel output: {exc}") from exc
    summary_document = read_json(output_dir / "summary.json")
    if set(summary_document) != {"central_result", "reader_facing"}:
        raise TracerValidationError("Panel summary schema changed")
    summary = summary_document["central_result"]
    if not isinstance(summary, dict):
        raise TracerValidationError("Panel central summary is not an object")
    return (
        PanelArtifacts(
            candidates=candidates,
            evaluations=evaluations,
            pairs=pairs,
            gaps=gaps,
            case_reports=case_reports,
            aggregate_reports=aggregate_reports,
            summary=summary,
        ),
        summary_document,
    )


def panel_report(artifacts: PanelArtifacts) -> PanelReport:
    return PanelReport(
        candidate_count=len(artifacts.candidates),
        evaluation_count=len(artifacts.evaluations),
        case_count=len(artifacts.summary["cases"]),
        method_count=len(artifacts.summary["methods"]),
        opportunity_reference_count=int(
            artifacts.summary["reference_sets"]["opportunities"]["n"]
        ),
        gap_reference_count=int(
            artifacts.summary["reference_sets"]["gaps"]["n"]
        ),
        matches_frozen_central_record=True,
    )


def validate_complete_panel_output(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
    contract: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> PanelReport:
    validate_panel_output_manifest(output_dir)
    artifacts, summary_document = load_panel_output(output_dir)
    validate_panel_result(workspace_root, panel_dir, contract, artifacts)
    if summary_document["reader_facing"] != reader_facing_panel(
        artifacts.summary, contract["reader_facing_rounding"]
    ):
        raise TracerValidationError("Panel reader-facing rounding changed")
    provenance = read_json(output_dir / "provenance.json")
    if (
        provenance.get("tracer_id") != PANEL_ID
        or provenance.get("execution_device") != "cpu"
        or provenance.get("runtime") != runtime
        or provenance.get("inputs") != contract["frozen_inputs"]
        or provenance.get("estimator_compatibility")
        != contract["estimator_compatibility"]
        or provenance.get("implementation_sources")
        != implementation_sources(workspace_root)
    ):
        raise TracerValidationError("Panel provenance changed")
    validation_report = read_json(output_dir / "validation_report.json")
    if (
        validation_report.get("tracer_id") != PANEL_ID
        or validation_report.get("fail_closed") is not True
        or validation_report.get("checks") != VALIDATION_CHECKS
        or validation_report.get("output_manifest_sha256")
        != sha256(output_dir / "output_manifest.json")
    ):
        raise TracerValidationError("Panel validation report changed")
    report = panel_report(artifacts)
    if any(
        validation_report.get(key) != value
        for key, value in asdict(report).items()
    ):
        raise TracerValidationError("Panel validation report counts changed")
    return report


__all__ = [
    "OUTPUT_ARTIFACT_NAMES",
    "VALIDATION_CHECKS",
    "load_panel_output",
    "panel_output_manifest",
    "panel_report",
    "validate_complete_panel_output",
    "validate_panel_output_manifest",
    "validate_panel_result",
]
