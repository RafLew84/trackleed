"""Fail-closed validation for Ag(111) sensitivity artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from trackleed.analysis.scientific_panel_common import (
    CASE_SPECS,
    MEASUREMENT_CONTRACT_PATH,
    METHODS,
    SCHEMA_VERSION,
    TracerValidationError,
    read_json,
    sha256,
)
from trackleed.analysis.scientific_panel_contract import PreparedPanel, prepare_panel
from trackleed.analysis.scientific_sensitivity import SensitivityArtifacts
from trackleed.analysis.scientific_sensitivity_common import (
    DATA_ARTIFACT_NAMES,
    OUTPUT_FILE_NAMES,
    SENSITIVITY_ID,
    SensitivityReport,
    implementation_sources,
)
from trackleed.analysis.scientific_sensitivity_records import (
    RejectionCauseCountRecord,
    SensitivityResultRecord,
    WidthTriggerRecord,
)
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_csv,
)


def validate_sensitivity_artifacts(artifacts: SensitivityArtifacts) -> None:
    """Validate the complete in-memory matrix before writing any result."""

    if (
        len(artifacts.variant_summaries) != 15
        or len(artifacts.candidates) != 4860
        or len(artifacts.evaluations) != 24300
    ):
        raise TracerValidationError("Sensitivity record count changed")
    if Counter(row["roi_side_px"] for row in artifacts.candidates) != {
        32: 1620,
        48: 1620,
        64: 1620,
    }:
        raise TracerValidationError("Sensitivity ROI candidate sets changed")
    candidate_keys = {row["record_key"] for row in artifacts.candidates}
    if len(candidate_keys) != len(artifacts.candidates):
        raise TracerValidationError("Sensitivity candidate keys are not unique")
    evaluation_keys = {
        (row["variant_id"], row["record_key"]) for row in artifacts.evaluations
    }
    if len(evaluation_keys) != len(artifacts.evaluations):
        raise TracerValidationError("Sensitivity evaluation keys are not unique")
    if Counter(row["variant_id"] for row in artifacts.evaluations) != {
        variant_id: 1620 for variant_id in artifacts.variant_summaries
    }:
        raise TracerValidationError("Sensitivity evaluation matrix changed")
    for variant in artifacts.variant_summaries.values():
        for summary in variant["methods"].values():
            if (
                summary["physics_baseline_visible_opportunities"] != 221
                or summary["post_gap_branch_consistent_recovery"]["evaluated_gap_count"]
                != 24
                or summary["geometry_inconsistency_denominator"]
                != summary["accepted_measurements_all_frames"]
                or summary["pair_residual_denominator"]
                != summary["jointly_consistent_pair_frames"]
            ):
                raise TracerValidationError("Sensitivity denominator contract changed")
    if not artifacts.central_variant_matches_frozen_record:
        raise TracerValidationError(
            "Central sensitivity result differs from frozen panel"
        )
    if artifacts.width_trigger.run_width_sensitivity:
        if len(artifacts.width_summaries) != 5:
            raise TracerValidationError("Triggered width sweep is incomplete")
    elif artifacts.width_summaries or artifacts.width_evaluations:
        raise TracerValidationError("Width sweep ran without a trigger")


def _validate_coverage_matrix(
    prepared: PreparedPanel,
    candidates: list[CandidateRecord],
    evaluations: list[EvaluationRecord],
    results: list[SensitivityResultRecord],
    rejection_counts: list[RejectionCauseCountRecord],
    audit: list[WidthTriggerRecord],
    variants: dict[str, object],
    width: dict[str, object],
) -> None:
    workspace_root = prepared.workspace_root
    measurement_contract = read_json(workspace_root / MEASUREMENT_CONTRACT_PATH)
    expected_variant_ids = {
        str(variant["variant_id"])
        for variant in measurement_contract["sensitivity_variants"]
    }
    width_contracts = width["width_contracts"]
    if not isinstance(width_contracts, dict):
        raise TracerValidationError("Sensitivity width results changed")
    expected_result_ids = expected_variant_ids | set(width_contracts)
    if (
        set(variants) != expected_variant_ids
        or Counter(row["roi_side_px"] for row in candidates)
        != {32: 1620, 48: 1620, 64: 1620}
        or Counter(row["variant_id"] for row in evaluations)
        != {variant_id: 1620 for variant_id in expected_variant_ids}
        or Counter(row["variant_id"] for row in results)
        != {variant_id: 25 for variant_id in expected_result_ids}
        or Counter(row["scope"] for row in results)
        != {
            "case": 20 * len(expected_result_ids),
            "method": 5 * len(expected_result_ids),
        }
        or Counter(row["scope"] for row in audit) != {"case": 20, "method": 5}
    ):
        raise TracerValidationError("Sensitivity coverage matrix changed")
    expected_rejection_keys = {
        (row["variant_id"], row["scope"], row["case_id"], row["method"])
        for row in results
    }
    actual_rejection_counts = Counter(
        (row["variant_id"], row["scope"], row["case_id"], row["method"])
        for row in rejection_counts
    )
    if actual_rejection_counts != {key: 9 for key in expected_rejection_keys}:
        raise TracerValidationError("Sensitivity rejection-cause matrix changed")
    for variant in variants.values():
        if not isinstance(variant, dict) or (
            set(variant.get("cases", {})) != {case.case_id for case in CASE_SPECS}
            or set(variant.get("methods", {})) != set(METHODS)
        ):
            raise TracerValidationError("Sensitivity summary matrix changed")
        for method_summary in variant["methods"].values():
            if (
                method_summary["physics_baseline_visible_opportunities"] != 221
                or method_summary["post_gap_branch_consistent_recovery"][
                    "evaluated_gap_count"
                ]
                != 24
                or method_summary["geometry_inconsistency_denominator"]
                != method_summary["accepted_measurements_all_frames"]
                or method_summary["pair_residual_denominator"]
                != method_summary["jointly_consistent_pair_frames"]
            ):
                raise TracerValidationError("Sensitivity summary denominator changed")


def validate_sensitivity_output(
    workspace_root: Path,
    panel_dir: Path,
    output_dir: Path,
) -> SensitivityReport:
    """Validate an existing sensitivity assessment without rewriting it."""

    prepared = prepare_panel(workspace_root, panel_dir, output_dir)
    if not output_dir.is_dir():
        raise TracerValidationError("Sensitivity output directory is missing")
    if {
        path.name for path in output_dir.iterdir() if path.is_file()
    } != OUTPUT_FILE_NAMES:
        raise TracerValidationError("Sensitivity output inventory changed")
    manifest = read_json(output_dir / "output_manifest.json")
    hashes = manifest.get("files_sha256")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("sensitivity_id") != SENSITIVITY_ID
        or not isinstance(hashes, dict)
        or set(hashes) != set(DATA_ARTIFACT_NAMES)
    ):
        raise TracerValidationError("Sensitivity output manifest changed")
    for name, expected_hash in hashes.items():
        if sha256(output_dir / name) != expected_hash:
            raise TracerValidationError(f"Sensitivity output changed: {name}")

    _, candidates = read_csv(output_dir / "candidate_records.csv", CandidateRecord)
    _, evaluations = read_csv(output_dir / "evaluation_records.csv", EvaluationRecord)
    _, results = read_csv(
        output_dir / "sensitivity_results.csv", SensitivityResultRecord
    )
    _, rejection_counts = read_csv(
        output_dir / "rejection_cause_counts.csv", RejectionCauseCountRecord
    )
    _, audit = read_csv(output_dir / "gate_margin_audit.csv", WidthTriggerRecord)
    summary = read_json(output_dir / "summary.json")
    provenance = read_json(output_dir / "provenance.json")
    variants = summary.get("variants")
    width = summary.get("conditional_width_sensitivity")
    if not isinstance(variants, dict) or not isinstance(width, dict):
        raise TracerValidationError("Sensitivity summary contract changed")
    width_contracts = width.get("width_contracts")
    width_contract_count = (
        len(width_contracts) if isinstance(width_contracts, dict) else -1
    )
    decision = width.get("decision")
    if (
        len(candidates) != 4860
        or len(evaluations) != 24300
        or len(results) != 375 + 25 * width_contract_count
        or len(rejection_counts) != (375 + 25 * width_contract_count) * 9
        or len(audit) != 25
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("sensitivity_id") != SENSITIVITY_ID
        or len(variants) != 15
        or width_contract_count not in {0, 5}
        or decision
        != (
            "run_all_width_contracts"
            if width_contract_count == 5
            else "not_run_no_trigger"
        )
        or width.get("central_contract_changed") is not False
        or width.get("optimizer_hard_bounds_changed") is not False
        or width.get("automatic_configuration_selection") is not False
        or width.get("method_audit") != audit[:5]
        or width.get("case_diagnostics") != audit[5:]
    ):
        raise TracerValidationError("Sensitivity output contract changed")
    _validate_coverage_matrix(
        prepared,
        candidates,
        evaluations,
        results,
        rejection_counts,
        audit,
        variants,
        width,
    )
    if (
        provenance.get("schema_version") != SCHEMA_VERSION
        or provenance.get("sensitivity_id") != SENSITIVITY_ID
        or provenance.get("execution_device") != "cpu"
        or provenance.get("runtime") != prepared.runtime
        or provenance.get("tracker_inference_run") is not False
        or provenance.get("central_contract_changed") is not False
        or provenance.get("automatic_configuration_selection") is not False
        or provenance.get("baseline_package_manifest_sha256")
        != sha256(
            prepared.workspace_root
            / "manuscript"
            / "scientific_baseline"
            / "package_manifest.json"
        )
        or provenance.get("central_panel_package_manifest_sha256")
        != sha256(prepared.panel_dir / "package_manifest.json")
        or provenance.get("implementation_sources")
        != implementation_sources(prepared.workspace_root)
    ):
        raise TracerValidationError("Sensitivity provenance changed")

    expected_evaluation_path = (
        prepared.panel_dir / prepared.contract["expected_files"]["evaluation_records"]
    )
    _, frozen_central = read_csv(expected_evaluation_path, EvaluationRecord)
    actual_central = [
        row for row in evaluations if row["variant_id"] == "roi_48_proposal_15"
    ]
    if actual_central != frozen_central:
        raise TracerValidationError(
            "Central sensitivity output differs from frozen panel"
        )
    report = SensitivityReport(
        variant_count=len(variants),
        unique_roi_candidate_set_count=len({row["roi_side_px"] for row in candidates}),
        candidate_count=len(candidates),
        evaluation_count=len(evaluations),
        width_sensitivity_triggered=decision == "run_all_width_contracts",
        width_contract_count=width_contract_count,
        central_variant_matches_frozen_record=True,
    )
    validation = read_json(output_dir / "validation_report.json")
    if (
        validation.get("report") != asdict(report)
        or validation.get("output_manifest_sha256")
        != sha256(output_dir / "output_manifest.json")
        or validation.get("fail_closed") is not True
    ):
        raise TracerValidationError("Sensitivity validation report changed")
    return report


__all__ = ["validate_sensitivity_artifacts", "validate_sensitivity_output"]
