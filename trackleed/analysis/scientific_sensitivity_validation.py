"""Fail-closed validation for Ag(111) sensitivity artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path

from trackleed.analysis.scientific_panel_common import (
    MEASUREMENT_CONTRACT_PATH,
    SCHEMA_VERSION,
    TracerValidationError,
    read_json,
    sha256,
)
from trackleed.analysis.scientific_panel_contract import prepare_panel
from trackleed.analysis.scientific_sensitivity import SensitivityArtifacts
from trackleed.analysis.scientific_sensitivity_common import (
    CENTRAL_VARIANT_ID,
    CENTRAL_WIDTH_CONTRACT_ID,
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
from trackleed.analysis.scientific_sensitivity_reconstruction import (
    reconstruct_sensitivity_results,
    validate_sensitivity_coverage_matrix,
)
from trackleed.analysis.scientific_sensitivity_reporting import (
    rejection_cause_count_records,
    sensitivity_result_records,
)
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_csv,
)


def _records_match(
    actual: list[CandidateRecord], expected: list[CandidateRecord]
) -> bool:
    """Compare typed CSV rows while treating paired NaN values as equal."""

    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if actual_row.keys() != expected_row.keys():
            return False
        for field, actual_value in actual_row.items():
            expected_value = expected_row[field]
            if (
                isinstance(actual_value, float)
                and isinstance(expected_value, float)
                and math.isnan(actual_value)
                and math.isnan(expected_value)
            ):
                continue
            if actual_value != expected_value:
                return False
    return True


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
        if (
            len(artifacts.width_summaries) != 4
            or len(artifacts.width_evaluations) != 6480
            or Counter(row["variant_id"] for row in artifacts.width_evaluations)
            != {variant_id: 1620 for variant_id in artifacts.width_summaries}
            or CENTRAL_WIDTH_CONTRACT_ID in artifacts.width_summaries
        ):
            raise TracerValidationError("Triggered width sweep is incomplete")
    elif artifacts.width_summaries or artifacts.width_evaluations:
        raise TracerValidationError("Width sweep ran without a trigger")


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
    additional_width_contract_count = (
        width_contract_count - 1 if width_contract_count == 5 else 0
    )
    expected_evaluation_count = 24300 + 1620 * additional_width_contract_count
    expected_result_count = 375 + 25 * additional_width_contract_count
    if (
        len(candidates) != 4860
        or len(evaluations) != expected_evaluation_count
        or len(results) != expected_result_count
        or len(rejection_counts) != expected_result_count * 9
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
    validate_sensitivity_coverage_matrix(
        prepared,
        candidates,
        evaluations,
        results,
        rejection_counts,
        audit,
        variants,
        width,
    )
    measurement_contract = read_json(
        prepared.workspace_root / MEASUREMENT_CONTRACT_PATH
    )
    reconstructed = reconstruct_sensitivity_results(
        prepared,
        candidates,
        measurement_contract,
    )
    expected_width_contracts: dict[str, object] = {}
    if reconstructed.triggered_methods:
        expected_width_contracts = {
            CENTRAL_WIDTH_CONTRACT_ID: {
                "reuses_variant_id": CENTRAL_VARIANT_ID,
            },
            **reconstructed.width_summaries,
        }
    if (
        evaluations != reconstructed.evaluations
        or variants != reconstructed.variants
        or width_contracts != expected_width_contracts
        or audit != reconstructed.audit
        or width.get("triggered_methods") != list(reconstructed.triggered_methods)
        or results
        != sensitivity_result_records(
            reconstructed.variants,
            reconstructed.width_summaries,
        )
        or rejection_counts
        != rejection_cause_count_records(
            reconstructed.variants,
            reconstructed.width_summaries,
        )
    ):
        raise TracerValidationError(
            "Sensitivity results do not reconstruct from candidate records"
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
    expected_candidate_path = (
        prepared.panel_dir / prepared.contract["expected_files"]["candidate_records"]
    )
    _, frozen_candidates = read_csv(expected_candidate_path, CandidateRecord)
    _, frozen_central = read_csv(expected_evaluation_path, EvaluationRecord)
    actual_central_candidates = [row for row in candidates if row["roi_side_px"] == 48]
    actual_central = [
        row for row in evaluations if row["variant_id"] == CENTRAL_VARIANT_ID
    ]
    if (
        not _records_match(actual_central_candidates, frozen_candidates)
        or actual_central != frozen_central
    ):
        raise TracerValidationError(
            "Central sensitivity output differs from frozen panel"
        )
    report = SensitivityReport(
        variant_count=len(variants),
        unique_roi_candidate_set_count=len({row["roi_side_px"] for row in candidates}),
        candidate_count=len(candidates),
        evaluation_count=expected_evaluation_count,
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
