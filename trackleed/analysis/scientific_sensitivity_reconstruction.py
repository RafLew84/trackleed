"""Reconstruction checks for persisted Ag(111) sensitivity results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from trackleed.analysis.scientific_panel_common import (
    CASE_SPECS,
    GAPS_PATH,
    MEASUREMENT_CONTRACT_PATH,
    METHODS,
    TracerValidationError,
    read_json,
)
from trackleed.analysis.scientific_panel_contract import PreparedPanel
from trackleed.analysis.scientific_sensitivity import (
    audit_width_trigger,
    complete_sensitivity_variants,
    complete_width_variant,
    reconstruct_candidate_sets,
)
from trackleed.analysis.scientific_sensitivity_common import (
    CENTRAL_VARIANT_ID,
    CENTRAL_WIDTH_CONTRACT_ID,
)
from trackleed.analysis.scientific_sensitivity_records import (
    RejectionCauseCountRecord,
    SensitivityResultRecord,
    WidthTriggerRecord,
)
from trackleed.analysis.scientific_sensitivity_reporting import summarize_variant
from trackleed.analysis.scientific_tracer_evaluation import evaluate_candidates
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_csv,
    read_raw_csv,
)


@dataclass(frozen=True)
class ReconstructedSensitivity:
    variants: dict[str, dict[str, Any]]
    width_summaries: dict[str, dict[str, Any]]
    evaluations: list[EvaluationRecord]
    audit: list[WidthTriggerRecord]
    triggered_methods: tuple[str, ...]


def reconstruct_authenticated_candidates(
    prepared: PreparedPanel,
    measurement_contract: dict[str, Any],
) -> list[CandidateRecord]:
    """Rebuild the ROI matrix, reusing central rows and refitting ROI 32/64."""

    _, variants = complete_sensitivity_variants(measurement_contract)
    candidate_path = (
        prepared.panel_dir / prepared.contract["expected_files"]["candidate_records"]
    )
    _, central_candidates = read_csv(candidate_path, CandidateRecord)
    candidate_sets = reconstruct_candidate_sets(
        prepared.workspace_root,
        prepared.contract,
        central_candidates,
        variants,
    )
    return [row for roi_side in (32, 48, 64) for row in candidate_sets[roi_side]]


def validate_sensitivity_coverage_matrix(
    prepared: PreparedPanel,
    candidates: list[CandidateRecord],
    evaluations: list[EvaluationRecord],
    results: list[SensitivityResultRecord],
    rejection_counts: list[RejectionCauseCountRecord],
    audit: list[WidthTriggerRecord],
    variants: dict[str, Any],
    width: dict[str, Any],
) -> None:
    """Check the declared variant, slice and rejection-cause matrix."""

    measurement_contract = read_json(
        prepared.workspace_root / MEASUREMENT_CONTRACT_PATH
    )
    expected_variant_ids = {
        str(variant["variant_id"])
        for variant in measurement_contract["sensitivity_variants"]
    }
    width_contracts = width.get("width_contracts")
    if not isinstance(width_contracts, dict):
        raise TracerValidationError("Sensitivity width results changed")
    expected_width_ids = {
        str(contract["contract_id"])
        for contract in measurement_contract["conditional_width_sensitivity"][
            "contracts"
        ]
    }
    width_triggered = width.get("decision") == "run_all_width_contracts"
    if width_triggered:
        if set(width_contracts) != expected_width_ids or width_contracts.get(
            CENTRAL_WIDTH_CONTRACT_ID
        ) != {"reuses_variant_id": CENTRAL_VARIANT_ID}:
            raise TracerValidationError("Sensitivity width contract matrix changed")
        additional_width_ids = expected_width_ids - {CENTRAL_WIDTH_CONTRACT_ID}
    else:
        if width_contracts:
            raise TracerValidationError("Untriggered width contract matrix changed")
        additional_width_ids = set()

    expected_result_ids = expected_variant_ids | additional_width_ids
    if (
        set(variants) != expected_variant_ids
        or Counter(row["roi_side_px"] for row in candidates)
        != {32: 1620, 48: 1620, 64: 1620}
        or Counter(row["variant_id"] for row in evaluations)
        != {variant_id: 1620 for variant_id in expected_result_ids}
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

    additional_width_summaries = {
        variant_id: width_contracts[variant_id] for variant_id in additional_width_ids
    }
    for variant in {**variants, **additional_width_summaries}.values():
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


def reconstruct_sensitivity_results(
    prepared: PreparedPanel,
    candidates: list[CandidateRecord],
    measurement_contract: dict[str, Any],
) -> ReconstructedSensitivity:
    """Re-derive every classification, summary and trigger from candidate rows."""

    central, variants = complete_sensitivity_variants(measurement_contract)
    _, frozen_gaps = read_raw_csv(prepared.workspace_root / GAPS_PATH)
    candidate_sets = {
        roi_side: [row for row in candidates if row["roi_side_px"] == roi_side]
        for roi_side in (32, 48, 64)
    }
    expected_evaluations: list[EvaluationRecord] = []
    expected_summaries: dict[str, dict[str, Any]] = {}
    central_evaluations: list[EvaluationRecord] = []
    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_candidates = candidate_sets[int(variant["roi_side_px"])]
        variant_evaluations = evaluate_candidates(
            variant_candidates,
            variant,
            variant_id=variant_id,
        )
        expected_evaluations.extend(variant_evaluations)
        expected_summaries[variant_id] = summarize_variant(
            prepared.workspace_root,
            variant,
            variant_candidates,
            variant_evaluations,
            frozen_gaps,
        )
        if variant_id == CENTRAL_VARIANT_ID:
            central_evaluations = variant_evaluations

    central_candidates = candidate_sets[48]
    width_audit = audit_width_trigger(
        central_candidates,
        central_evaluations,
        measurement_contract,
    )
    expected_width_summaries: dict[str, dict[str, Any]] = {}
    if width_audit.run_width_sensitivity:
        for width_value in measurement_contract["conditional_width_sensitivity"][
            "contracts"
        ]:
            variant = complete_width_variant(
                central,
                measurement_contract,
                width_value,
            )
            variant_id = str(variant["variant_id"])
            if variant_id == CENTRAL_WIDTH_CONTRACT_ID:
                continue
            width_evaluations = evaluate_candidates(
                central_candidates,
                variant,
                variant_id=variant_id,
            )
            expected_evaluations.extend(width_evaluations)
            expected_width_summaries[variant_id] = summarize_variant(
                prepared.workspace_root,
                variant,
                central_candidates,
                width_evaluations,
                frozen_gaps,
            )
    return ReconstructedSensitivity(
        variants=expected_summaries,
        width_summaries=expected_width_summaries,
        evaluations=expected_evaluations,
        audit=[*width_audit.method_records, *width_audit.case_records],
        triggered_methods=width_audit.triggered_methods,
    )


__all__ = [
    "ReconstructedSensitivity",
    "reconstruct_authenticated_candidates",
    "reconstruct_sensitivity_results",
    "validate_sensitivity_coverage_matrix",
]
