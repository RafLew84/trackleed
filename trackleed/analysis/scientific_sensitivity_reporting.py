"""Per-slice summaries and flat source records for Ag(111) sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from trackleed.analysis.scientific_panel_common import CASE_SPECS, METHODS
from trackleed.analysis.scientific_panel_outcomes import (
    reconstruct_panel_gaps,
    reconstruct_panel_pairs,
)
from trackleed.analysis.scientific_panel_reporting import summarize_panel
from trackleed.analysis.scientific_panel_records import (
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_sensitivity_records import (
    RejectionCauseCountRecord,
    SensitivityResultRecord,
    SensitivityScope,
)
from trackleed.analysis.scientific_tracer_records import (
    REJECTION_CAUSES,
    CandidateRecord,
    EvaluationRecord,
)


def _rejection_counts(
    evaluations: Sequence[EvaluationRecord],
) -> dict[str, dict[str, int]]:
    first = {cause: 0 for cause in REJECTION_CAUSES}
    full = {cause: 0 for cause in REJECTION_CAUSES}
    for evaluation in evaluations:
        first_cause = evaluation["first_rejection_cause"]
        if first_cause is not None:
            first[first_cause] += 1
        for cause in evaluation["full_rejection_causes"].split("+"):
            if cause:
                full[cause] += 1
    return {"first": first, "full": full}


def _summarize_slice(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    pairs: Sequence[PanelPairRecord],
    gaps: Sequence[PanelGapRecord],
) -> dict[str, Any]:
    summary = summarize_panel(candidates, evaluations, pairs, gaps)
    summary.update(
        {
            "candidate_records_all_frames": len(candidates),
            "proposal_available_all_frames": sum(
                bool(row["proposal_available"]) for row in candidates
            ),
            "solver_completed_fits_all_frames": sum(
                row["solver_status"] == "completed" for row in candidates
            ),
            "finite_candidate_fits_all_frames": sum(
                bool(row["finite_candidate"]) for row in candidates
            ),
            "rejection_causes": _rejection_counts(evaluations),
        }
    )
    return summary


def summarize_variant(
    workspace_root: Path,
    variant: Mapping[str, Any],
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    frozen_gaps: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Summarize one variant per case and across all cases for each method."""

    cases: dict[str, dict[str, dict[str, Any]]] = {}
    pairs: list[PanelPairRecord] = []
    gaps: list[PanelGapRecord] = []
    for case_spec in CASE_SPECS:
        case_methods: dict[str, dict[str, Any]] = {}
        cases[case_spec.case_id] = case_methods
        for method in METHODS:
            selected_candidates = [
                row
                for row in candidates
                if row["case_id"] == case_spec.case_id and row["method"] == method
            ]
            keys = {row["record_key"] for row in selected_candidates}
            selected_evaluations = [
                row for row in evaluations if row["record_key"] in keys
            ]
            selected_pairs = reconstruct_panel_pairs(
                selected_candidates,
                selected_evaluations,
                workspace_root,
                case_spec.case_id,
                case_spec.series_id,
                case_spec.pair_id,
                method,
            )
            selected_gaps = reconstruct_panel_gaps(
                selected_evaluations,
                workspace_root,
                case_spec.case_id,
                case_spec.series_id,
                case_spec.pair_id,
                method,
                [row for row in frozen_gaps if row["case_id"] == case_spec.case_id],
            )
            case_methods[method] = _summarize_slice(
                selected_candidates,
                selected_evaluations,
                selected_pairs,
                selected_gaps,
            )
            pairs.extend(selected_pairs)
            gaps.extend(selected_gaps)

    methods: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_candidates = [row for row in candidates if row["method"] == method]
        method_keys = {row["record_key"] for row in method_candidates}
        methods[method] = _summarize_slice(
            method_candidates,
            [row for row in evaluations if row["record_key"] in method_keys],
            [row for row in pairs if row["method"] == method],
            [row for row in gaps if row["method"] == method],
        )
    return {
        "contract": dict(variant),
        "reference_sets": {
            "opportunities": {
                "set_id": variant["opportunity_set_id"],
                "n": 221,
            },
            "gaps": {"set_id": variant["gap_set_id"], "n": 24},
        },
        "cases": cases,
        "methods": methods,
    }


def _result_record(
    variant_id: str,
    contract: Mapping[str, Any],
    scope: SensitivityScope,
    case_id: str | None,
    method: str,
    summary: Mapping[str, Any],
) -> SensitivityResultRecord:
    case_by_id = {case.case_id: case for case in CASE_SPECS}
    case = case_by_id.get(case_id) if case_id is not None else None
    gaps = summary["post_gap_branch_consistent_recovery"]
    residuals = summary["absolute_pair_distance_residual_px"]
    visible = int(summary["physics_baseline_visible_opportunities"])
    branch_consistent = int(summary["branch_consistent_measurements"])
    accepted = int(summary["accepted_measurements_all_frames"])
    inconsistent = int(summary["geometry_inconsistent_accepted_fits"])
    return SensitivityResultRecord(
        variant_id=variant_id,
        family=str(contract["family"]),
        scope=scope,
        case_id=case_id,
        series_id=case.series_id if case is not None else None,
        pair_id=case.pair_id if case is not None else None,
        method=method,
        roi_side_px=int(contract["roi_side_px"]),
        proposal_to_fit_max_px=float(contract["proposal_to_fit_max_px"]),
        branch_tolerance_px=float(contract["branch_tolerance_px"]),
        r_squared_min=float(contract["r_squared_min"]),
        amplitude_threshold_min=float(contract["amplitude_threshold_min"]),
        sigma_min_px=float(contract["width_gate"]["sigma_min_px"]),
        sigma_max_px=float(contract["width_gate"]["sigma_max_px"]),
        candidate_records_all_frames=int(summary["candidate_records_all_frames"]),
        proposal_available_all_frames=int(summary["proposal_available_all_frames"]),
        solver_completed_fits_all_frames=int(
            summary["solver_completed_fits_all_frames"]
        ),
        finite_candidate_fits_all_frames=int(
            summary["finite_candidate_fits_all_frames"]
        ),
        physics_baseline_visible_opportunities=visible,
        branch_consistent_measurements=branch_consistent,
        branch_consistent_coverage=branch_consistent / visible,
        evaluated_gap_count=int(gaps["evaluated_gap_count"]),
        branch_consistent_recovered_gap_count=int(
            gaps["branch_consistent_recovered_gap_count"]
        ),
        branch_consistent_recovery_rate=float(gaps["branch_consistent_recovery_rate"]),
        accepted_measurements_all_frames=accepted,
        geometry_inconsistent_accepted_fits=inconsistent,
        geometry_inconsistency_denominator=int(
            summary["geometry_inconsistency_denominator"]
        ),
        geometry_inconsistent_fraction=inconsistent / accepted,
        jointly_consistent_pair_frames=int(summary["jointly_consistent_pair_frames"]),
        pair_residual_denominator=int(summary["pair_residual_denominator"]),
        median_absolute_pair_distance_residual_px=float(residuals["median"]),
        p95_absolute_pair_distance_residual_px=float(residuals["p95"]),
        maximum_absolute_pair_distance_residual_px=float(residuals["max"]),
    )


def sensitivity_result_records(
    variant_summaries: Mapping[str, Mapping[str, Any]],
    width_summaries: Mapping[str, Mapping[str, Any]],
) -> list[SensitivityResultRecord]:
    """Flatten per-case and aggregate variant summaries for source-data use."""

    rows: list[SensitivityResultRecord] = []
    for variant_id, variant in {**variant_summaries, **width_summaries}.items():
        contract = variant["contract"]
        for case_spec in CASE_SPECS:
            for method in METHODS:
                rows.append(
                    _result_record(
                        variant_id,
                        contract,
                        "case",
                        case_spec.case_id,
                        method,
                        variant["cases"][case_spec.case_id][method],
                    )
                )
        for method in METHODS:
            rows.append(
                _result_record(
                    variant_id,
                    contract,
                    "method",
                    None,
                    method,
                    variant["methods"][method],
                )
            )
    return rows


def rejection_cause_count_records(
    variant_summaries: Mapping[str, Mapping[str, Any]],
    width_summaries: Mapping[str, Mapping[str, Any]],
) -> list[RejectionCauseCountRecord]:
    """Flatten first and full rejection-cause counts for every result slice."""

    summaries = {**variant_summaries, **width_summaries}
    rows: list[RejectionCauseCountRecord] = []
    for result in sensitivity_result_records(variant_summaries, width_summaries):
        variant = summaries[result["variant_id"]]
        if result["scope"] == "case":
            summary = variant["cases"][result["case_id"]][result["method"]]
        else:
            summary = variant["methods"][result["method"]]
        counts = summary["rejection_causes"]
        for cause in REJECTION_CAUSES:
            rows.append(
                RejectionCauseCountRecord(
                    variant_id=result["variant_id"],
                    family=result["family"],
                    scope=result["scope"],
                    case_id=result["case_id"],
                    method=result["method"],
                    cause=cause,
                    first_rejection_count=int(counts["first"][cause]),
                    full_rejection_count=int(counts["full"][cause]),
                )
            )
    return rows


__all__ = [
    "rejection_cause_count_records",
    "sensitivity_result_records",
    "summarize_variant",
]
