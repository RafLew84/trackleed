"""Case, aggregate and reader-facing reporting for the Ag(111) panel."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from trackleed.analysis.scientific_panel_common import DISPLAY_METHODS
from trackleed.analysis.scientific_panel_records import (
    AggregateReportRecord,
    CaseReportRecord,
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_tracer_records import CandidateRecord, EvaluationRecord


AGGREGATE_METHOD_ORDER = ("locotrack", "ostrack", "dam4sam", "lk", "easyleed")


def summarize_panel(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    pairs: Sequence[PanelPairRecord],
    gaps: Sequence[PanelGapRecord],
) -> dict[str, Any]:
    by_key = {row["record_key"]: row for row in candidates}
    baseline_visible = [
        row for row in evaluations if row["physics_baseline_status"] == "visible_fit"
    ]
    consistent = [
        row
        for row in baseline_visible
        if row["measurement_status"] == "branch_consistent"
    ]
    accepted = [row for row in evaluations if bool(row["measurement_valid"])]
    inconsistent = [
        row
        for row in accepted
        if row["measurement_status"] == "geometry_inconsistent"
    ]
    residuals = [abs(float(row["distance_residual_px"])) for row in pairs]
    recovered = [row for row in gaps if bool(row["recovered"])]
    return {
        "physics_baseline_visible_opportunities": len(baseline_visible),
        "native_visible_on_baseline_visible": sum(
            bool(by_key[row["record_key"]]["proposal_available"])
            for row in baseline_visible
        ),
        "branch_consistent_measurements": len(consistent),
        "fit_coverage_on_baseline_visible": len(consistent) / len(baseline_visible),
        "accepted_measurements_all_frames": len(accepted),
        "geometry_inconsistent_accepted_fits": len(inconsistent),
        "geometry_inconsistency_denominator": len(accepted),
        "jointly_consistent_pair_frames": len(pairs),
        "pair_residual_denominator": len(pairs),
        "absolute_pair_distance_residual_px": {
            "median": float(np.median(residuals)),
            "p95": float(np.percentile(residuals, 95)),
            "max": max(residuals),
        },
        "post_gap_recovery": {
            "evaluated_gap_count": len(gaps),
            "recovered_gap_count": len(recovered),
            "rate": len(recovered) / len(gaps),
            "zero_delay_count": sum(
                int(row["recovery_delay_frames"]) == 0 for row in recovered
            ),
        },
    }


def case_report(
    case_spec: tuple[str, str, str, str, str, int],
    method: str,
    summary: Mapping[str, Any],
) -> CaseReportRecord:
    _case_id, _series_id, _pair_id, series, pair, _frame_count = case_spec
    accepted = int(summary["accepted_measurements_all_frames"])
    inconsistent = int(summary["geometry_inconsistent_accepted_fits"])
    gaps = summary["post_gap_recovery"]
    return CaseReportRecord(
        series=series,
        pair=pair,
        method=method,
        display_method=DISPLAY_METHODS[method],
        visible_opportunities=int(
            summary["physics_baseline_visible_opportunities"]
        ),
        branch_consistent_measurements=int(
            summary["branch_consistent_measurements"]
        ),
        fit_coverage=float(summary["fit_coverage_on_baseline_visible"]),
        evaluated_gaps=int(gaps["evaluated_gap_count"]),
        recovered_gaps=int(gaps["recovered_gap_count"]),
        gap_recovery_rate=float(gaps["rate"]),
        accepted_measurements_all_frames=accepted,
        geometry_inconsistent_accepted_fits=inconsistent,
        geometry_inconsistent_fraction=inconsistent / accepted,
        jointly_consistent_pair_frames=int(
            summary["jointly_consistent_pair_frames"]
        ),
        case_median_abs_distance_residual_px=float(
            summary["absolute_pair_distance_residual_px"]["median"]
        ),
    )


def aggregate_report(
    method: str,
    summary: Mapping[str, Any],
) -> AggregateReportRecord:
    accepted = int(summary["accepted_measurements_all_frames"])
    inconsistent = int(summary["geometry_inconsistent_accepted_fits"])
    gaps = summary["post_gap_recovery"]
    residuals = summary["absolute_pair_distance_residual_px"]
    return AggregateReportRecord(
        method=method,
        display_method=DISPLAY_METHODS[method],
        visible_opportunities=int(
            summary["physics_baseline_visible_opportunities"]
        ),
        branch_consistent_measurements=int(
            summary["branch_consistent_measurements"]
        ),
        weighted_fit_coverage=float(
            summary["fit_coverage_on_baseline_visible"]
        ),
        evaluated_gaps=int(gaps["evaluated_gap_count"]),
        recovered_gaps=int(gaps["recovered_gap_count"]),
        weighted_gap_recovery_rate=float(gaps["rate"]),
        accepted_measurements_all_frames=accepted,
        geometry_inconsistent_accepted_fits=inconsistent,
        geometry_inconsistent_fraction=inconsistent / accepted,
        jointly_consistent_pair_frames=int(
            summary["jointly_consistent_pair_frames"]
        ),
        median_abs_distance_residual_px=float(residuals["median"]),
        p95_abs_distance_residual_px=float(residuals["p95"]),
        maximum_abs_distance_residual_px=float(residuals["max"]),
    )




def reader_facing_panel(
    summary: Mapping[str, Any],
    rounding: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    rate_places = int(rounding["rates_percent"]["decimal_places"])
    residual_places = int(
        rounding["pair_distance_residual_px"]["decimal_places"]
    )
    result: dict[str, dict[str, str]] = {}
    for method in AGGREGATE_METHOD_ORDER:
        method_summary = summary["methods"][method]
        visible = int(method_summary["physics_baseline_visible_opportunities"])
        consistent = int(method_summary["branch_consistent_measurements"])
        accepted = int(method_summary["accepted_measurements_all_frames"])
        inconsistent = int(
            method_summary["geometry_inconsistent_accepted_fits"]
        )
        gaps = method_summary["post_gap_recovery"]
        result[method] = {
            "coverage": (
                f"{consistent}/{visible} "
                f"({100.0 * consistent / visible:.{rate_places}f}%)"
            ),
            "post_gap_recovery": (
                f"{int(gaps['recovered_gap_count'])}/"
                f"{int(gaps['evaluated_gap_count'])} "
                f"({100.0 * float(gaps['rate']):.{rate_places}f}%)"
            ),
            "geometry_inconsistency": (
                f"{inconsistent}/{accepted} "
                f"({100.0 * inconsistent / accepted:.{rate_places}f}%)"
            ),
            "conditional_pair_residual_median": (
                f"{float(method_summary['absolute_pair_distance_residual_px']['median']):.{residual_places}f} px "
                f"(n={int(method_summary['pair_residual_denominator'])})"
            ),
        }
    return result


__all__ = ["AGGREGATE_METHOD_ORDER", "aggregate_report", "case_report", "reader_facing_panel", "summarize_panel"]
