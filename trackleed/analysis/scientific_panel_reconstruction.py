"""Orchestrate complete Ag(111) reconstruction behind one typed interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trackleed.analysis.scientific_panel_candidates import (
    central_contract,
    panel_record_key,
    reconstruct_panel_candidates,
)
from trackleed.analysis.scientific_panel_common import (
    CASE_SPECS,
    GAPS_PATH,
    METHODS,
)
from trackleed.analysis.scientific_panel_outcomes import (
    reconstruct_panel_gaps,
    reconstruct_panel_pairs,
)
from trackleed.analysis.scientific_panel_records import (
    AggregateReportRecord,
    CaseReportRecord,
    PanelGapRecord,
    PanelPairRecord,
)
from trackleed.analysis.scientific_panel_reporting import (
    AGGREGATE_METHOD_ORDER,
    aggregate_report,
    case_report,
    reader_facing_panel,
    summarize_panel,
)
from trackleed.analysis.scientific_tracer_evaluation import evaluate_candidates
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
    read_raw_csv,
)


@dataclass(frozen=True)
class PanelArtifacts:
    candidates: list[CandidateRecord]
    evaluations: list[EvaluationRecord]
    pairs: list[PanelPairRecord]
    gaps: list[PanelGapRecord]
    case_reports: list[CaseReportRecord]
    aggregate_reports: list[AggregateReportRecord]
    summary: dict[str, Any]


def reconstruct_panel_artifacts(
    workspace_root: Path,
    contract: Mapping[str, Any],
) -> PanelArtifacts:
    """Reconstruct all central-panel records behind one typed interface."""

    candidates = reconstruct_panel_candidates(workspace_root, contract)
    evaluations = evaluate_candidates(candidates, central_contract(workspace_root))
    _, frozen_gaps = read_raw_csv(workspace_root / GAPS_PATH)
    pairs: list[PanelPairRecord] = []
    gaps: list[PanelGapRecord] = []
    case_reports: list[CaseReportRecord] = []
    case_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for case_spec in CASE_SPECS:
        case_id, series_id, pair_id, _series, _pair, _frame_count = case_spec
        case_summaries[case_id] = {}
        for method in METHODS:
            selected_candidates = [
                row
                for row in candidates
                if row["case_id"] == case_id and row["method"] == method
            ]
            keys = {row["record_key"] for row in selected_candidates}
            selected_evaluations = [
                row for row in evaluations if row["record_key"] in keys
            ]
            selected_pairs = reconstruct_panel_pairs(
                selected_candidates,
                selected_evaluations,
                workspace_root,
                case_id,
                series_id,
                pair_id,
                method,
            )
            selected_gaps = reconstruct_panel_gaps(
                selected_evaluations,
                workspace_root,
                case_id,
                series_id,
                pair_id,
                method,
                [row for row in frozen_gaps if row["case_id"] == case_id],
            )
            case_summary = summarize_panel(
                selected_candidates,
                selected_evaluations,
                selected_pairs,
                selected_gaps,
            )
            case_summaries[case_id][method] = case_summary
            case_reports.append(case_report(case_spec, method, case_summary))
            pairs.extend(selected_pairs)
            gaps.extend(selected_gaps)

    method_summaries: dict[str, dict[str, Any]] = {}
    aggregate_reports: list[AggregateReportRecord] = []
    for method in AGGREGATE_METHOD_ORDER:
        method_candidates = [row for row in candidates if row["method"] == method]
        method_keys = {row["record_key"] for row in method_candidates}
        method_summary = summarize_panel(
            method_candidates,
            [row for row in evaluations if row["record_key"] in method_keys],
            [row for row in pairs if row["method"] == method],
            [row for row in gaps if row["method"] == method],
        )
        method_summaries[method] = method_summary
        aggregate_reports.append(aggregate_report(method, method_summary))

    summary = {
        "reference_sets": {
            "opportunities": {
                "set_id": "baseline_visible_opportunities_v1",
                "n": 221,
            },
            "gaps": {"set_id": "evaluable_iv_gaps_v1", "n": 24},
        },
        "cases": case_summaries,
        "methods": method_summaries,
    }
    return PanelArtifacts(
        candidates=candidates,
        evaluations=evaluations,
        pairs=pairs,
        gaps=gaps,
        case_reports=case_reports,
        aggregate_reports=aggregate_reports,
        summary=summary,
    )




__all__ = [
    "PanelArtifacts",
    "panel_record_key",
    "reader_facing_panel",
    "reconstruct_panel_artifacts",
    "reconstruct_panel_candidates",
]
