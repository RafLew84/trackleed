"""Predeclared gate-margin and width-only materiality audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from trackleed.analysis.scientific_panel_common import CASE_SPECS, METHODS
from trackleed.analysis.scientific_sensitivity_records import (
    SensitivityScope,
    WidthTriggerRecord,
)
from trackleed.analysis.scientific_tracer_records import (
    CandidateRecord,
    EvaluationRecord,
)


@dataclass(frozen=True)
class WidthTriggerAudit:
    method_records: tuple[WidthTriggerRecord, ...]
    case_records: tuple[WidthTriggerRecord, ...]
    triggered_methods: tuple[str, ...]
    run_width_sensitivity: bool


def _is_width_only_rejection(evaluation: EvaluationRecord) -> bool:
    causes = {
        cause
        for cause in evaluation["full_rejection_causes"].split("+")
        if cause and cause != "physics_branch_consistency"
    }
    return causes == {"width"}


def audit_width_trigger(
    candidates: Sequence[CandidateRecord],
    evaluations: Sequence[EvaluationRecord],
    contract: dict[str, Any],
) -> WidthTriggerAudit:
    """Evaluate the predeclared width trigger separately for every method."""

    central = contract["central_contract"]
    width_gate = central["width_gate"]
    sigma_min = float(width_gate["sigma_min_px"])
    sigma_max = float(width_gate["sigma_max_px"])
    width_span = sigma_max - sigma_min
    trigger = contract["conditional_width_sensitivity"]["trigger"]
    gate_rule = trigger["gate_margin"]
    near_fraction = float(gate_rule["minimum_fraction"])
    near_distance = float(gate_rule["normalized_distance_to_bound_max"])
    width_only_threshold = int(trigger["width_only_rejection"]["minimum_count"])
    evaluation_by_key = {row["record_key"]: row for row in evaluations}
    method_records: list[WidthTriggerRecord] = []
    case_records: list[WidthTriggerRecord] = []
    triggered_methods: list[str] = []

    def record_for(
        selected: Sequence[CandidateRecord],
        *,
        scope: SensitivityScope,
        case_id: str | None,
        method: str,
    ) -> WidthTriggerRecord:
        finite = [row for row in selected if row["finite_candidate"]]
        near_boundary = [
            row
            for row in finite
            if min(
                abs(float(row["sigma_x"]) - sigma_min),
                abs(sigma_max - float(row["sigma_x"])),
                abs(float(row["sigma_y"]) - sigma_min),
                abs(sigma_max - float(row["sigma_y"])),
            )
            / width_span
            <= near_distance
        ]
        near_threshold = math.ceil(near_fraction * len(finite))
        width_only = [
            row
            for row in selected
            if row["opportunity_key"] is not None
            and _is_width_only_rejection(evaluation_by_key[row["record_key"]])
        ]
        return WidthTriggerRecord(
            scope=scope,
            case_id=case_id,
            method=method,
            finite_candidate_fit_count=len(finite),
            near_width_boundary_count=len(near_boundary),
            near_width_boundary_threshold_count=near_threshold,
            width_only_rejection_count=len(width_only),
            width_only_rejection_threshold_count=width_only_threshold,
            gate_margin_triggered=bool(finite) and len(near_boundary) >= near_threshold,
            width_only_rejection_triggered=(len(width_only) >= width_only_threshold),
            drives_global_decision=scope == "method",
        )

    for method in METHODS:
        method_candidates = [row for row in candidates if row["method"] == method]
        method_record = record_for(
            method_candidates,
            scope="method",
            case_id=None,
            method=method,
        )
        if (
            method_record["gate_margin_triggered"]
            or method_record["width_only_rejection_triggered"]
        ):
            triggered_methods.append(method)
        method_records.append(method_record)
        for case_spec in CASE_SPECS:
            case_records.append(
                record_for(
                    [
                        row
                        for row in method_candidates
                        if row["case_id"] == case_spec.case_id
                    ],
                    scope="case",
                    case_id=case_spec.case_id,
                    method=method,
                )
            )

    return WidthTriggerAudit(
        method_records=tuple(method_records),
        case_records=tuple(case_records),
        triggered_methods=tuple(triggered_methods),
        run_width_sensitivity=bool(triggered_methods),
    )


__all__ = ["WidthTriggerAudit", "audit_width_trigger"]
