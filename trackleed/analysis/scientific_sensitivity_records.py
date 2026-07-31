"""Typed output records for Ag(111) sensitivity assessment."""

from __future__ import annotations

from typing import Literal, TypedDict, get_type_hints


SensitivityScope = Literal["case", "method"]


class SensitivityResultRecord(TypedDict):
    variant_id: str
    family: str
    scope: SensitivityScope
    case_id: str | None
    series_id: str | None
    pair_id: str | None
    method: str
    roi_side_px: int
    proposal_to_fit_max_px: float
    branch_tolerance_px: float
    r_squared_min: float
    amplitude_threshold_min: float
    sigma_min_px: float
    sigma_max_px: float
    candidate_records_all_frames: int
    proposal_available_all_frames: int
    solver_completed_fits_all_frames: int
    finite_candidate_fits_all_frames: int
    physics_baseline_visible_opportunities: int
    branch_consistent_measurements: int
    branch_consistent_coverage: float
    evaluated_gap_count: int
    branch_consistent_recovered_gap_count: int
    branch_consistent_recovery_rate: float
    accepted_measurements_all_frames: int
    geometry_inconsistent_accepted_fits: int
    geometry_inconsistency_denominator: int
    geometry_inconsistent_fraction: float
    jointly_consistent_pair_frames: int
    pair_residual_denominator: int
    median_absolute_pair_distance_residual_px: float
    p95_absolute_pair_distance_residual_px: float
    maximum_absolute_pair_distance_residual_px: float


class RejectionCauseCountRecord(TypedDict):
    variant_id: str
    family: str
    scope: SensitivityScope
    case_id: str | None
    method: str
    cause: str
    first_rejection_count: int
    full_rejection_count: int


class WidthTriggerRecord(TypedDict):
    scope: SensitivityScope
    case_id: str | None
    method: str
    finite_candidate_fit_count: int
    near_width_boundary_count: int
    near_width_boundary_threshold_count: int
    width_only_rejection_count: int
    width_only_rejection_threshold_count: int
    gate_margin_triggered: bool
    width_only_rejection_triggered: bool
    drives_global_decision: bool


SENSITIVITY_RESULT_FIELDS = tuple(get_type_hints(SensitivityResultRecord))
REJECTION_CAUSE_COUNT_FIELDS = tuple(get_type_hints(RejectionCauseCountRecord))
WIDTH_TRIGGER_FIELDS = tuple(get_type_hints(WidthTriggerRecord))


__all__ = [
    "REJECTION_CAUSE_COUNT_FIELDS",
    "SENSITIVITY_RESULT_FIELDS",
    "WIDTH_TRIGGER_FIELDS",
    "RejectionCauseCountRecord",
    "SensitivityResultRecord",
    "SensitivityScope",
    "WidthTriggerRecord",
]
