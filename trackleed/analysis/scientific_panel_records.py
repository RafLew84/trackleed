"""Typed record schemas emitted by the complete Ag(111) central panel."""

from __future__ import annotations

from typing import TypedDict, get_type_hints


class PanelPairRecord(TypedDict):
    record_key: str
    case_id: str
    series_id: str
    pair_id: str
    method: str
    frame: int
    energy_ev: float
    pair_distance_px: float
    predicted_distance_px: float
    distance_residual_px: float
    midpoint_x: float
    midpoint_y: float
    midpoint_drift_px: float


class PanelGapRecord(TypedDict):
    gap_key: str
    case_id: str
    series_id: str
    pair_id: str
    method: str
    branch: str
    direction: str
    gap_start_energy_ev: float
    gap_end_energy_ev: float
    gap_frame_count: int
    first_expected_recovery_energy_ev: float
    recovered: bool
    recovery_energy_ev: float | None
    recovery_delay_frames: int | None


class CaseReportRecord(TypedDict):
    series: str
    pair: str
    method: str
    display_method: str
    visible_opportunities: int
    branch_consistent_measurements: int
    fit_coverage: float
    evaluated_gaps: int
    recovered_gaps: int
    gap_recovery_rate: float
    accepted_measurements_all_frames: int
    geometry_inconsistent_accepted_fits: int
    geometry_inconsistent_fraction: float
    jointly_consistent_pair_frames: int
    case_median_abs_distance_residual_px: float


class AggregateReportRecord(TypedDict):
    method: str
    display_method: str
    visible_opportunities: int
    branch_consistent_measurements: int
    weighted_fit_coverage: float
    evaluated_gaps: int
    recovered_gaps: int
    weighted_gap_recovery_rate: float
    accepted_measurements_all_frames: int
    geometry_inconsistent_accepted_fits: int
    geometry_inconsistent_fraction: float
    jointly_consistent_pair_frames: int
    median_abs_distance_residual_px: float
    p95_abs_distance_residual_px: float
    maximum_abs_distance_residual_px: float


PANEL_PAIR_FIELDS = tuple(get_type_hints(PanelPairRecord))
PANEL_GAP_FIELDS = tuple(get_type_hints(PanelGapRecord))
CASE_REPORT_FIELDS = tuple(get_type_hints(CaseReportRecord))
AGGREGATE_REPORT_FIELDS = tuple(get_type_hints(AggregateReportRecord))
HISTORICAL_CENTRAL_FIELDS = (
    "case_id",
    "series_id",
    "pair_id",
    "tracker",
    "frame",
    "energy_ev",
    "branch",
    "physics_baseline_status",
    "native_visible",
    "native_x",
    "native_y",
    "native_mask_area",
    "native_mask_score",
    "native_mask_component_count",
    "predicted_x",
    "predicted_y",
    "fit_valid",
    "fit_reason",
    "measurement_valid",
    "measurement_status",
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


__all__ = [
    "AGGREGATE_REPORT_FIELDS",
    "CASE_REPORT_FIELDS",
    "HISTORICAL_CENTRAL_FIELDS",
    "PANEL_GAP_FIELDS",
    "PANEL_PAIR_FIELDS",
    "AggregateReportRecord",
    "CaseReportRecord",
    "PanelGapRecord",
    "PanelPairRecord",
]
