"""Typed record schemas and CSV codecs for the scientific tracer."""

from __future__ import annotations

import csv
from pathlib import Path
import types
from typing import (
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


MeasurementStatus = Literal[
    "proposal_unavailable",
    "roi_invalid",
    "solver_failure",
    "non_finite_parameters",
    "gaussian_rejected",
    "geometry_inconsistent",
    "branch_consistent",
]
RejectionCause = Literal[
    "proposal_unavailable",
    "roi_invalid",
    "solver_failure",
    "non_finite_parameters",
    "proposal_to_fit",
    "r_squared",
    "amplitude",
    "width",
    "physics_branch_consistency",
]
REJECTION_CAUSES = get_args(RejectionCause)
SolverStatus = Literal["not_run", "failed", "completed"]
MissingState = Literal[
    "proposal_unavailable",
    "roi_outside_image",
    "solver_failure",
    "non_finite_parameters",
    "none",
]


class CandidateRecord(TypedDict):
    record_key: str
    case_id: str
    series_id: str
    pair_id: str
    method: str
    frame: int
    energy_ev: float
    branch: str
    physics_baseline_status: str
    opportunity_key: str | None
    image_artifact_id: str
    image_source_path: str
    image_sha256: str
    proposal_source_path: str
    proposal_sha256: str
    proposal_available: bool
    native_x: float | None
    native_y: float | None
    native_mask_area: float | None
    native_mask_score: float | None
    native_mask_component_count: int | None
    predicted_x: float
    predicted_y: float
    roi_side_px: int
    crop_x0: int | None
    crop_y0: int | None
    crop_x1: int | None
    crop_y1: int | None
    roi_valid: bool
    solver_status: SolverStatus
    finite_candidate: bool
    missing_state: MissingState
    center_x: float | None
    center_y: float | None
    amplitude: float | None
    r_squared: float | None
    volume: float | None
    sigma_x: float | None
    sigma_y: float | None
    theta: float | None
    offset: float | None
    center_x_error: float | None
    center_y_error: float | None
    amplitude_error: float | None
    sigma_x_error: float | None
    sigma_y_error: float | None
    theta_error: float | None
    offset_error: float | None
    volume_error: float | None
    proposal_offset_px: float | None
    branch_error_px: float | None
    radial_residual_px: float | None
    tangential_residual_px: float | None
    proposal_to_fit_margin_px: float | None
    r_squared_margin: float | None
    amplitude_margin: float | None
    sigma_x_lower_margin_px: float | None
    sigma_x_upper_margin_px: float | None
    sigma_y_lower_margin_px: float | None
    sigma_y_upper_margin_px: float | None
    branch_margin_px: float | None


class EvaluationRecord(TypedDict):
    record_key: str
    case_id: str
    method: str
    frame: int
    energy_ev: float
    branch: str
    physics_baseline_status: str
    opportunity_key: str | None
    variant_id: str
    accepted_fit: bool
    measurement_valid: bool
    measurement_status: MeasurementStatus
    classification: MeasurementStatus
    first_rejection_cause: RejectionCause | None
    full_rejection_causes: str
    legacy_fit_reason: str
    proposal_to_fit_margin_px: float | None
    r_squared_margin: float | None
    amplitude_margin: float | None
    sigma_x_lower_margin_px: float | None
    sigma_x_upper_margin_px: float | None
    sigma_y_lower_margin_px: float | None
    sigma_y_upper_margin_px: float | None
    branch_margin_px: float | None


class PairRecord(TypedDict):
    tracker: str
    frame: int
    energy_ev: float
    pair_distance_px: float
    predicted_distance_px: float
    distance_residual_px: float
    midpoint_x: float
    midpoint_y: float
    midpoint_drift_px: float


class GapRecord(TypedDict):
    tracker: str
    branch: str
    direction: str
    gap_start_energy_ev: float
    gap_end_energy_ev: float
    gap_frame_count: int
    first_expected_recovery_energy_ev: float
    recovered: bool
    recovery_energy_ev: float | None
    recovery_delay_frames: int | None


CANDIDATE_FIELDS = tuple(get_type_hints(CandidateRecord))
EVALUATION_FIELDS = tuple(get_type_hints(EvaluationRecord))
PAIR_FIELDS = tuple(get_type_hints(PairRecord))
GAP_FIELDS = tuple(get_type_hints(GapRecord))
MARGIN_FIELDS = (
    "proposal_to_fit_margin_px",
    "r_squared_margin",
    "amplitude_margin",
    "sigma_x_lower_margin_px",
    "sigma_x_upper_margin_px",
    "sigma_y_lower_margin_px",
    "sigma_y_upper_margin_px",
    "branch_margin_px",
)


Record = TypeVar("Record", bound=Mapping[str, object])


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected frozen boolean, got {value!r}")


def _field_spec(
    annotation: object,
) -> tuple[type[object], bool, frozenset[object] | None]:
    nullable = False
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        union_members = get_args(annotation)
        nullable = type(None) in union_members
        members = [value for value in union_members if value is not type(None)]
        if len(members) != 1:
            raise TypeError(f"Unsupported tracer record type: {annotation!r}")
        annotation = members[0]
        origin = get_origin(annotation)
    allowed_values = None
    if origin is Literal:
        literal_values = get_args(annotation)
        literal_types = {type(value) for value in literal_values}
        if len(literal_types) != 1:
            raise TypeError(f"Unsupported tracer record type: {annotation!r}")
        annotation = literal_types.pop()
        allowed_values = frozenset(literal_values)
    if annotation not in {bool, float, int, str}:
        raise TypeError(f"Unsupported tracer record type: {annotation!r}")
    return annotation, nullable, allowed_values


def decode_csv_row(
    row: Mapping[str, str],
    record_type: type[Record],
) -> Record:
    """Decode one complete CSV row according to its typed record schema."""

    annotations = get_type_hints(record_type)
    if set(row) != set(annotations):
        raise ValueError("Tracer CSV row does not match its typed schema")
    decoded: dict[str, object] = {}
    for field, annotation in annotations.items():
        value = row[field]
        value_type, nullable, allowed_values = _field_spec(annotation)
        if value == "" and nullable:
            decoded[field] = None
        elif value == "" and value_type is not str:
            raise ValueError(f"Required tracer field is blank: {field}")
        elif value_type is bool:
            decoded[field] = parse_bool(value)
        else:
            decoded[field] = value_type(value)
        if (
            decoded[field] is not None
            and allowed_values is not None
            and decoded[field] not in allowed_values
        ):
            raise ValueError(f"Invalid tracer value for {field}: {decoded[field]!r}")
    return decoded  # type: ignore[return-value]


def read_csv(
    path: Path,
    record_type: type[Record],
) -> tuple[list[str], list[Record]]:
    """Read and type-decode a complete tracer record CSV."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        expected = list(get_type_hints(record_type))
        if fields != expected:
            raise ValueError(f"Tracer CSV schema changed: {path.name}")
        return fields, [decode_csv_row(row, record_type) for row in reader]


def read_raw_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV whose schema is owned outside the tracer record layer."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(
    path: Path,
    record_type: type[Record],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write records using the field order declared by their typed schema."""

    fields = list(get_type_hints(record_type))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "CANDIDATE_FIELDS",
    "EVALUATION_FIELDS",
    "GAP_FIELDS",
    "MARGIN_FIELDS",
    "PAIR_FIELDS",
    "REJECTION_CAUSES",
    "CandidateRecord",
    "EvaluationRecord",
    "GapRecord",
    "MeasurementStatus",
    "MissingState",
    "PairRecord",
    "RejectionCause",
    "SolverStatus",
    "decode_csv_row",
    "parse_bool",
    "read_csv",
    "read_raw_csv",
    "write_csv",
]
