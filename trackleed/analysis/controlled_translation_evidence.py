"""Typed analysis of the frozen controlled-translation evidence matrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Literal, cast

MethodId = Literal[
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
    "oracle_roi_control",
]
PairId = Literal["1-6", "3-5"]
PAIR_IDS: tuple[PairId, ...] = ("1-6", "3-5")
SourceFrameOrder = Literal[0, 372, 378, 700]
EvidenceRole = Literal["tracker", "oracle_control"]
EVIDENCE_ROLES: tuple[EvidenceRole, ...] = ("tracker", "oracle_control")
METHOD_ORDER: tuple[MethodId, ...] = (
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
    "oracle_roi_control",
)
TRIAL_ORDER: tuple[SourceFrameOrder, ...] = (0, 372, 378, 700)
TRIAL_LABELS: dict[SourceFrameOrder, str] = {
    0: "initial low-temperature state",
    372: "800 deg C plateau onset",
    378: "post-plateau contraction minimum",
    700: "final cooled state",
}
SPOT_OPPORTUNITIES_PER_TRIAL = 20
PAIR_OPPORTUNITIES_PER_TRIAL = 10


@dataclass(frozen=True)
class Coverage:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise ValueError("Coverage requires 0 <= numerator <= denominator")

    @property
    def fraction(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class SpotObservation:
    method: MethodId
    source_frame_order: SourceFrameOrder
    spot_id: int
    translation_step: int
    commanded_dx_px: float
    commanded_dy_px: float
    is_evaluation_step: bool
    proposal_available: bool
    fit_valid: bool
    native_displacement_error_px: float | None
    measurement_displacement_error_px: float | None
    role: EvidenceRole
    display_method: str
    temperature_c: int
    gaussian_center_x_px: float | None
    gaussian_center_y_px: float | None


@dataclass(frozen=True)
class PairObservation:
    method: MethodId
    source_frame_order: SourceFrameOrder
    pair: PairId
    translation_step: int
    is_evaluation_step: bool
    pair_valid: bool
    delta_d_over_d_percent: float | None
    role: EvidenceRole
    display_method: str
    temperature_c: int


@dataclass(frozen=True)
class ControlledTranslationEvidence:
    spots: tuple[SpotObservation, ...]
    pairs: tuple[PairObservation, ...]


@dataclass(frozen=True)
class TrialOutcome:
    source_frame_order: SourceFrameOrder
    trial: str
    temperature_c: int
    method: MethodId
    display_method: str
    role: EvidenceRole
    proposal_coverage: Coverage
    measurement_coverage: Coverage
    conditional_measurement_error_n: int
    measurement_error_median_px: float
    measurement_error_p95_px: float
    measurement_error_max_px: float
    pair_coverage: Coverage
    conditional_pair_invariance_n: int
    maximum_abs_delta_d_over_d_pp: float

def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def _as_pair_id(value: str) -> PairId:
    if value not in PAIR_IDS:
        raise ValueError(f"Controlled-translation pair inventory changed: {value}")
    return cast(PairId, value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _linear_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("A conditional percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def load_controlled_translation_evidence(
    source_directory: Path,
) -> ControlledTranslationEvidence:
    """Parse the authenticated source tables into typed observations."""

    source_directory = source_directory.resolve()
    spot_rows = _read_csv(source_directory / "equivariance_results.csv")
    pair_rows = _read_csv(source_directory / "pair_invariance.csv")
    spots = tuple(
        SpotObservation(
            method=cast(MethodId, row["method"]),
            source_frame_order=cast(
                SourceFrameOrder,
                int(row["source_frame_order"]),
            ),
            spot_id=int(row["spot_id"]),
            translation_step=int(row["translation_step"]),
            commanded_dx_px=float(row["commanded_dx_original_px"]),
            commanded_dy_px=float(row["commanded_dy_original_px"]),
            is_evaluation_step=_as_bool(row["is_evaluation_step"]),
            proposal_available=_as_bool(row["proposal_available"]),
            fit_valid=_as_bool(row["fit_valid"]),
            native_displacement_error_px=_optional_float(
                row["native_displacement_error_px"]
            ),
            measurement_displacement_error_px=_optional_float(
                row["measurement_displacement_error_px"]
            ),
            role=cast(EvidenceRole, row["role"]),
            display_method=row["display_method"],
            temperature_c=int(float(row["temperature_c"])),
            gaussian_center_x_px=_optional_float(
                row["gaussian_center_x_original_px"]
            ),
            gaussian_center_y_px=_optional_float(
                row["gaussian_center_y_original_px"]
            ),
        )
        for row in spot_rows
    )
    pairs = tuple(
        PairObservation(
            method=cast(MethodId, row["method"]),
            source_frame_order=cast(
                SourceFrameOrder,
                int(row["source_frame_order"]),
            ),
            pair=_as_pair_id(row["pair"]),
            translation_step=int(row["translation_step"]),
            is_evaluation_step=_as_bool(row["is_evaluation_step"]),
            pair_valid=_as_bool(row["pair_valid"]),
            delta_d_over_d_percent=_optional_float(
                row["delta_d_over_d_percent"]
            ),
            role=cast(EvidenceRole, row["role"]),
            display_method=row["display_method"],
            temperature_c=int(float(row["temperature_c"])),
        )
        for row in pair_rows
    )
    if (
        len(spots) != 576
        or len(pairs) != 288
        or {row.method for row in spots} != set(METHOD_ORDER)
        or {row.method for row in pairs} != set(METHOD_ORDER)
        or {row.source_frame_order for row in spots} != set(TRIAL_ORDER)
        or {row.source_frame_order for row in pairs} != set(TRIAL_ORDER)
        or {row.role for row in spots} != set(EVIDENCE_ROLES)
        or {row.role for row in pairs} != set(EVIDENCE_ROLES)
        or {row.pair for row in pairs} != set(PAIR_IDS)
    ):
        raise ValueError("Controlled-translation evidence matrix changed")
    return ControlledTranslationEvidence(spots=spots, pairs=pairs)


def build_trial_outcomes(source_directory: Path) -> list[TrialOutcome]:
    """Build one full-precision outcome per source trial and method."""

    evidence = load_controlled_translation_evidence(source_directory)
    outcomes: list[TrialOutcome] = []
    for trial in TRIAL_ORDER:
        for method in METHOD_ORDER:
            trial_spots = [
                row
                for row in evidence.spots
                if row.method == method
                and row.source_frame_order == trial
                and row.is_evaluation_step
            ]
            trial_pairs = [
                row
                for row in evidence.pairs
                if row.method == method
                and row.source_frame_order == trial
                and row.is_evaluation_step
            ]
            if (
                len(trial_spots) != SPOT_OPPORTUNITIES_PER_TRIAL
                or len(trial_pairs) != PAIR_OPPORTUNITIES_PER_TRIAL
            ):
                raise ValueError(
                    f"Controlled-translation opportunity set changed: {trial}/{method}"
                )
            measurement_errors = [
                row.measurement_displacement_error_px
                for row in trial_spots
                if row.fit_valid and row.measurement_displacement_error_px is not None
            ]
            pair_changes = [
                abs(row.delta_d_over_d_percent)
                for row in trial_pairs
                if row.pair_valid and row.delta_d_over_d_percent is not None
            ]
            if not measurement_errors or not pair_changes:
                raise ValueError(
                    f"Controlled-translation conditional outcome is empty: "
                    f"{trial}/{method}"
                )
            proposal_coverage = Coverage(
                sum(row.proposal_available for row in trial_spots),
                SPOT_OPPORTUNITIES_PER_TRIAL,
            )
            measurement_coverage = Coverage(
                len(measurement_errors),
                SPOT_OPPORTUNITIES_PER_TRIAL,
            )
            pair_coverage = Coverage(
                len(pair_changes),
                PAIR_OPPORTUNITIES_PER_TRIAL,
            )
            outcomes.append(
                TrialOutcome(
                    source_frame_order=cast(SourceFrameOrder, trial),
                    trial=TRIAL_LABELS[trial],
                    temperature_c=trial_spots[0].temperature_c,
                    method=cast(MethodId, method),
                    display_method=trial_spots[0].display_method,
                    role=trial_spots[0].role,
                    proposal_coverage=proposal_coverage,
                    measurement_coverage=measurement_coverage,
                    conditional_measurement_error_n=len(measurement_errors),
                    measurement_error_median_px=statistics.median(
                        measurement_errors
                    ),
                    measurement_error_p95_px=_linear_percentile(
                        measurement_errors,
                        95.0,
                    ),
                    measurement_error_max_px=max(measurement_errors),
                    pair_coverage=pair_coverage,
                    conditional_pair_invariance_n=len(pair_changes),
                    maximum_abs_delta_d_over_d_pp=max(pair_changes),
                )
            )
    if len(outcomes) != len(TRIAL_ORDER) * len(METHOD_ORDER):
        raise ValueError("Controlled-translation trial matrix changed")
    return outcomes


__all__ = [
    "ControlledTranslationEvidence",
    "Coverage",
    "EvidenceRole",
    "EVIDENCE_ROLES",
    "MethodId",
    "METHOD_ORDER",
    "PAIR_IDS",
    "PairObservation",
    "PairId",
    "SourceFrameOrder",
    "SpotObservation",
    "TRIAL_LABELS",
    "TRIAL_ORDER",
    "TrialOutcome",
    "build_trial_outcomes",
    "load_controlled_translation_evidence",
]
