"""Typed analysis of the frozen controlled-translation evidence matrix."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Literal, cast

from trackleed.publication.controlled_translation_sources import (
    METHOD_ORDER,
    TRIAL_LABELS,
    TRIAL_ORDER,
    read_csv,
    validate_controlled_translation_source_data,
)


MethodId = Literal[
    "locotrack_base_offline",
    "ostrack_vitb_256_mae_ce",
    "dam4sam_sam21pp_l",
    "lucas_kanade_skip_anchor",
    "easyleed_kalman",
    "oracle_roi_control",
]
SourceFrameOrder = Literal[0, 372, 378, 700]
EvidenceRole = Literal["tracker", "oracle_control"]
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
    pair: str
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

    def as_record(self) -> dict[str, int | float | str]:
        """Return the stable, full-precision Table S5 record."""

        return {
            "source_frame_order": self.source_frame_order,
            "trial": self.trial,
            "temperature_c": self.temperature_c,
            "method": self.method,
            "display_method": self.display_method,
            "role": self.role,
            "available_proposal_count": self.proposal_coverage.numerator,
            "spot_opportunity_n": self.proposal_coverage.denominator,
            "proposal_coverage": self.proposal_coverage.fraction,
            "accepted_measurement_count": self.measurement_coverage.numerator,
            "measurement_coverage": self.measurement_coverage.fraction,
            "conditional_measurement_error_n": (
                self.conditional_measurement_error_n
            ),
            "measurement_error_median_px": self.measurement_error_median_px,
            "measurement_error_p95_px": self.measurement_error_p95_px,
            "measurement_error_max_px": self.measurement_error_max_px,
            "valid_pair_count": self.pair_coverage.numerator,
            "pair_opportunity_n": self.pair_coverage.denominator,
            "pair_coverage": self.pair_coverage.fraction,
            "conditional_pair_invariance_n": (
                self.conditional_pair_invariance_n
            ),
            "maximum_abs_delta_d_over_d_pp": (
                self.maximum_abs_delta_d_over_d_pp
            ),
        }


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


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
    validate_controlled_translation_source_data(source_directory)
    spot_rows = read_csv(source_directory / "equivariance_results.csv")
    pair_rows = read_csv(source_directory / "pair_invariance.csv")
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
            pair=row["pair"],
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
    "MethodId",
    "PairObservation",
    "SourceFrameOrder",
    "SpotObservation",
    "TrialOutcome",
    "build_trial_outcomes",
    "load_controlled_translation_evidence",
]
