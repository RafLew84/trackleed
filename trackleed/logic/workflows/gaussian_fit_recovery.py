"""Deterministic Gaussian fitting attempts and bounded recovery records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class GaussianFitterConfig:
    fitter_id: str = "trackleed_gaussian_curve_fit_v1"
    center_preference_factor: float | None = None
    theta_bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class FitValidation:
    min_r_squared: float = 0.85
    min_amplitude: float = 0.05
    center_margin: float = 2.0
    min_sigma: float = 0.5


@dataclass(frozen=True)
class FitRecoveryConfig:
    enabled: bool = False
    policy_version: str = "bounded-v1"


class RecoveryAttemptStatus(str, Enum):
    ACCEPTED = "accepted"
    FIT_FAILED = "fit_failed"
    VALIDATION_FAILED = "validation_failed"
    OUTSIDE_ORIGINAL_MARGIN = "outside_original_margin"
    OUT_OF_BOUNDS = "out_of_bounds"


class RecoveryStatus(str, Enum):
    DISABLED = "disabled"
    NOT_NEEDED = "not_needed"
    RECOVERED = "recovered"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous_recovery"


class RecoverySelectionReason(str, Enum):
    BASELINE_FIT_ACCEPTED = "baseline_fit_accepted"
    RECOVERY_DISABLED = "recovery_disabled"
    RECOVERY_FAILED = "recovery_failed"
    AMBIGUOUS_CANDIDATES = "accepted_candidates_exceed_1px_tolerance"
    SMALLEST_TRANSFORM_HIGHEST_QUALITY = "smallest_transform_then_highest_r_squared"


class RecoveryFailureCode(str, Enum):
    FIT_FAILED = "fit_failed"
    VALIDATION_FAILED = "validation_failed"
    RECOVERY_FAILED = "recovery_failed"
    AMBIGUOUS_RECOVERY = "ambiguous_recovery"


@dataclass(frozen=True)
class RecoveryAttempt:
    index: int
    offset_x: int
    offset_y: int
    expand_by: int
    effective_roi: Rect
    status: RecoveryAttemptStatus
    r_squared: float | None = None
    global_center_x: float | None = None
    global_center_y: float | None = None
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryRecord:
    status: RecoveryStatus
    policy_version: str
    original_roi: Rect
    effective_roi: Rect | None
    attempts: tuple[RecoveryAttempt, ...]
    selection_reason: RecoverySelectionReason


@dataclass(frozen=True)
class GaussianRecoveryOutcome:
    result: Mapping[str, Any] | None
    record: RecoveryRecord
    failure_code: RecoveryFailureCode | None = None
    failure_message: str | None = None
    failure_reasons: tuple[str, ...] = ()


class GaussianRecoveryCancelled(RuntimeError):
    """Raised instead of returning a partial recovery history."""


_FALLBACK_TRANSFORMS = (
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (0, 0, 2),
)


def fit_gaussian_with_recovery(
    image: np.ndarray,
    original_roi: Rect,
    fitter: GaussianFitterConfig,
    validation: FitValidation,
    recovery: FitRecoveryConfig,
    is_cancelled: Callable[[], bool],
) -> GaussianRecoveryOutcome:
    """Run the baseline Gaussian attempt and return immutable provenance."""
    if is_cancelled():
        raise GaussianRecoveryCancelled("Gaussian fit recovery was cancelled")
    raw_result, attempt = _run_attempt(
        image,
        original_roi,
        original_roi,
        fitter,
        validation,
        0,
        0,
        0,
        0,
        enforce_original_margin=False,
    )
    if is_cancelled():
        raise GaussianRecoveryCancelled("Gaussian fit recovery was cancelled")
    if attempt.status is RecoveryAttemptStatus.ACCEPTED:
        return _successful_outcome(
            raw_result,
            attempt,
            original_roi,
            recovery,
        )
    if not recovery.enabled:
        return _failed_outcome(original_roi, recovery, (attempt,))
    attempts, candidates = _run_fallbacks(
        image,
        original_roi,
        fitter,
        validation,
        is_cancelled,
    )
    all_attempts = (attempt, *attempts)
    if candidates:
        if _candidates_disagree(candidates):
            return GaussianRecoveryOutcome(
                None,
                RecoveryRecord(
                    RecoveryStatus.AMBIGUOUS,
                    recovery.policy_version,
                    original_roi,
                    None,
                    all_attempts,
                    RecoverySelectionReason.AMBIGUOUS_CANDIDATES,
                ),
                RecoveryFailureCode.AMBIGUOUS_RECOVERY,
                "Accepted Gaussian recovery candidates disagree",
                ("candidate_separation",),
            )
        selected_result, selected_attempt = _select_candidate(candidates)
        return GaussianRecoveryOutcome(
            selected_result,
            RecoveryRecord(
                RecoveryStatus.RECOVERED,
                recovery.policy_version,
                original_roi,
                selected_attempt.effective_roi,
                all_attempts,
                RecoverySelectionReason.SMALLEST_TRANSFORM_HIGHEST_QUALITY,
            ),
        )
    return _failed_outcome(original_roi, recovery, all_attempts)


def _successful_outcome(
    raw_result: Mapping[str, Any] | None,
    attempt: RecoveryAttempt,
    original_roi: Rect,
    recovery: FitRecoveryConfig,
) -> GaussianRecoveryOutcome:
    status = RecoveryStatus.NOT_NEEDED if recovery.enabled else RecoveryStatus.DISABLED
    return GaussianRecoveryOutcome(
        raw_result,
        RecoveryRecord(
            status,
            recovery.policy_version,
            original_roi,
            original_roi,
            (attempt,),
            RecoverySelectionReason.BASELINE_FIT_ACCEPTED,
        ),
    )


def _failed_outcome(
    original_roi: Rect,
    recovery: FitRecoveryConfig,
    attempts: tuple[RecoveryAttempt, ...],
) -> GaussianRecoveryOutcome:
    baseline = attempts[0]
    enabled = recovery.enabled
    failure_code = (
        RecoveryFailureCode.RECOVERY_FAILED
        if enabled
        else (
            RecoveryFailureCode.FIT_FAILED
            if baseline.status is RecoveryAttemptStatus.FIT_FAILED
            else RecoveryFailureCode.VALIDATION_FAILED
        )
    )
    return GaussianRecoveryOutcome(
        None,
        RecoveryRecord(
            RecoveryStatus.FAILED if enabled else RecoveryStatus.DISABLED,
            recovery.policy_version,
            original_roi,
            None,
            attempts,
            RecoverySelectionReason.RECOVERY_FAILED
            if enabled
            else RecoverySelectionReason.RECOVERY_DISABLED,
        ),
        failure_code,
        "No Gaussian recovery candidate passed validation"
        if enabled
        else (
            "Gaussian optimizer returned no fit"
            if failure_code is RecoveryFailureCode.FIT_FAILED
            else "Gaussian fit did not pass physical validation"
        ),
        baseline.rejection_reasons,
    )


def _run_fallbacks(
    image: np.ndarray,
    original_roi: Rect,
    fitter: GaussianFitterConfig,
    validation: FitValidation,
    is_cancelled: Callable[[], bool],
) -> tuple[
    tuple[RecoveryAttempt, ...],
    tuple[tuple[Mapping[str, Any], RecoveryAttempt], ...],
]:
    attempts = []
    candidates = []
    for index, (offset_x, offset_y, expand_by) in enumerate(
        _FALLBACK_TRANSFORMS,
        start=1,
    ):
        if is_cancelled():
            raise GaussianRecoveryCancelled("Gaussian fit recovery was cancelled")
        effective_roi = _transformed_roi(
            original_roi,
            offset_x,
            offset_y,
            expand_by,
        )
        raw_result, attempt = _run_attempt(
            image,
            original_roi,
            effective_roi,
            fitter,
            validation,
            index,
            offset_x,
            offset_y,
            expand_by,
            enforce_original_margin=True,
        )
        attempts.append(attempt)
        if raw_result is not None and attempt.status is RecoveryAttemptStatus.ACCEPTED:
            candidates.append((raw_result, attempt))
        if is_cancelled():
            raise GaussianRecoveryCancelled("Gaussian fit recovery was cancelled")
    return tuple(attempts), tuple(candidates)


def _transformed_roi(
    original: Rect,
    offset_x: int,
    offset_y: int,
    expand_by: int,
) -> Rect:
    return Rect(
        original.x + offset_x - expand_by,
        original.y + offset_y - expand_by,
        original.width + 2 * expand_by,
        original.height + 2 * expand_by,
    )


def _select_candidate(
    candidates: tuple[tuple[Mapping[str, Any], RecoveryAttempt], ...],
) -> tuple[Mapping[str, Any], RecoveryAttempt]:
    return min(
        candidates,
        key=lambda item: (
            max(
                abs(item[1].offset_x),
                abs(item[1].offset_y),
                item[1].expand_by,
            ),
            -(item[1].r_squared or 0.0),
            item[1].index,
        ),
    )


def _candidates_disagree(
    candidates: tuple[tuple[Mapping[str, Any], RecoveryAttempt], ...],
) -> bool:
    positions = tuple(
        (attempt.global_center_x, attempt.global_center_y)
        for _result, attempt in candidates
    )
    for index, (first_x, first_y) in enumerate(positions):
        if first_x is None or first_y is None:
            continue
        for second_x, second_y in positions[index + 1 :]:
            if second_x is None or second_y is None:
                continue
            if np.hypot(first_x - second_x, first_y - second_y) > 1.0:
                return True
    return False


def _run_attempt(
    image: np.ndarray,
    original_roi: Rect,
    effective_roi: Rect,
    fitter: GaussianFitterConfig,
    validation: FitValidation,
    index: int,
    offset_x: int,
    offset_y: int,
    expand_by: int,
    *,
    enforce_original_margin: bool,
) -> tuple[Mapping[str, Any] | None, RecoveryAttempt]:
    if not _roi_is_inside_image(effective_roi, image.shape):
        return None, RecoveryAttempt(
            index,
            offset_x,
            offset_y,
            expand_by,
            effective_roi,
            RecoveryAttemptStatus.OUT_OF_BOUNDS,
        )
    roi_pixels = image[
        effective_roi.y : effective_roi.y + effective_roi.height,
        effective_roi.x : effective_roi.x + effective_roi.width,
    ]
    raw_result = fit_spot_to_gaussian(
        roi_pixels,
        center_preference_factor=fitter.center_preference_factor,
        theta_bounds=fitter.theta_bounds,
    )
    if raw_result is None:
        return None, RecoveryAttempt(
            index,
            offset_x,
            offset_y,
            expand_by,
            effective_roi,
            RecoveryAttemptStatus.FIT_FAILED,
        )
    global_x = effective_roi.x + float(raw_result["center_x"])
    global_y = effective_roi.y + float(raw_result["center_y"])
    reasons = _validation_failures(raw_result, roi_pixels.shape, validation)
    status = RecoveryAttemptStatus.VALIDATION_FAILED
    if not reasons and enforce_original_margin:
        status = (
            RecoveryAttemptStatus.ACCEPTED
            if _inside_original_margin(global_x, global_y, original_roi)
            else RecoveryAttemptStatus.OUTSIDE_ORIGINAL_MARGIN
        )
    elif not reasons:
        status = RecoveryAttemptStatus.ACCEPTED
    return raw_result, RecoveryAttempt(
        index=index,
        offset_x=offset_x,
        offset_y=offset_y,
        expand_by=expand_by,
        effective_roi=effective_roi,
        status=status,
        r_squared=float(raw_result.get("r_squared", 0.0)),
        global_center_x=global_x,
        global_center_y=global_y,
        rejection_reasons=(
            reasons
            if reasons
            else (
                ()
                if status is RecoveryAttemptStatus.ACCEPTED
                else ("original_roi_margin",)
            )
        ),
    )


def _roi_is_inside_image(roi: Rect, image_shape: tuple[int, ...]) -> bool:
    height, width = image_shape[:2]
    return (
        roi.width > 0
        and roi.height > 0
        and roi.x >= 0
        and roi.y >= 0
        and roi.x + roi.width <= width
        and roi.y + roi.height <= height
    )


def _inside_original_margin(
    global_x: float,
    global_y: float,
    original: Rect,
) -> bool:
    margin = 2.0
    return (
        original.x - margin < global_x < original.x + original.width + margin
        and original.y - margin < global_y < original.y + original.height + margin
    )


def _validation_failures(
    result: Mapping[str, Any],
    roi_shape: tuple[int, ...],
    config: FitValidation,
) -> tuple[str, ...]:
    height, width = roi_shape[:2]
    checks = (
        (float(result.get("r_squared", 0.0)) < config.min_r_squared, "r_squared"),
        (float(result.get("amplitude", 0.0)) < config.min_amplitude, "amplitude"),
        (
            not -config.center_margin
            < float(result.get("center_x", 0.0))
            < width + config.center_margin,
            "center_x",
        ),
        (
            not -config.center_margin
            < float(result.get("center_y", 0.0))
            < height + config.center_margin,
            "center_y",
        ),
        (not config.min_sigma < float(result.get("sigma_x", 0.0)) < width, "sigma_x"),
        (not config.min_sigma < float(result.get("sigma_y", 0.0)) < height, "sigma_y"),
    )
    return tuple(name for failed, name in checks if failed)
