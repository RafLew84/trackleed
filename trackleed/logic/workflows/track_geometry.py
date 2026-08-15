"""Headless pattern geometry derived from an immutable Gaussian ``FitSet``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json

import numpy as np

from trackleed.analysis.pattern_geometry import (
    ManualEllipse,
    ellipse_corrected_radius,
    euclidean_radius,
    deterministic_ransac_center,
    point_set_diagnostics,
    smooth_center_series,
    summarize_radii,
)
from trackleed.analysis.peak_analyzer import fit_circle_center, fit_ellipse_center
from trackleed.logic.workflows.track_analysis import FitSet


class GeometryMethod(str, Enum):
    PER_FRAME = "per_frame"
    GLOBAL_MEDIAN = "global_median"
    CIRCLE_FIT = "circle_fit"
    CIRCLE_FIT_TEMPORAL = "circle_fit_temporal"
    CIRCLE_FIT_BBOX_FALLBACK = "circle_fit_bbox_fallback"
    MANUAL_ELLIPSE = "manual_ellipse"
    ELLIPSE_FIT = "ellipse_fit"


class GeometryStatus(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    GEOMETRY_UNAVAILABLE = "geometry_unavailable"


class GeometryUnavailableReason(str, Enum):
    NO_REFERENCE_TRACKS = "no_reference_tracks"
    INSUFFICIENT_POINTS = "insufficient_points"
    COLLINEAR_POINTS = "collinear_points"
    NUMERICALLY_UNRELIABLE = "numerically_unreliable"
    MANUAL_ELLIPSE_REQUIRED = "manual_ellipse_required"


@dataclass(frozen=True)
class GeometryParameters:
    ransac_iterations: int = 100
    ransac_threshold: float = 15.0
    random_seed: int = 20260811
    temporal_window: int = 5
    temporal_alpha: float = 0.25
    max_condition_number: float = 1.0e12


@dataclass(frozen=True)
class GeometryFingerprint:
    sha256: str

    def __post_init__(self) -> None:
        is_hex = all(
            character in "0123456789abcdef" for character in self.sha256.lower()
        )
        if len(self.sha256) != 64 or not is_hex:
            raise ValueError("GeometryFingerprint must contain a SHA-256 hex digest")


@dataclass(frozen=True)
class GeometryRequest:
    method: GeometryMethod = GeometryMethod.CIRCLE_FIT
    center_reference_track_ids: tuple[int, ...] = ()
    parameters: GeometryParameters = field(default_factory=GeometryParameters)
    manual_ellipse: ManualEllipse | None = None
    previous: GeometryResult | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class GeometryProgress:
    completed: int
    total: int
    frame_id: int


def _ignore_progress(_progress: GeometryProgress) -> None:
    return None


def _never_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class GeometryRunContext:
    report_progress: Callable[[GeometryProgress], None] = field(
        default=_ignore_progress,
        compare=False,
        repr=False,
    )
    is_cancelled: Callable[[], bool] = field(
        default=_never_cancelled,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class GeometryFailure:
    reason: GeometryUnavailableReason
    message: str


@dataclass(frozen=True)
class PatternCenter:
    center_x: float
    center_y: float
    error_x: float
    error_y: float
    method_used: GeometryMethod
    circle_radius: float | None = None
    radius_x: float | None = None
    radius_y: float | None = None
    angle_degrees: float | None = None
    radius_scale: float | None = None


@dataclass(frozen=True)
class ObservationGeometry:
    track_id: int
    frame_id: int
    radius: float
    radius_error: float | None


@dataclass(frozen=True)
class FrameGeometry:
    frame_id: int
    status: GeometryStatus
    center: PatternCenter | None
    average_radius: float | None = None
    average_radius_error: float | None = None
    failure: GeometryFailure | None = None


@dataclass(frozen=True)
class GeometryResult:
    fingerprint: GeometryFingerprint
    status: GeometryStatus
    requested_method: GeometryMethod
    center_reference_track_ids: tuple[int, ...]
    observations: tuple[ObservationGeometry, ...]
    frames: tuple[FrameGeometry, ...]
    failure: GeometryFailure | None = None


class GeometryCancelled(RuntimeError):
    """Raised instead of exposing partial pattern geometry."""


class GeometryPreflightError(ValueError):
    """Raised when a geometry request is internally inconsistent."""


def derive_geometry(
    fit_set: FitSet,
    request: GeometryRequest,
    context: GeometryRunContext | None = None,
) -> GeometryResult:
    """Derive immutable centers and radii without reading or changing fit inputs."""
    context = context or GeometryRunContext()
    _validate_request(request)
    if context.is_cancelled():
        raise GeometryCancelled("Pattern geometry was cancelled")
    fingerprint = _geometry_fingerprint(fit_set, request)
    if request.previous is not None and request.previous.fingerprint == fingerprint:
        return request.previous
    centers: dict[int, PatternCenter] = {}
    failures: dict[int, GeometryFailure] = {}
    frame_ids = _frame_ids(fit_set)
    if request.method is GeometryMethod.MANUAL_ELLIPSE:
        centers, failures = _manual_centers(frame_ids, request)
        _report_frames(frame_ids, context)
    else:
        for index, frame_id in enumerate(frame_ids):
            if context.is_cancelled():
                raise GeometryCancelled("Pattern geometry was cancelled")
            positions, used_bbox = _reference_positions(
                fit_set,
                request,
                frame_id,
            )
            raw_center, failure = _estimate_center(
                positions,
                request,
                frame_id,
            )
            if raw_center is not None:
                method_used = (
                    GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK
                    if used_bbox
                    else _base_method(request.method)
                )
                centers[frame_id] = _freeze_center(raw_center, method_used)
            elif failure is not None:
                failures[frame_id] = failure
            context.report_progress(
                GeometryProgress(index + 1, len(frame_ids), frame_id)
            )
        centers, failures = _apply_cross_frame_method(
            frame_ids,
            centers,
            failures,
            request,
        )
    observations = _observation_geometry(fit_set, centers, request)
    frames = _frame_geometry(fit_set, centers, failures, observations)
    status = _overall_status(frames)
    if context.is_cancelled():
        raise GeometryCancelled("Pattern geometry was cancelled")
    result_failure = None
    if status is GeometryStatus.GEOMETRY_UNAVAILABLE:
        result_failure = next(iter(failures.values()), None) or _failure(
            GeometryUnavailableReason.NO_REFERENCE_TRACKS
            if not request.center_reference_track_ids
            else GeometryUnavailableReason.INSUFFICIENT_POINTS
        )
    return GeometryResult(
        fingerprint,
        status,
        request.method,
        request.center_reference_track_ids,
        observations,
        frames,
        result_failure,
    )


def _validate_request(request: GeometryRequest) -> None:
    if len(set(request.center_reference_track_ids)) != len(
        request.center_reference_track_ids
    ):
        raise GeometryPreflightError("Center-reference track IDs must be unique")
    parameters = request.parameters
    if parameters.ransac_iterations < 1:
        raise GeometryPreflightError("RANSAC iterations must be positive")
    if (
        not np.isfinite(parameters.ransac_threshold)
        or parameters.ransac_threshold <= 0.0
    ):
        raise GeometryPreflightError("RANSAC threshold must be positive and finite")
    if parameters.temporal_window < 1 or parameters.temporal_window % 2 == 0:
        raise GeometryPreflightError("Temporal window must be positive and odd")
    if (
        not np.isfinite(parameters.temporal_alpha)
        or not 0.0 < parameters.temporal_alpha <= 1.0
    ):
        raise GeometryPreflightError("Temporal alpha must be in (0, 1]")
    if (
        not np.isfinite(parameters.max_condition_number)
        or parameters.max_condition_number < 1.0
    ):
        raise GeometryPreflightError(
            "Maximum condition number must be finite and at least one"
        )


def _geometry_fingerprint(
    fit_set: FitSet,
    request: GeometryRequest,
) -> GeometryFingerprint:
    canonical = {
        "fit_set": [
            {
                "track_id": item.track_id,
                "frame_id": item.frame_id,
                "fingerprint": item.fingerprint.sha256,
                "status": item.status.value,
                "center": None
                if item.result is None
                else [item.result.global_center_x, item.result.global_center_y],
                "center_error": None
                if item.result is None
                else [item.result.errors.center_x, item.result.errors.center_y],
            }
            for item in fit_set.observations
        ],
        "method": request.method.value,
        "parameters": request.parameters.__dict__,
        "center_reference_track_ids": request.center_reference_track_ids,
        "manual_ellipse": (
            None if request.manual_ellipse is None else request.manual_ellipse.__dict__
        ),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return GeometryFingerprint(sha256(payload.encode()).hexdigest())


def _frame_ids(fit_set: FitSet) -> tuple[int, ...]:
    return tuple(sorted({item.frame_id for item in fit_set.observations}))


def _reference_positions(
    fit_set: FitSet,
    request: GeometryRequest,
    frame_id: int,
) -> tuple[list[dict[str, float]], bool]:
    reference_tracks = set(request.center_reference_track_ids)
    positions = []
    used_bbox = False
    for item in fit_set.observations:
        if item.frame_id != frame_id or item.track_id not in reference_tracks:
            continue
        if item.result is not None:
            positions.append(
                {
                    "center_x": item.result.global_center_x,
                    "center_y": item.result.global_center_y,
                }
            )
        elif request.method is GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK:
            roi = item.provenance.original_roi
            positions.append(
                {
                    "center_x": roi.x + roi.width / 2.0,
                    "center_y": roi.y + roi.height / 2.0,
                }
            )
            used_bbox = True
    return positions, used_bbox


def _freeze_center(
    values: dict[str, float],
    method: GeometryMethod,
) -> PatternCenter:
    return PatternCenter(
        center_x=float(values["center_x"]),
        center_y=float(values["center_y"]),
        error_x=float(values.get("error_x", 0.0)),
        error_y=float(values.get("error_y", 0.0)),
        method_used=method,
        circle_radius=(
            float(values["circle_radius"]) if "circle_radius" in values else None
        ),
        radius_x=float(values["radius_x"]) if "radius_x" in values else None,
        radius_y=float(values["radius_y"]) if "radius_y" in values else None,
        angle_degrees=(float(values["angle_deg"]) if "angle_deg" in values else None),
        radius_scale=(
            float(values["radius_scale_px"]) if "radius_scale_px" in values else None
        ),
    )


def _estimate_center(
    positions: list[dict[str, float]],
    request: GeometryRequest,
    frame_id: int,
) -> tuple[dict[str, float] | None, GeometryFailure | None]:
    if not request.center_reference_track_ids:
        return None, _failure(GeometryUnavailableReason.NO_REFERENCE_TRACKS)
    minimum = 4 if _base_method(request.method) is GeometryMethod.PER_FRAME else 3
    if len(positions) < minimum:
        return None, _failure(GeometryUnavailableReason.INSUFFICIENT_POINTS)
    diagnostics = point_set_diagnostics(positions)
    if diagnostics is None:
        return None, _failure(GeometryUnavailableReason.NUMERICALLY_UNRELIABLE)
    rank, condition = diagnostics
    if rank < 2:
        return None, _failure(GeometryUnavailableReason.COLLINEAR_POINTS)
    if (
        not np.isfinite(condition)
        or condition > request.parameters.max_condition_number
    ):
        return None, _failure(GeometryUnavailableReason.NUMERICALLY_UNRELIABLE)
    method = _base_method(request.method)
    try:
        if method is GeometryMethod.PER_FRAME:
            center = deterministic_ransac_center(
                positions,
                iterations=request.parameters.ransac_iterations,
                threshold=request.parameters.ransac_threshold,
                seed=request.parameters.random_seed + frame_id,
            )
        elif method is GeometryMethod.ELLIPSE_FIT:
            center = fit_ellipse_center(positions)
        else:
            center = fit_circle_center(positions)
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        return None, _failure(GeometryUnavailableReason.NUMERICALLY_UNRELIABLE)
    if center is None or not all(np.isfinite(value) for value in center.values()):
        return None, _failure(GeometryUnavailableReason.NUMERICALLY_UNRELIABLE)
    return center, None


def _base_method(method: GeometryMethod) -> GeometryMethod:
    if method is GeometryMethod.GLOBAL_MEDIAN:
        return GeometryMethod.PER_FRAME
    if method in {
        GeometryMethod.CIRCLE_FIT_TEMPORAL,
        GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK,
    }:
        return GeometryMethod.CIRCLE_FIT
    return method


def _manual_centers(
    frame_ids: tuple[int, ...],
    request: GeometryRequest,
) -> tuple[dict[int, PatternCenter], dict[int, GeometryFailure]]:
    ellipse = request.manual_ellipse
    valid = (
        ellipse is not None
        and all(
            np.isfinite(value)
            for value in (
                ellipse.center_x,
                ellipse.center_y,
                ellipse.radius_x,
                ellipse.radius_y,
                ellipse.angle_degrees,
            )
        )
        and ellipse.radius_x > 0.0
        and ellipse.radius_y > 0.0
    )
    if not valid:
        failure = _failure(GeometryUnavailableReason.MANUAL_ELLIPSE_REQUIRED)
        return {}, {frame_id: failure for frame_id in frame_ids}
    assert ellipse is not None
    scale = float(np.sqrt(ellipse.radius_x * ellipse.radius_y))
    center = PatternCenter(
        ellipse.center_x,
        ellipse.center_y,
        0.0,
        0.0,
        GeometryMethod.MANUAL_ELLIPSE,
        radius_x=ellipse.radius_x,
        radius_y=ellipse.radius_y,
        angle_degrees=ellipse.angle_degrees,
        radius_scale=scale,
    )
    return {frame_id: center for frame_id in frame_ids}, {}


def _apply_cross_frame_method(
    frame_ids: tuple[int, ...],
    centers: dict[int, PatternCenter],
    failures: dict[int, GeometryFailure],
    request: GeometryRequest,
) -> tuple[dict[int, PatternCenter], dict[int, GeometryFailure]]:
    if not centers:
        return centers, failures
    if request.method is GeometryMethod.GLOBAL_MEDIAN:
        center = PatternCenter(
            center_x=float(np.median([item.center_x for item in centers.values()])),
            center_y=float(np.median([item.center_y for item in centers.values()])),
            error_x=float(np.median([item.error_x for item in centers.values()])),
            error_y=float(np.median([item.error_y for item in centers.values()])),
            method_used=GeometryMethod.GLOBAL_MEDIAN,
        )
        return {frame_id: center for frame_id in frame_ids}, {}
    if request.method is GeometryMethod.CIRCLE_FIT_TEMPORAL:
        return _smooth_centers(frame_ids, centers, request), {}
    return centers, failures


def _smooth_centers(
    frame_ids: tuple[int, ...],
    centers: dict[int, PatternCenter],
    request: GeometryRequest,
) -> dict[int, PatternCenter]:
    smoothed = smooth_center_series(
        frame_ids,
        {
            frame_id: {
                "center_x": center.center_x,
                "center_y": center.center_y,
                "error_x": center.error_x,
                "error_y": center.error_y,
                **(
                    {"circle_radius": center.circle_radius}
                    if center.circle_radius is not None
                    else {}
                ),
            }
            for frame_id, center in centers.items()
        },
        window=request.parameters.temporal_window,
        alpha=request.parameters.temporal_alpha,
    )
    return {
        frame_id: PatternCenter(
            values["center_x"],
            values["center_y"],
            values["error_x"],
            values["error_y"],
            GeometryMethod.CIRCLE_FIT_TEMPORAL,
            circle_radius=values.get("circle_radius"),
        )
        for frame_id, values in smoothed.items()
    }


def _failure(reason: GeometryUnavailableReason) -> GeometryFailure:
    messages = {
        GeometryUnavailableReason.NO_REFERENCE_TRACKS: (
            "No center-reference tracks were declared"
        ),
        GeometryUnavailableReason.INSUFFICIENT_POINTS: (
            "Too few fitted reference observations determine the pattern geometry"
        ),
        GeometryUnavailableReason.COLLINEAR_POINTS: (
            "Reference observations are collinear"
        ),
        GeometryUnavailableReason.NUMERICALLY_UNRELIABLE: (
            "Pattern geometry is numerically unreliable"
        ),
        GeometryUnavailableReason.MANUAL_ELLIPSE_REQUIRED: (
            "Manual ellipse geometry requires a valid ellipse"
        ),
    }
    return GeometryFailure(reason, messages[reason])


def _observation_geometry(
    fit_set: FitSet,
    centers: dict[int, PatternCenter],
    request: GeometryRequest,
) -> tuple[ObservationGeometry, ...]:
    observations = []
    for item in fit_set.observations:
        if item.result is None or item.frame_id not in centers:
            continue
        center = centers[item.frame_id]
        center_values = {
            "center_x": center.center_x,
            "center_y": center.center_y,
            "error_x": center.error_x,
            "error_y": center.error_y,
        }
        radius: float
        radius_error: float | None
        if center.radius_x is not None and center.radius_y is not None:
            radius, radius_error = ellipse_corrected_radius(
                item.result.global_center_x,
                item.result.global_center_y,
                item.result.errors.center_x,
                item.result.errors.center_y,
                center_values,
                ManualEllipse(
                    center.center_x,
                    center.center_y,
                    center.radius_x,
                    center.radius_y,
                    center.angle_degrees or 0.0,
                ),
                center.radius_scale,
            )
        else:
            radius, radius_error = euclidean_radius(
                item.result.global_center_x,
                item.result.global_center_y,
                item.result.errors.center_x,
                item.result.errors.center_y,
                center_values,
            )
        observations.append(
            ObservationGeometry(
                item.track_id,
                item.frame_id,
                radius,
                radius_error,
            )
        )
    return tuple(observations)


def _frame_geometry(
    fit_set: FitSet,
    centers: dict[int, PatternCenter],
    failures: dict[int, GeometryFailure],
    observations: tuple[ObservationGeometry, ...],
) -> tuple[FrameGeometry, ...]:
    frames = []
    for frame_id in _frame_ids(fit_set):
        radii = [item.radius for item in observations if item.frame_id == frame_id]
        average, error = summarize_radii(radii)
        center = centers.get(frame_id)
        frames.append(
            FrameGeometry(
                frame_id,
                GeometryStatus.AVAILABLE
                if center is not None
                else GeometryStatus.GEOMETRY_UNAVAILABLE,
                center,
                average,
                error,
                failures.get(frame_id),
            )
        )
    return tuple(frames)


def _overall_status(frames: tuple[FrameGeometry, ...]) -> GeometryStatus:
    available = sum(frame.status is GeometryStatus.AVAILABLE for frame in frames)
    if available == len(frames) and frames:
        return GeometryStatus.AVAILABLE
    if available:
        return GeometryStatus.PARTIALLY_AVAILABLE
    return GeometryStatus.GEOMETRY_UNAVAILABLE


def _report_frames(
    frame_ids: tuple[int, ...],
    context: GeometryRunContext,
) -> None:
    for index, frame_id in enumerate(frame_ids):
        if context.is_cancelled():
            raise GeometryCancelled("Pattern geometry was cancelled")
        context.report_progress(GeometryProgress(index + 1, len(frame_ids), frame_id))
