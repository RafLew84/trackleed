"""Pure legacy geometry seam consuming an immutable :class:`FitSet`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trackleed.analysis.peak_analyzer import (
    find_pattern_center,
    fit_circle_center,
    fit_ellipse_center,
)
from trackleed.analysis.pattern_geometry import (
    ManualEllipse,
    ellipse_corrected_radius,
    euclidean_radius,
    smooth_center_series,
    summarize_radii,
)
from trackleed.logic.workflows.track_analysis import FitSet, GaussianFitResult, Rect


class LegacyCenterMode(str, Enum):
    PER_FRAME = "per_frame"
    GLOBAL_MEDIAN = "global_median"
    CIRCLE_FIT = "circle_fit"
    CIRCLE_FIT_TEMPORAL = "circle_fit_temporal"
    CIRCLE_FIT_BBOX_FALLBACK = "circle_fit_bbox_fallback"
    MANUAL_ELLIPSE = "manual_ellipse"
    ELLIPSE_FIT = "ellipse_fit"


@dataclass(frozen=True)
class GeometryObservationRoi:
    track_id: int
    frame_id: int
    roi: Rect


@dataclass(frozen=True)
class LegacyGeometryRequest:
    observation_rois: tuple[GeometryObservationRoi, ...]
    center_mode: LegacyCenterMode
    manual_ellipse: ManualEllipse | None = None


@dataclass(frozen=True)
class ObservationGeometry:
    track_id: int
    frame_id: int
    radius: float
    radius_error: float | None


@dataclass(frozen=True)
class FrameGeometry:
    frame_id: int
    pattern_center: tuple[tuple[str, float], ...] | None
    average_radius: float | None = None
    average_radius_error: float | None = None


@dataclass(frozen=True)
class LegacyGeometryResult:
    observations: tuple[ObservationGeometry, ...]
    frames: tuple[FrameGeometry, ...]


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


class LegacyGeometryCancelled(RuntimeError):
    """Raised instead of returning partial frame geometry."""


def manual_ellipse_from_mapping(
    values: Mapping[str, float] | None,
) -> ManualEllipse | None:
    if not values:
        return None
    try:
        ellipse = ManualEllipse(
            center_x=float(values["cx"]),
            center_y=float(values["cy"]),
            radius_x=float(values["rx"]),
            radius_y=float(values["ry"]),
            angle_degrees=float(values.get("angle_deg", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    numbers = (
        ellipse.center_x,
        ellipse.center_y,
        ellipse.radius_x,
        ellipse.radius_y,
        ellipse.angle_degrees,
    )
    if not all(np.isfinite(value) for value in numbers):
        return None
    return ellipse if ellipse.radius_x > 0.0 and ellipse.radius_y > 0.0 else None


def derive_legacy_geometry(
    fit_set: FitSet,
    request: LegacyGeometryRequest,
    context: GeometryRunContext | None = None,
) -> LegacyGeometryResult:
    """Derive current center/radius semantics without mutating ``FitSet``."""
    context = context or GeometryRunContext()
    rois = {
        (item.track_id, item.frame_id): item.roi for item in request.observation_rois
    }
    frame_ids = sorted({item.frame_id for item in request.observation_rois})
    if (
        request.center_mode is LegacyCenterMode.MANUAL_ELLIPSE
        and request.manual_ellipse is None
    ):
        raise ValueError("manual_ellipse mode requires a valid ellipse")
    centers, observations = _first_pass(fit_set, request, rois, frame_ids, context)
    if request.center_mode is LegacyCenterMode.GLOBAL_MEDIAN:
        center = _median_center(tuple(centers.values()))
        centers = {frame_id: center for frame_id in frame_ids if center is not None}
        observations = _radii_for_centers(fit_set, centers, request)
        _report_second_pass(frame_ids, context)
    elif request.center_mode is LegacyCenterMode.CIRCLE_FIT_TEMPORAL:
        centers = smooth_legacy_centers(frame_ids, centers)
        observations = _radii_for_centers(fit_set, centers, request)
        _report_second_pass(frame_ids, context)
    frames = _frame_results(frame_ids, centers, observations)
    if context.is_cancelled():
        raise LegacyGeometryCancelled("Track geometry was cancelled")
    return LegacyGeometryResult(tuple(observations), frames)


def _first_pass(
    fit_set: FitSet,
    request: LegacyGeometryRequest,
    rois: dict[tuple[int, int], Rect],
    frame_ids: list[int],
    context: GeometryRunContext,
) -> tuple[dict[int, dict[str, float]], list[ObservationGeometry]]:
    centers: dict[int, dict[str, float]] = {}
    per_frame = request.center_mode not in {
        LegacyCenterMode.GLOBAL_MEDIAN,
        LegacyCenterMode.CIRCLE_FIT_TEMPORAL,
    }
    progress_total = len(frame_ids) if per_frame else 2 * len(frame_ids)
    for index, frame_id in enumerate(frame_ids):
        if context.is_cancelled():
            raise LegacyGeometryCancelled("Track geometry was cancelled")
        positions = _positions_for_frame(fit_set, rois, frame_id, request.center_mode)
        center = _estimate_center(positions, request)
        if center is not None:
            centers[frame_id] = center
        context.report_progress(GeometryProgress(index + 1, progress_total, frame_id))
    observations = _radii_for_centers(fit_set, centers, request) if per_frame else []
    return centers, observations


def _positions_for_frame(
    fit_set: FitSet,
    rois: dict[tuple[int, int], Rect],
    frame_id: int,
    mode: LegacyCenterMode,
) -> list[dict[str, float]]:
    positions = []
    for item in fit_set.observations:
        if item.frame_id != frame_id:
            continue
        if item.result is not None:
            positions.append(
                {
                    "center_x": item.result.global_center_x,
                    "center_y": item.result.global_center_y,
                }
            )
        elif mode is LegacyCenterMode.CIRCLE_FIT_BBOX_FALLBACK:
            roi = rois.get((item.track_id, item.frame_id))
            if roi is not None:
                positions.append(
                    {
                        "center_x": roi.x + roi.width / 2.0,
                        "center_y": roi.y + roi.height / 2.0,
                    }
                )
    return positions


def _estimate_center(
    positions: list[dict[str, float]],
    request: LegacyGeometryRequest,
) -> dict[str, float] | None:
    mode = request.center_mode
    if mode is LegacyCenterMode.MANUAL_ELLIPSE:
        ellipse = request.manual_ellipse
        assert ellipse is not None
        return {
            "center_x": ellipse.center_x,
            "center_y": ellipse.center_y,
            "error_x": 0.0,
            "error_y": 0.0,
        }
    return estimate_legacy_center(positions, mode)


def estimate_legacy_center(
    positions: list[dict[str, float]],
    mode: LegacyCenterMode,
) -> dict[str, float] | None:
    """Calculate a center using the current Track Analysis semantics."""
    if not positions:
        return None
    if mode is LegacyCenterMode.ELLIPSE_FIT:
        return fit_ellipse_center(positions)
    if mode in {
        LegacyCenterMode.CIRCLE_FIT,
        LegacyCenterMode.CIRCLE_FIT_TEMPORAL,
        LegacyCenterMode.CIRCLE_FIT_BBOX_FALLBACK,
    }:
        return fit_circle_center(positions)
    return find_pattern_center(positions)


def _radii_for_centers(
    fit_set: FitSet,
    centers: dict[int, dict[str, float]],
    request: LegacyGeometryRequest,
) -> list[ObservationGeometry]:
    observations = []
    for item in fit_set.observations:
        center = centers.get(item.frame_id)
        if item.result is None or center is None:
            continue
        ellipse = None
        if request.center_mode is LegacyCenterMode.MANUAL_ELLIPSE:
            ellipse = request.manual_ellipse
        elif request.center_mode is LegacyCenterMode.ELLIPSE_FIT:
            ellipse = _ellipse_from_center(center)
        radius, radius_error = _radius(item.result, center, ellipse)
        observations.append(
            ObservationGeometry(
                item.track_id,
                item.frame_id,
                radius,
                radius_error,
            )
        )
    return observations


def _ellipse_from_center(center: dict[str, float]) -> ManualEllipse | None:
    required = {"center_x", "center_y", "radius_x", "radius_y"}
    if not required.issubset(center):
        return None
    return ManualEllipse(
        center["center_x"],
        center["center_y"],
        center["radius_x"],
        center["radius_y"],
        center.get("angle_deg", 0.0),
    )


def _radius(
    result: GaussianFitResult,
    center: dict[str, float],
    ellipse: ManualEllipse | None,
) -> tuple[float, float | None]:
    if ellipse is not None:
        return ellipse_corrected_radius(
            result.global_center_x,
            result.global_center_y,
            result.errors.center_x,
            result.errors.center_y,
            center,
            ellipse,
        )
    return euclidean_radius(
        result.global_center_x,
        result.global_center_y,
        result.errors.center_x,
        result.errors.center_y,
        center,
    )


def _frame_results(
    frame_ids: list[int],
    centers: dict[int, dict[str, float]],
    observations: list[ObservationGeometry],
) -> tuple[FrameGeometry, ...]:
    frames = []
    for frame_id in frame_ids:
        radii = [item.radius for item in observations if item.frame_id == frame_id]
        average, error = summarize_radii(radii)
        center = centers.get(frame_id)
        frozen_center = None if center is None else tuple(center.items())
        frames.append(FrameGeometry(frame_id, frozen_center, average, error))
    return tuple(frames)


def _median_center(
    centers: tuple[dict[str, float], ...],
) -> dict[str, float] | None:
    if not centers:
        return None
    return {
        "center_x": float(np.median([item["center_x"] for item in centers])),
        "center_y": float(np.median([item["center_y"] for item in centers])),
        "error_x": float(np.median([item.get("error_x", 0.0) for item in centers])),
        "error_y": float(np.median([item.get("error_y", 0.0) for item in centers])),
    }


def _report_second_pass(
    frame_ids: list[int],
    context: GeometryRunContext,
) -> None:
    total = 2 * len(frame_ids)
    for index, frame_id in enumerate(frame_ids):
        if context.is_cancelled():
            raise LegacyGeometryCancelled("Track geometry was cancelled")
        context.report_progress(
            GeometryProgress(len(frame_ids) + index + 1, total, frame_id)
        )


def smooth_legacy_centers(
    frame_ids: list[int],
    raw_centers: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Stabilize the current circle-fit center sequence over time."""
    return smooth_center_series(
        frame_ids,
        raw_centers,
        window=5,
        alpha=0.25,
    )
