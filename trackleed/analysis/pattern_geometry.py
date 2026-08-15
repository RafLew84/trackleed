"""Deterministic numerical kernels for pattern-center geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import random

import numpy as np


@dataclass(frozen=True)
class ManualEllipse:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle_degrees: float = 0.0


def euclidean_radius(
    global_center_x: float,
    global_center_y: float,
    spot_error_x: float,
    spot_error_y: float,
    center: Mapping[str, float],
) -> tuple[float, float | None]:
    """Return ordinary spot radius and its propagated uncertainty."""
    dx = global_center_x - center["center_x"]
    dy = global_center_y - center["center_y"]
    radius = float(np.sqrt(dx**2 + dy**2))
    if radius <= 1e-6:
        return radius, None
    var_x = spot_error_x**2 + center.get("error_x", 0.0) ** 2
    var_y = spot_error_y**2 + center.get("error_y", 0.0) ** 2
    error = float(np.sqrt(dx**2 * var_x + dy**2 * var_y) / radius)
    return radius, error


def ellipse_corrected_radius(
    global_center_x: float,
    global_center_y: float,
    spot_error_x: float,
    spot_error_y: float,
    center: Mapping[str, float],
    ellipse: ManualEllipse,
    radius_scale: float | None = None,
) -> tuple[float, float]:
    """Return the current affine ellipse-corrected radius and uncertainty."""
    dx = global_center_x - center["center_x"]
    dy = global_center_y - center["center_y"]
    angle = np.deg2rad(ellipse.angle_degrees)
    cos_angle, sin_angle = float(np.cos(angle)), float(np.sin(angle))
    xr = cos_angle * dx + sin_angle * dy
    yr = -sin_angle * dx + cos_angle * dy
    inv_x, inv_y = 1.0 / ellipse.radius_x, 1.0 / ellipse.radius_y
    u, v = xr * inv_x, yr * inv_y
    normalized = float(np.sqrt(u**2 + v**2))
    scale = (
        float(np.sqrt(ellipse.radius_x * ellipse.radius_y))
        if radius_scale is None
        else float(radius_scale)
    )
    radius = scale * normalized
    if normalized <= 1e-9:
        return radius, 0.0
    dqx = (u * cos_angle * inv_x - v * sin_angle * inv_y) / normalized
    dqy = (u * sin_angle * inv_x + v * cos_angle * inv_y) / normalized
    var_x = spot_error_x**2 + center.get("error_x", 0.0) ** 2
    var_y = spot_error_y**2 + center.get("error_y", 0.0) ** 2
    error = scale * np.sqrt(dqx**2 * var_x + dqy**2 * var_y)
    return float(radius), float(error)


def summarize_radii(
    radii: Sequence[float],
) -> tuple[float | None, float | None]:
    """Return the per-frame mean and standard error for multiple radii."""
    if len(radii) <= 1:
        return None, None
    return (
        float(np.mean(radii)),
        float(np.std(radii) / np.sqrt(len(radii))),
    )


def point_set_diagnostics(
    positions: Sequence[Mapping[str, float]],
) -> tuple[int, float] | None:
    """Return centered rank/condition, or ``None`` for non-finite input."""
    points = np.asarray(
        [[item["center_x"], item["center_y"]] for item in positions],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(points)):
        return None
    centered = points - np.mean(points, axis=0)
    design = np.column_stack((centered, np.ones(len(centered))))
    try:
        return int(np.linalg.matrix_rank(centered)), float(np.linalg.cond(design))
    except np.linalg.LinAlgError:
        return None


def deterministic_ransac_center(
    positions: Sequence[Mapping[str, float]],
    *,
    iterations: int,
    threshold: float,
    seed: int,
) -> dict[str, float] | None:
    """Estimate a centrosymmetric center without ambient random state."""
    points = np.asarray(
        [[item["center_x"], item["center_y"]] for item in positions],
        dtype=np.float64,
    )
    hypotheses = list(combinations(range(len(points)), 2))
    random.Random(seed).shuffle(hypotheses)
    best_midpoints: list[np.ndarray] = []
    best_residual = float("inf")
    for first, second in hypotheses[:iterations]:
        candidate = (points[first] + points[second]) / 2.0
        midpoints, residual = _symmetric_midpoints(points, candidate, threshold)
        if len(midpoints) > len(best_midpoints) or (
            len(midpoints) == len(best_midpoints) and residual < best_residual
        ):
            best_midpoints = midpoints
            best_residual = residual
    if len(best_midpoints) < 2:
        return None
    values = np.asarray(best_midpoints, dtype=np.float64)
    center = np.mean(values, axis=0)
    errors = np.std(values, axis=0)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "error_x": float(errors[0]),
        "error_y": float(errors[1]),
    }


def smooth_center_series(
    frame_ids: Sequence[int],
    centers: Mapping[int, Mapping[str, float]],
    *,
    window: int,
    alpha: float,
) -> dict[int, dict[str, float]]:
    """Interpolate and smooth centers using current Track Analysis semantics."""
    if not frame_ids or not centers:
        return {}
    values = _center_series(frame_ids, centers)
    finite_centers = np.isfinite(values["center_x"]) & np.isfinite(values["center_y"])
    if not np.any(finite_centers):
        return {}
    raw_x = _interpolate_nans(values["center_x"])
    raw_y = _interpolate_nans(values["center_y"])
    center_x = _ema(_rolling_median(raw_x, window), alpha)
    center_y = _ema(_rolling_median(raw_y, window), alpha)
    _clamp_center_steps(center_x, center_y, raw_x, raw_y)
    radius = _ema(
        _rolling_median(_interpolate_nans(values["circle_radius"]), window),
        alpha,
    )
    return _smoothed_center_records(
        frame_ids,
        center_x,
        center_y,
        values["center_x"],
        values["center_y"],
        values["error_x"],
        values["error_y"],
        radius,
    )


def _symmetric_midpoints(
    points: np.ndarray,
    candidate: np.ndarray,
    threshold: float,
) -> tuple[list[np.ndarray], float]:
    remaining = set(range(len(points)))
    midpoints = []
    residual = 0.0
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        if not remaining:
            break
        reflected = 2.0 * candidate - points[first]
        second = min(
            remaining,
            key=lambda index: float(np.linalg.norm(points[index] - reflected)),
        )
        distance = float(np.linalg.norm(points[second] - reflected))
        if distance < threshold:
            remaining.remove(second)
            midpoints.append((points[first] + points[second]) / 2.0)
            residual += distance
    return midpoints, residual


def _center_series(
    frame_ids: Sequence[int],
    centers: Mapping[int, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    names = ("center_x", "center_y", "error_x", "error_y", "circle_radius")
    return {
        name: np.asarray(
            [centers.get(frame_id, {}).get(name, np.nan) for frame_id in frame_ids],
            dtype=np.float64,
        )
        for name in names
    }


def _clamp_center_steps(
    center_x: np.ndarray,
    center_y: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
) -> None:
    steps = np.hypot(np.diff(raw_x), np.diff(raw_y))
    median = _nan_median_or_default(steps, 2.0)
    deviation = _nan_median_or_default(np.abs(steps - median), 0.5)
    max_step = float(min(25.0, max(1.5, median + 4.0 * deviation + 2.0)))
    for index in range(1, center_x.size):
        dx = center_x[index] - center_x[index - 1]
        dy = center_y[index] - center_y[index - 1]
        distance = float(np.hypot(dx, dy))
        if distance > max_step and distance > 1e-9:
            scale = max_step / distance
            center_x[index] = center_x[index - 1] + dx * scale
            center_y[index] = center_y[index - 1] + dy * scale


def _nan_median_or_default(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float(default)


def _smoothed_center_records(
    frame_ids: Sequence[int],
    center_x: np.ndarray,
    center_y: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    error_x: np.ndarray,
    error_y: np.ndarray,
    radius: np.ndarray,
) -> dict[int, dict[str, float]]:
    records = {}
    default_error_x = _nan_median_or_default(error_x, 1.0)
    default_error_y = _nan_median_or_default(error_y, 1.0)
    for index, frame_id in enumerate(frame_ids):
        resolved_error_x = (
            float(error_x[index]) if np.isfinite(error_x[index]) else default_error_x
        )
        resolved_error_y = (
            float(error_y[index]) if np.isfinite(error_y[index]) else default_error_y
        )
        if np.isfinite(raw_x[index]) and np.isfinite(raw_y[index]):
            resolved_error_x = max(
                resolved_error_x,
                abs(float(raw_x[index] - center_x[index])),
            )
            resolved_error_y = max(
                resolved_error_y,
                abs(float(raw_y[index] - center_y[index])),
            )
        record = {
            "center_x": float(center_x[index]),
            "center_y": float(center_y[index]),
            "error_x": resolved_error_x,
            "error_y": resolved_error_y,
        }
        if np.isfinite(radius[index]) and radius[index] > 0.0:
            record["circle_radius"] = float(radius[index])
        records[frame_id] = record
    return records


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    mask = np.isfinite(values)
    if not np.any(mask):
        return values.copy()
    if np.sum(mask) == 1:
        return np.full_like(values, values[mask][0], dtype=float)
    indices: np.ndarray = np.arange(values.size, dtype=float)
    result: np.ndarray = values.astype(float, copy=True)
    result[~mask] = np.interp(indices[~mask], indices[mask], result[mask])
    return result


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    width = max(1, window | 1)
    half = width // 2
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        result[index] = np.median(
            values[max(0, index - half) : min(values.size, index + half + 1)]
        )
    return result


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, values.size):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result
