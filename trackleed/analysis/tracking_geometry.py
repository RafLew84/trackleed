from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import numpy as np

from trackleed.analysis.peak_analyzer import find_pattern_center


def estimate_pattern_center_from_point_sets(
    point_sets: Sequence[Sequence[tuple[float, float]]],
    *,
    min_points_per_set: int = 4,
    max_points_per_set: int = 200,
    max_sets: int = 10,
) -> dict[str, float] | None:
    """Aggregate per-frame RANSAC centers without depending on GUI models."""
    candidates: list[dict[str, float]] = []
    selected_sets = point_sets[:max_sets] if max_sets > 0 else point_sets
    for points in selected_sets:
        if len(points) < min_points_per_set:
            continue
        positions = [
            {"center_x": float(x), "center_y": float(y)}
            for x, y in points[:max_points_per_set]
        ]
        center = find_pattern_center(positions)
        if center is not None:
            candidates.append(center)
    if not candidates:
        return None
    return {
        "center_x": float(np.median([item["center_x"] for item in candidates])),
        "center_y": float(np.median([item["center_y"] for item in candidates])),
        "error_x": float(np.median([item.get("error_x", 0.0) for item in candidates])),
        "error_y": float(np.median([item.get("error_y", 0.0) for item in candidates])),
    }


def radial_unit_and_radius(
    center_xy: Tuple[float, float],
    point_xy: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    """
    Compute the radial unit vector and radius from center to point.

    Returns `(unit_vector, radius)`. The unit vector is zero when the radius is
    too small to define a stable direction.
    """
    vec = np.array(point_xy, dtype=np.float32) - np.array(center_xy, dtype=np.float32)
    radius = float(np.linalg.norm(vec))
    if radius > 1e-6:
        return vec / radius, radius
    return np.zeros(2, dtype=np.float32), 0.0


def angular_offset_deg(radial_unit: np.ndarray, displacement: np.ndarray) -> float:
    """
    Angle in degrees between the expected radial direction and a displacement.

    Returns `0.0` if either vector is near zero.
    """
    disp = np.array(displacement, dtype=np.float32)
    ru = np.array(radial_unit, dtype=np.float32)
    disp_norm = float(np.linalg.norm(disp))
    ru_norm = float(np.linalg.norm(ru))
    if disp_norm < 1e-9 or ru_norm < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(ru, disp) / (ru_norm * disp_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
