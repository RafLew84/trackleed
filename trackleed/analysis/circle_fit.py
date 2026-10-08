"""Algebraic circle fitting with residual-based parameter uncertainties."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def fit_circle_center(
    spot_positions: Sequence[Mapping[str, float]],
) -> dict[str, float] | None:
    """Preserve the existing circle estimator and propagate covariance to R.

    The optional radius error is a linearized standard uncertainty assuming
    independent, equally distributed algebraic residuals. Three points cannot
    estimate residual variance. It excludes image-calibration/systematic error.
    """
    if len(spot_positions) < 3:
        return None
    points = np.array(
        [[point["center_x"], point["center_y"]] for point in spot_positions],
        dtype=np.float64,
    )
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack((x, y, np.ones_like(x)))
    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design, -(x * x + y * y), rcond=None,
    )
    if rank < 3:
        return None
    a, b, c = coefficients
    center_x, center_y = -a / 2.0, -b / 2.0
    radius_squared = center_x * center_x + center_y * center_y - c
    if radius_squared <= 0:
        return None
    radius = np.sqrt(radius_squared)
    result = {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "circle_radius": float(radius),
    }
    if residuals.size and len(points) > 3:
        variance = float(residuals[0] / (len(points) - 3))
        covariance = variance * np.linalg.inv(design.T @ design)
        result["error_x"] = float(np.sqrt(max(covariance[0, 0], 0.0)) / 2.0)
        result["error_y"] = float(np.sqrt(max(covariance[1, 1], 0.0)) / 2.0)
        gradient = np.array([a / (4 * radius), b / (4 * radius), -1 / (2 * radius)])
        result["circle_radius_error"] = float(
            np.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
        )
    else:
        radial = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        error = float(np.std(radial - radius)) if radial.size else 0.0
        result["error_x"] = error
        result["error_y"] = error
    return result
