from __future__ import annotations

from typing import Tuple

import numpy as np


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
