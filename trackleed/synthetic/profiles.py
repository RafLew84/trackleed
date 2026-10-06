"""Finite Gaussian geometry, evaluated at image pixel centers."""
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianShape:
    sigma_u_px: float = 4.0
    sigma_v_px: float = 4.0
    theta_rad: float = 0.0

    def extents(self, extent: float = 4) -> tuple[float, float]:
        cosine, sine = math.cos(self.theta_rad), math.sin(self.theta_rad)
        return (extent * math.hypot(self.sigma_u_px * cosine, self.sigma_v_px * sine),
                extent * math.hypot(self.sigma_u_px * sine, self.sigma_v_px * cosine))

    def squared_radius(self, dx: NDArray, dy: NDArray) -> NDArray:
        with np.errstate(over="ignore"):
            if self.sigma_u_px == self.sigma_v_px:
                # Preserve the circular arithmetic of historical packages.
                u, v = dx / self.sigma_u_px, dy / self.sigma_v_px
            else:
                cosine, sine = math.cos(self.theta_rad), math.sin(self.theta_rad)
                u = (cosine * dx + sine * dy) / self.sigma_u_px
                v = (-sine * dx + cosine * dy) / self.sigma_v_px
            return u * u + v * v

    def overlaps(self, dx: float, dy: float, extent: int) -> bool:
        """Closed ellipses with this shared shape overlap in normalized coordinates."""
        if self.sigma_u_px == self.sigma_v_px:
            return math.hypot(dx, dy) <= extent * (self.sigma_u_px + self.sigma_v_px)
        cosine, sine = math.cos(self.theta_rad), math.sin(self.theta_rad)
        u = (cosine * dx + sine * dy) / self.sigma_u_px
        v = (-sine * dx + cosine * dy) / self.sigma_v_px
        return math.hypot(u, v) <= 2 * extent


def sample_profile(x: float, y: float, shape: GaussianShape) -> tuple[NDArray, NDArray, NDArray]:
    ex, ey = shape.extents()
    columns = np.arange(math.ceil(x - ex), math.floor(x + ex) + 1)
    rows = np.arange(math.ceil(y - ey), math.floor(y + ey) + 1)
    dx, dy = np.meshgrid(columns - x, rows - y)
    q2 = shape.squared_radius(dx, dy)
    profile = np.where(q2 <= 16, np.exp(-q2 / 2), 0)
    if not profile.sum() > 0:
        raise ValueError("Unsupported spatial sampling: W0 = 0")
    return rows, columns, profile
