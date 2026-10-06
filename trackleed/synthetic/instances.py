"""Independent nominal and signal masks, with half-open raster bounding boxes."""
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from trackleed.synthetic.profiles import GaussianShape


@dataclass(frozen=True)
class InstanceMasks:
    reflection_id: str
    geometry: NDArray[np.uint8]
    signal: NDArray[np.uint8]
    bbox_geometry: list[int] | None
    bbox_signal: list[int] | None


def make_instance_masks(
    reflection_id: str, x: float, y: float, amplitude: float,
    shape: GaussianShape, image_size: tuple[int, int],
) -> InstanceMasks:
    width, height = image_size
    ex, ey = shape.extents(3)
    columns = np.arange(max(0, math.ceil(x - ex)), min(width - 1, math.floor(x + ex)) + 1)
    rows = np.arange(max(0, math.ceil(y - ey)), min(height - 1, math.floor(y + ey)) + 1)
    dx, dy = np.meshgrid(columns - x, rows - y)
    geometry = np.zeros((height, width), dtype=np.uint8)
    geometry[np.ix_(rows, columns)] = (shape.squared_radius(dx, dy) <= 9).astype(np.uint8) * 255
    signal = geometry.copy() if amplitude > 0 else np.zeros_like(geometry)
    return InstanceMasks(reflection_id, geometry, signal, _bbox(geometry), _bbox(signal))


def _bbox(mask: NDArray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]


def mask_paths(schedule_id: str, index: int, reflection_id: str) -> tuple[str, str]:
    prefix = f"sequences/{schedule_id}/masks/frame_{index:06d}/{reflection_id}"
    return f"{prefix}_geometry.png", f"{prefix}_signal.png"
