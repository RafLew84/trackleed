"""Static apparatus geometry and its consistent synthetic appearance."""
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FullFrame:
    kind: str = "full_frame"


@dataclass(frozen=True)
class Circle:
    center_px: tuple[float, float]
    radius_px: float
    kind: str = "circle"


@dataclass(frozen=True)
class VerticalBand:
    x_range_px: tuple[float, float]
    kind: str = "vertical_band"


@dataclass(frozen=True)
class Polygon:
    vertices_px: tuple[tuple[float, float], ...]
    kind: str = "polygon"


Region = FullFrame | Circle | VerticalBand | Polygon


@dataclass(frozen=True)
class Apparatus:
    screen: FullFrame | Circle
    occluders: tuple[Circle | VerticalBand | Polygon, ...] = ()
    outside_screen_level: float = .02
    occluder_level: float = .04


@dataclass(frozen=True)
class ApparatusImage:
    screen: NDArray[np.bool_]
    occluders: NDArray[np.bool_]
    background: NDArray[np.float64]

    @property
    def signal_region(self) -> NDArray[np.bool_]:
        return self.screen & ~self.occluders


def parse_apparatus(value: Any) -> Apparatus:
    values = _object(value, {"screen", "occluders", "outside_screen_level", "occluder_level"})
    screen = _region(values.get("screen", {"kind": "full_frame"}))
    if not isinstance(screen, (FullFrame, Circle)):
        raise ValueError("apparatus.screen supports full_frame or circle")
    occluders = values.get("occluders", [])
    if not isinstance(occluders, (list, tuple)):
        raise ValueError("apparatus.occluders must be an array")
    parsed = []
    for raw in occluders:
        region = _region(raw)
        if isinstance(region, FullFrame):
            raise ValueError("apparatus.occluders supports circle, vertical_band or polygon")
        parsed.append(region)
    outside = _finite(values.get("outside_screen_level", .02), "outside_screen_level")
    occluded = _finite(values.get("occluder_level", .04), "occluder_level")
    if outside < 0 or occluded < 0:
        raise ValueError("apparatus background levels must be nonnegative")
    return Apparatus(screen, tuple(parsed), outside, occluded)


def _region(value: Any) -> Region:
    if not isinstance(value, dict):
        raise ValueError("apparatus region must be an object")
    kind = value.get("kind")
    if kind == "full_frame":
        _object(value, {"kind"})
        return FullFrame()
    if kind == "circle":
        values = _object(value, {"kind", "center_px", "radius_px"})
        center = _pair(values.get("center_px"), "center_px")
        radius = _finite(values.get("radius_px"), "radius_px")
        if radius <= 0:
            raise ValueError("apparatus.radius_px must be positive")
        return Circle(center, radius)
    if kind == "polygon":
        values = _object(value, {"kind", "vertices_px"})
        return _polygon(values.get("vertices_px"))
    if kind == "vertical_band":
        values = _object(value, {"kind", "x_range_px"})
        bounds = _pair(values.get("x_range_px"), "x_range_px")
        if bounds[0] > bounds[1]:
            raise ValueError("apparatus.x_range_px must be ordered")
        return VerticalBand(bounds)
    raise ValueError("apparatus region kind must be full_frame, circle, vertical_band or polygon")


def _object(value: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"apparatus object permits only {sorted(allowed)}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"apparatus.{label} must be finite")
    return float(value)


def _pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"apparatus.{label} must contain two coordinates")
    return _finite(value[0], label), _finite(value[1], label)


def prepare_apparatus(
    apparatus: Apparatus | None, image_size: tuple[int, int], background_level: float,
) -> ApparatusImage:
    width, height = image_size
    screen: NDArray[np.bool_] = np.ones((height, width), dtype=np.bool_)
    occluders = np.zeros_like(screen)
    background = np.full((height, width), background_level, dtype=np.float64)
    if apparatus is not None:
        screen = _raster(apparatus.screen, image_size)
        for region in apparatus.occluders:
            occluders |= _raster(region, image_size)
        background[~screen] = apparatus.outside_screen_level
        background[screen & occluders] = apparatus.occluder_level
    return ApparatusImage(screen, occluders, background)


def _raster(region: Region, image_size: tuple[int, int]) -> NDArray[np.bool_]:
    width, height = image_size
    if isinstance(region, FullFrame):
        return np.ones((height, width), dtype=np.bool_)
    y, x = np.ogrid[:height, :width]
    if isinstance(region, Circle):
        with np.errstate(over="ignore"):
            return np.hypot(x - region.center_px[0], y - region.center_px[1]) <= region.radius_px
    if isinstance(region, Polygon):
        vertices = region.vertices_px
        sides = []
        for (ax, ay), (bx, by) in zip(vertices, (*vertices[1:], vertices[0])):
            sides.append((bx - ax) * (y - ay) - (by - ay) * (x - ax))
        return np.logical_and.reduce([side >= 0 for side in sides]) | np.logical_and.reduce([side <= 0 for side in sides])
    return np.broadcast_to((x >= region.x_range_px[0]) & (x <= region.x_range_px[1]), (height, width))


def _polygon(value: Any) -> Polygon:
    if not isinstance(value, (list, tuple)) or not 3 <= len(value) <= 64:
        raise ValueError("apparatus polygon needs 3..64 vertices")
    vertices = tuple(_pair(point, "vertices_px") for point in value)
    if any(abs(v) > 1e9 for point in vertices for v in point):
        raise ValueError("apparatus polygon coordinates must be within +/-1e9 pixels")
    if len(set(vertices)) != len(vertices):
        raise ValueError("apparatus polygon vertices must be distinct")
    winding = 0
    for index, ((ax, ay), (bx, by)) in enumerate(zip(vertices, (*vertices[1:], vertices[0]))):
        sides = [(bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                 for i, (cx, cy) in enumerate(vertices) if i not in (index, (index + 1) % len(vertices))]
        sign = 1 if all(side > 0 for side in sides) else -1 if all(side < 0 for side in sides) else 0
        if sign == 0 or (winding and winding != sign):
            raise ValueError("apparatus polygon must be simple and strictly convex")
        winding = sign
    return Polygon(vertices)
