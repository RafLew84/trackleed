from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def normalize_polygon_points(
    polygon_points: Iterable[Sequence[float]] | None,
    *,
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> list[tuple[float, float]]:
    """Normalize polygon vertices, optionally clipping them to image bounds."""
    normalized: list[tuple[float, float]] = []

    bounds: tuple[float, float] | None = None
    if image_shape is not None:
        image_h, image_w = _normalize_image_shape(image_shape)
        bounds = (float(max(0, image_w - 1)), float(max(0, image_h - 1)))

    for point in polygon_points or []:
        if not isinstance(point, Sequence) or len(point) != 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if bounds is not None:
            max_x, max_y = bounds
            x = float(np.clip(x, 0.0, max_x))
            y = float(np.clip(y, 0.0, max_y))
        if normalized:
            prev_x, prev_y = normalized[-1]
            if abs(prev_x - x) < 1e-6 and abs(prev_y - y) < 1e-6:
                continue
        normalized.append((x, y))

    if len(normalized) >= 2:
        first_x, first_y = normalized[0]
        last_x, last_y = normalized[-1]
        if abs(first_x - last_x) < 1e-6 and abs(first_y - last_y) < 1e-6:
            normalized.pop()

    return normalized


def polygon_points_are_valid(
    polygon_points: Iterable[Sequence[float]] | None,
    *,
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> bool:
    points = normalize_polygon_points(polygon_points, image_shape=image_shape)
    if len(points) < 3:
        return False
    return abs(_polygon_signed_area(points)) >= 0.5


def polygon_points_to_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    polygon_points: Iterable[Sequence[float]] | None,
) -> np.ndarray:
    """
    Rasterize polygon vertices to a binary mask of shape [H, W].

    Degenerate or invalid polygons return an all-false mask.
    """
    image_h, image_w = _normalize_image_shape(image_shape)
    mask = np.zeros((image_h, image_w), dtype=bool)

    points = normalize_polygon_points(polygon_points, image_shape=(image_h, image_w))
    if len(points) < 3 or abs(_polygon_signed_area(points)) < 0.5:
        return mask

    pil_image_module = __import__("PIL.Image", fromlist=["Image"])
    pil_draw_module = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])

    raster = pil_image_module.new("L", (image_w, image_h), 0)
    draw = pil_draw_module.Draw(raster)
    draw.polygon(points, outline=1, fill=1)
    raster_np = np.array(raster, dtype=np.uint8)
    return raster_np > 0


def _normalize_image_shape(
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must contain at least two dimensions, got: {image_shape}")
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image_shape must be positive, got: {image_shape}")
    return image_h, image_w


def _polygon_signed_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return 0.5 * float(area)
