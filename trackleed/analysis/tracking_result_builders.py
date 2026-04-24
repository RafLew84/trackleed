from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.core.data_models import SpotROI, Track


def build_track(
    *,
    track_id: int,
    spots_by_image_id: Mapping[int, SpotROI] | None = None,
) -> Track:
    return Track(
        track_id=int(track_id),
        spots_by_image_id=dict(spots_by_image_id or {}),
    )


def build_track_with_initial_spot(
    *,
    track_id: int,
    frame_id: int,
    spot: SpotROI,
) -> Track:
    return build_track(
        track_id=track_id,
        spots_by_image_id={int(frame_id): spot},
    )


def build_spot_roi(
    *,
    spot_id: int,
    roi_rect: QRect,
    center_x: float,
    center_y: float,
    source: str,
    visible: bool | None = None,
    intensity: float | None = None,
    polygon_seed_points: Iterable[tuple[float, float]] | None = None,
) -> SpotROI:
    if intensity is None:
        intensity = 1.0 if bool(visible) else 0.0
    spot = SpotROI(
        id=int(spot_id),
        roi_rect=roi_rect,
        center_x=float(center_x),
        center_y=float(center_y),
        intensity=float(intensity),
        source=str(source),
    )
    if polygon_seed_points is not None:
        spot.polygon_seed_points = [
            (float(point[0]), float(point[1])) for point in polygon_seed_points
        ]
    return spot


def set_spot_attributes(spot: SpotROI, attributes: Mapping[str, Any]) -> None:
    for name, value in attributes.items():
        setattr(spot, str(name), value)


def apply_prefixed_point_metadata(
    spot: SpotROI,
    *,
    prefix: str,
    raw_x: float,
    raw_y: float,
    refined_x: float,
    refined_y: float,
    visible: bool,
) -> None:
    set_spot_attributes(
        spot,
        {
            f"{prefix}_raw_x": _float_or_nan(raw_x),
            f"{prefix}_raw_y": _float_or_nan(raw_y),
            f"{prefix}_refined_x": _float_or_nan(refined_x),
            f"{prefix}_refined_y": _float_or_nan(refined_y),
            f"{prefix}_visibility": bool(visible),
        },
    )


def bbox_center_and_aspect_ratio_from_xyxy(
    bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
) -> tuple[np.ndarray | None, float | None]:
    if bbox_xyxy is None:
        return None, None
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return None, None
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    width = float(x1 - x0)
    height = float(y1 - y0)
    if width <= 0.0 or height <= 0.0:
        return None, None
    center_xy = np.array(
        [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
        dtype=np.float32,
    )
    return center_xy, float(width / height)


def apply_bbox_xywh_metadata(
    spot: SpotROI,
    *,
    bbox_xywh: np.ndarray | tuple[float, float, float, float] | None,
    bbox_attribute_name: str,
) -> None:
    values = None
    if bbox_xywh is not None:
        bbox = np.asarray(bbox_xywh, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            values = tuple(float(v) for v in bbox[:4].tolist())
    setattr(spot, str(bbox_attribute_name), values)


def apply_bbox_xyxy_metadata(
    spot: SpotROI,
    *,
    bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    bbox_attribute_name: str,
    aspect_ratio_attribute_name: str | None = None,
) -> None:
    values = None
    aspect_ratio = None
    if bbox_xyxy is not None:
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            values = tuple(float(v) for v in bbox[:4].tolist())
            _, aspect_ratio = bbox_center_and_aspect_ratio_from_xyxy(bbox[:4])
    setattr(spot, str(bbox_attribute_name), values)
    if aspect_ratio_attribute_name is not None:
        setattr(
            spot,
            str(aspect_ratio_attribute_name),
            float(aspect_ratio) if aspect_ratio is not None else None,
        )


def _float_or_nan(value: float) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return scalar if np.isfinite(scalar) else float("nan")
