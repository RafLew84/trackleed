"""Qt/model translation at the boundary of the headless detection workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.logic.workflows.spot_detection import (
    CircleRegion,
    ClassicDetectionConfig,
    DetectionConfig,
    DetectionFrame,
    DetectionRequest,
    DetectionResult,
    EllipseRegion,
    RectSnapshot,
    YoloDetectionConfig,
)


def build_detection_request(
    *,
    files: Sequence[LeedFile],
    spot_settings: Mapping[str, Any],
    background_file: LeedFile | None,
    metadata_masks: Sequence[QRect],
    analysis_circle: tuple[int, int, int] | None,
    analysis_ellipse: tuple[int, int, int, int, float] | None,
) -> DetectionRequest:
    """Snapshot mutable GUI state into neutral immutable workflow values."""
    backend = str(spot_settings.get("backend", ""))
    method_id = {"Classic": "classic", "YOLO": "yolo"}.get(backend, backend.lower())
    if method_id == "classic":
        values = spot_settings.get("classic", {})
        config: DetectionConfig = ClassicDetectionConfig(
            blur_sigma=float(values.get("blur_sigma", 20.0)),
            wth_radius=int(values.get("wth_radius", 8)),
            min_distance=int(values.get("min_distance", 10)),
            threshold_rel=float(values.get("threshold_rel", 0.1)),
            roi_size=int(values.get("roi_size", 32)),
        )
    elif method_id == "yolo":
        values = spot_settings.get("yolo", {})
        model_path = values.get("model_path")
        config = YoloDetectionConfig(
            model_path=Path(model_path) if model_path else None,
            confidence=float(values.get("confidence", 0.25)),
            iou=float(values.get("iou", 0.45)),
            max_detections=int(values.get("max_det", 300)),
            bbox_margin_px=int(values.get("bbox_margin_px", 2)),
            subtract_background=bool(values.get("subtract_bg", False)),
            device=str(values.get("device", "auto")),
        )
    else:
        config = ClassicDetectionConfig()

    circle = CircleRegion(*map(float, analysis_circle)) if analysis_circle else None
    ellipse = EllipseRegion(*map(float, analysis_ellipse)) if analysis_ellipse else None
    return DetectionRequest(
        method_id=method_id,
        frames=tuple(
            DetectionFrame(frame_id=index, source_path=Path(leed_file.filepath))
            for index, leed_file in enumerate(files)
        ),
        config=config,
        background_path=(
            Path(background_file.filepath) if background_file is not None else None
        ),
        metadata_masks=tuple(
            RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
            for rect in metadata_masks
        ),
        circle_region=circle,
        ellipse_region=ellipse,
    )


def publish_detection_result(
    files: Sequence[LeedFile],
    result: DetectionResult,
) -> None:
    """Atomically replace spots for exactly the files represented by ``result``."""
    expected_ids = tuple(range(len(files)))
    result_ids = tuple(frame.frame_id for frame in result.frames)
    if result_ids != expected_ids:
        raise ValueError(
            f"Detection result frames {result_ids} do not match request frames {expected_ids}"
        )

    replacements = tuple(
        [
            SpotROI(
                id=spot.id,
                roi_rect=QRect(
                    spot.roi.x,
                    spot.roi.y,
                    spot.roi.width,
                    spot.roi.height,
                ),
                center_x=spot.center_x,
                center_y=spot.center_y,
                intensity=spot.intensity,
                source=spot.source,
                polygon_seed_points=[
                    (float(point[0]), float(point[1]))
                    for point in spot.polygon_seed_points
                ],
            )
            for spot in frame.spots
        ]
        for frame in result.frames
    )
    for leed_file, spots in zip(files, replacements, strict=True):
        leed_file.spots = spots
