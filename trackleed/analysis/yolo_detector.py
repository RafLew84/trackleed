"""Compatibility adapter for the legacy Qt-based YOLO detector API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.spot_detection_runtime import YoloDetectionRuntime
from trackleed.core.data_models import SpotROI
from trackleed.core.spot_roi_adapter import spot_roi_from_detection
from trackleed.logic.workflows.spot_detection import (
    AnalysisRegionSnapshot,
    DetectionRuntimeInput,
    RectSnapshot,
    YoloDetectionConfig,
)

_DEFAULT_YOLO_CONFIG = YoloDetectionConfig()


class YOLODetectionError(RuntimeError):
    """Raised when YOLO inference fails."""


@dataclass
class YOLOSettings:
    model_path: str | None = None
    confidence: float = _DEFAULT_YOLO_CONFIG.confidence
    iou: float = _DEFAULT_YOLO_CONFIG.iou
    max_det: int = _DEFAULT_YOLO_CONFIG.max_detections
    bbox_margin_px: int = _DEFAULT_YOLO_CONFIG.bbox_margin_px
    device: str = _DEFAULT_YOLO_CONFIG.device


def detect_spots_with_yolo(
    image: np.ndarray,
    metadata_mask: list[QRect] | None,
    settings: YOLOSettings,
) -> list[SpotROI]:
    """Run the neutral YOLO runtime and translate its result to ``SpotROI``."""
    config = YoloDetectionConfig(
        model_path=Path(settings.model_path) if settings.model_path else None,
        confidence=settings.confidence,
        iou=settings.iou,
        max_detections=settings.max_det,
        bbox_margin_px=settings.bbox_margin_px,
        device=settings.device,
    )
    runtime_input = DetectionRuntimeInput(
        image=image,
        background_image=None,
        analysis_region=AnalysisRegionSnapshot(
            metadata_masks=tuple(
                RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
                for rect in metadata_mask or ()
            )
        ),
    )
    try:
        detected = YoloDetectionRuntime().detect(runtime_input, config)
    except Exception as exc:
        raise YOLODetectionError(str(exc)) from exc
    return [spot_roi_from_detection(spot) for spot in detected]
