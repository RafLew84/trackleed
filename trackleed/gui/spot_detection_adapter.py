"""Qt/model translation at the boundary of the headless detection workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile
from trackleed.core.spot_roi_adapter import spot_roi_from_detection
from trackleed.logic.workflows.spot_detection import (
    AnalysisRegionSnapshot,
    CircleRegion,
    ClassicDetectionConfig,
    DetectionConfig,
    DetectionFrame,
    DetectionProgress,
    DetectionRequest,
    DetectionResult,
    EllipseRegion,
    RectSnapshot,
    RunContext,
    SpotDetectionCancelled,
    YoloDetectionConfig,
    classic_config_from_mapping,
    run_spot_detection,
)


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


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
        config: DetectionConfig = classic_config_from_mapping(values)
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
    frame_ids = _stable_frame_ids(files)
    return DetectionRequest(
        method_id=method_id,
        frames=tuple(
            DetectionFrame(frame_id=frame_id, source_path=Path(leed_file.filepath))
            for frame_id, leed_file in zip(frame_ids, files, strict=True)
        ),
        config=config,
        background_path=(
            Path(background_file.filepath) if background_file is not None else None
        ),
        analysis_region=AnalysisRegionSnapshot(
            metadata_masks=tuple(
                RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
                for rect in metadata_masks
            ),
            circle=circle,
            ellipse=ellipse,
        ),
    )


def publish_detection_result(
    files: Sequence[LeedFile],
    result: DetectionResult,
) -> None:
    """Atomically replace spots for exactly the files represented by ``result``."""
    expected_ids = _stable_frame_ids(files)
    result_ids = tuple(frame.frame_id for frame in result.frames)
    if result_ids != expected_ids:
        raise ValueError(
            f"Detection result frames {result_ids} do not match request frames {expected_ids}"
        )

    replacements = tuple(
        [spot_roi_from_detection(spot) for spot in frame.spots]
        for frame in result.frames
    )
    for leed_file, spots in zip(files, replacements, strict=True):
        leed_file.spots = spots


def run_spot_detection_job(
    request: DetectionRequest,
    image_reader: Callable[[Path], np.ndarray],
    cancel_signal: CancellationSignal,
    *,
    progress_callback: Callable[[int], None],
) -> DetectionResult | None:
    """Host the public workflow in ``GenericWorker`` without GUI model access."""
    context = RunContext(
        image_reader=image_reader,
        report_progress=lambda progress: progress_callback(
            _progress_percentage(progress)
        ),
        is_cancelled=cancel_signal.is_set,
    )
    try:
        return run_spot_detection(request, context)
    except SpotDetectionCancelled:
        return None


def _progress_percentage(progress: DetectionProgress) -> int:
    if progress.total <= 0:
        return 100
    return round(100 * progress.completed / progress.total)


def _stable_frame_ids(files: Sequence[LeedFile]) -> tuple[int, ...]:
    frame_ids = tuple(leed_file.id for leed_file in files)
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError("Spot detection requires every requested frame to have an ID")
    stable_ids = tuple(int(frame_id) for frame_id in frame_ids if frame_id is not None)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Spot detection requires unique frame IDs")
    return stable_ids
