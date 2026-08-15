"""Qt/model translation for the headless detection-linking workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from trackleed.core.data_models import LeedFile, SpotROI
from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCancelled,
    DetectionLinkingMethodConfig,
    DetectionLinkingProgress,
    DetectionLinkingRequest,
    DirectionalLinkingConfig,
    HungarianLinkingConfig,
    RunContext,
    run_detection_linking,
)
from trackleed.logic.workflows.spot_detection import (
    DetectedSpot,
    FrameDetections,
    RectSnapshot,
)
from trackleed.logic.workflows.seeded_tracking import TrackingResult


@dataclass(frozen=True)
class DetectionLinkingDialogSettings:
    hungarian: HungarianLinkingConfig = HungarianLinkingConfig()
    directional: DirectionalLinkingConfig = DirectionalLinkingConfig()


class DetectionLinkingSettingsBinding:
    """Keep accepted linker settings in process memory only."""

    def __init__(self) -> None:
        self._settings = DetectionLinkingDialogSettings()

    @property
    def dialog_settings(self) -> DetectionLinkingDialogSettings:
        return self._settings

    def accept(self, settings: DetectionLinkingDialogSettings) -> None:
        self._settings = settings


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


def build_detection_linking_request(
    files: Sequence[LeedFile],
    config: DetectionLinkingMethodConfig,
    *,
    method_id: str = "hungarian",
) -> DetectionLinkingRequest:
    """Snapshot independent mutable detections into neutral workflow values."""
    frame_ids = _stable_frame_ids(files)
    return DetectionLinkingRequest(
        method_id,
        tuple(
            FrameDetections(
                frame_id,
                tuple(_snapshot_detection(spot) for spot in file.spots),
            )
            for frame_id, file in zip(frame_ids, files, strict=True)
        ),
        config,
    )


def run_detection_linking_job(
    request: DetectionLinkingRequest,
    cancel_signal: CancellationSignal,
    *,
    progress_callback: Callable[[int], None],
) -> TrackingResult | None:
    context = RunContext(
        report_progress=lambda progress: progress_callback(
            _progress_percentage(progress)
        ),
        is_cancelled=cancel_signal.is_set,
    )
    try:
        return run_detection_linking(request, context)
    except DetectionLinkingCancelled:
        return None


def _snapshot_detection(spot: SpotROI) -> DetectedSpot:
    rect = spot.roi_rect
    return DetectedSpot(
        int(spot.id),
        RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height()),
        float(spot.center_x),
        float(spot.center_y),
        float(spot.intensity),
        str(spot.source),
        tuple((float(x), float(y)) for x, y in spot.polygon_seed_points),
    )


def _stable_frame_ids(files: Sequence[LeedFile]) -> tuple[int, ...]:
    frame_ids = tuple(file.id for file in files)
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError("Detection linking requires every frame to have an ID")
    return tuple(int(frame_id) for frame_id in frame_ids if frame_id is not None)


def _progress_percentage(progress: DetectionLinkingProgress) -> int:
    if progress.total <= 0:
        return 100
    return round(100 * progress.completed / progress.total)
