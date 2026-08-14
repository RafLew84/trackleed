"""Qt/model translation at the boundary of the headless detection workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.spot_detection_runtime import prepare_classic_image
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
    yolo_config_from_mapping,
)

SPOT_DETECTION_METHOD_LABELS: Mapping[str, str] = MappingProxyType(
    {"classic": "Classic", "yolo": "YOLO"}
)


def spot_detection_method_id(value: str) -> str:
    normalized = value.strip()
    for method_id, label in SPOT_DETECTION_METHOD_LABELS.items():
        if normalized.casefold() == label.casefold():
            return method_id
    return normalized.lower()


@dataclass(frozen=True)
class SpotDetectionDialogSettings:
    selected_method_id: str
    classic: ClassicDetectionConfig
    yolo: YoloDetectionConfig
    legacy_dl: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SpotDetectionDialogSettings:
        return cls(
            selected_method_id=spot_detection_method_id(
                str(values.get("backend", "Classic"))
            ),
            classic=classic_config_from_mapping(values.get("classic", {})),
            yolo=yolo_config_from_mapping(values.get("yolo", {})),
            legacy_dl=MappingProxyType(deepcopy(dict(values.get("dl", {})))),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "backend": SPOT_DETECTION_METHOD_LABELS.get(
                self.selected_method_id,
                self.selected_method_id,
            ),
            "classic": {
                "blur_sigma": self.classic.blur_sigma,
                "wth_radius": self.classic.wth_radius,
                "min_distance": self.classic.min_distance,
                "threshold_rel": self.classic.threshold_rel,
                "roi_size": self.classic.roi_size,
            },
            "dl": deepcopy(dict(self.legacy_dl)),
            "yolo": {
                "model_path": str(self.yolo.model_path) if self.yolo.model_path else None,
                "confidence": self.yolo.confidence,
                "iou": self.yolo.iou,
                "max_det": self.yolo.max_detections,
                "bbox_margin_px": self.yolo.bbox_margin_px,
                "subtract_bg": self.yolo.subtract_background,
                "device": self.yolo.device,
            },
        }


class SpotDetectionSettingsBinding:
    """Own detector settings while exposing only neutral MainWindow operations."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._settings = SpotDetectionDialogSettings.from_mapping(values or {})

    @property
    def dialog_settings(self) -> SpotDetectionDialogSettings:
        return self._settings

    @property
    def roi_size(self) -> int:
        """Return the shared ROI size used by detection, editing, and tracking."""
        return self._settings.classic.roi_size

    def accept(self, settings: SpotDetectionDialogSettings) -> None:
        self._settings = settings

    def restore(self, values: Mapping[str, Any]) -> None:
        self._settings = SpotDetectionDialogSettings.from_mapping(values)

    def to_mapping(self) -> dict[str, Any]:
        return self._settings.to_mapping()

    def processed_preview(
        self,
        image: np.ndarray,
        background: np.ndarray | None,
        metadata_masks: Sequence[QRect],
    ) -> np.ndarray:
        """Render the legacy Classic preprocessing preview."""
        masks = tuple(
            RectSnapshot(rect.x(), rect.y(), rect.width(), rect.height())
            for rect in metadata_masks
        )
        return prepare_classic_image(
            image,
            background,
            masks,
            self._settings.classic,
            zero_masked_pixels=False,
        )


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


class SpotDetectionCompletionKind(Enum):
    SUCCEEDED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class SpotDetectionCompletion:
    """Typed GUI-thread outcome produced by the worker's separate signals."""

    kind: SpotDetectionCompletionKind
    result: DetectionResult | None = None
    error_message: str | None = None

    @classmethod
    def from_worker_result(
        cls,
        result: DetectionResult | None,
    ) -> SpotDetectionCompletion:
        kind = (
            SpotDetectionCompletionKind.SUCCEEDED
            if result is not None
            else SpotDetectionCompletionKind.CANCELLED
        )
        return cls(kind=kind, result=result)

    @classmethod
    def from_worker_error(cls, error_message: str) -> SpotDetectionCompletion:
        return cls(
            kind=SpotDetectionCompletionKind.FAILED,
            error_message=error_message,
        )

    @property
    def failed(self) -> bool:
        return self.kind is SpotDetectionCompletionKind.FAILED

    @property
    def cancelled(self) -> bool:
        return self.kind is SpotDetectionCompletionKind.CANCELLED

    @property
    def successful_result(self) -> DetectionResult:
        if self.kind is not SpotDetectionCompletionKind.SUCCEEDED or self.result is None:
            raise RuntimeError("Detection completion does not contain a successful result")
        return self.result


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
    method_id = spot_detection_method_id(backend)
    if method_id == "classic":
        values = spot_settings.get("classic", {})
        config: DetectionConfig = classic_config_from_mapping(values)
    elif method_id == "yolo":
        values = spot_settings.get("yolo", {})
        config = yolo_config_from_mapping(values)
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
