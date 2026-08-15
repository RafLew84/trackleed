"""Typed, headless orchestration for spot detection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np


@dataclass(frozen=True)
class RectSnapshot:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CircleRegion:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class EllipseRegion:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle_degrees: float = 0.0


@dataclass(frozen=True)
class AnalysisRegionSnapshot:
    metadata_masks: tuple[RectSnapshot, ...] = ()
    circle: CircleRegion | None = None
    ellipse: EllipseRegion | None = None


@dataclass(frozen=True)
class ClassicDetectionConfig:
    blur_sigma: float = 20.0
    wth_radius: int = 8
    min_distance: int = 10
    threshold_rel: float = 0.1
    roi_size: int = 32


def classic_config_from_mapping(
    values: Mapping[str, Any],
) -> ClassicDetectionConfig:
    """Normalize a legacy settings mapping using the contract defaults."""
    defaults = ClassicDetectionConfig()
    return ClassicDetectionConfig(
        blur_sigma=float(values.get("blur_sigma", defaults.blur_sigma)),
        wth_radius=int(values.get("wth_radius", defaults.wth_radius)),
        min_distance=int(values.get("min_distance", defaults.min_distance)),
        threshold_rel=float(values.get("threshold_rel", defaults.threshold_rel)),
        roi_size=int(values.get("roi_size", defaults.roi_size)),
    )


@dataclass(frozen=True)
class YoloDetectionConfig:
    model_path: Path | None = None
    confidence: float = 0.25
    iou: float = 0.45
    max_detections: int = 300
    bbox_margin_px: int = 2
    subtract_background: bool = False
    device: str = "auto"


def yolo_config_from_mapping(values: Mapping[str, Any]) -> YoloDetectionConfig:
    """Normalize a legacy YOLO settings mapping using contract defaults."""
    defaults = YoloDetectionConfig()
    model_path = values.get("model_path", defaults.model_path)
    return YoloDetectionConfig(
        model_path=Path(model_path) if model_path else None,
        confidence=float(values.get("confidence", defaults.confidence)),
        iou=float(values.get("iou", defaults.iou)),
        max_detections=int(values.get("max_det", defaults.max_detections)),
        bbox_margin_px=int(
            values.get("bbox_margin_px", defaults.bbox_margin_px)
        ),
        subtract_background=bool(
            values.get("subtract_bg", defaults.subtract_background)
        ),
        device=str(values.get("device", defaults.device)),
    )


DetectionConfig: TypeAlias = ClassicDetectionConfig | YoloDetectionConfig


@dataclass(frozen=True)
class DetectionFrame:
    frame_id: int
    source_path: Path


@dataclass(frozen=True)
class DetectionRequest:
    method_id: str
    frames: tuple[DetectionFrame, ...]
    config: DetectionConfig
    background_path: Path | None = None
    analysis_region: AnalysisRegionSnapshot = field(
        default_factory=AnalysisRegionSnapshot
    )


@dataclass(frozen=True)
class DetectedSpot:
    id: int
    roi: RectSnapshot
    center_x: float
    center_y: float
    intensity: float = 0.0
    source: str = "auto"
    polygon_seed_points: tuple[tuple[float, float], ...] = ()


INDEPENDENT_DETECTION_SOURCES = frozenset({"auto", "yolo"})


def is_independent_detection(spot: DetectedSpot) -> bool:
    """Return whether a spot came from an independent detector run."""
    return spot.source in INDEPENDENT_DETECTION_SOURCES


@dataclass(frozen=True)
class FrameDetections:
    frame_id: int
    spots: tuple[DetectedSpot, ...]


@dataclass(frozen=True)
class DetectionResult:
    frames: tuple[FrameDetections, ...]


@dataclass(frozen=True)
class DetectionProgress:
    completed: int
    total: int
    frame_id: int


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    method_id: str
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class DetectionCapabilities:
    supports_background: bool
    supports_metadata_masks: bool
    requires_model: bool
    supports_series: bool = True


class SpotDetectionFamily(str, Enum):
    CLASSICAL = "classical"
    NEURAL = "neural"


@dataclass(frozen=True)
class DetectionRuntimeInput:
    image: np.ndarray
    background_image: np.ndarray | None
    analysis_region: AnalysisRegionSnapshot


class DetectionRuntimeAdapter(Protocol):
    def preflight(self, config: DetectionConfig) -> tuple[PreflightIssue, ...]: ...

    def detect(
        self,
        runtime_input: DetectionRuntimeInput,
        config: DetectionConfig,
    ) -> tuple[DetectedSpot, ...]: ...


@dataclass(frozen=True)
class DetectionMethod:
    method_id: str
    display_name: str
    family: SpotDetectionFamily
    capabilities: DetectionCapabilities
    runtime: DetectionRuntimeAdapter


@dataclass(frozen=True)
class SpotDetectionCatalog:
    methods: tuple[DetectionMethod, ...]

    def method(self, method_id: str) -> DetectionMethod:
        matches = tuple(method for method in self.methods if method.method_id == method_id)
        if len(matches) != 1:
            raise UnknownDetectionMethod(f"Unknown spot detection method: {method_id}")
        return matches[0]

    def preflight(self, request: DetectionRequest) -> PreflightReport:
        method = self.method(request.method_id)
        issues = list(method.runtime.preflight(request.config))
        frame_ids = tuple(frame.frame_id for frame in request.frames)
        if not frame_ids:
            issues.append(PreflightIssue("frames_required", "Select at least one frame."))
        elif len(set(frame_ids)) != len(frame_ids):
            issues.append(
                PreflightIssue("duplicate_frame_id", "Frame IDs must be unique.")
            )
        if len(frame_ids) > 1 and not method.capabilities.supports_series:
            issues.append(
                PreflightIssue(
                    "mode_unavailable",
                    f"Method {request.method_id} does not support series detection.",
                )
            )
        if (
            request.background_path is not None
            and not method.capabilities.supports_background
        ):
            issues.append(
                PreflightIssue(
                    "background_unsupported",
                    f"Method {request.method_id} does not support a background image.",
                )
            )
        if (
            request.analysis_region.metadata_masks
            and not method.capabilities.supports_metadata_masks
        ):
            issues.append(
                PreflightIssue(
                    "masks_unsupported",
                    f"Method {request.method_id} does not support metadata masks.",
                )
            )
        if (
            isinstance(request.config, YoloDetectionConfig)
            and request.config.subtract_background
            and request.background_path is None
        ):
            issues.append(
                PreflightIssue(
                    "background_required",
                    "YOLO background subtraction requires a background image.",
                )
            )
        return PreflightReport(request.method_id, tuple(issues))


@dataclass(frozen=True)
class RunContext:
    image_reader: Callable[[Path], np.ndarray]
    report_progress: Callable[[DetectionProgress], None] = lambda _progress: None
    is_cancelled: Callable[[], bool] = lambda: False
    detection_catalog: SpotDetectionCatalog | None = None


class SpotDetectionError(RuntimeError):
    """Base error for a spot-detection run."""


class UnknownDetectionMethod(SpotDetectionError):
    """Raised for method IDs outside the headless catalog."""


class SpotDetectionPreflightError(SpotDetectionError):
    def __init__(self, report: PreflightReport):
        self.report = report
        details = "; ".join(issue.message for issue in report.issues)
        super().__init__(details or f"Method {report.method_id} is not ready")


class SpotDetectionCancelled(SpotDetectionError):
    """Raised before an incomplete result can escape the workflow."""


class SpotDetectionExecutionError(SpotDetectionError):
    def __init__(self, frame_id: int, cause: Exception):
        self.frame_id = frame_id
        super().__init__(f"Spot detection failed on frame {frame_id}: {cause}")


def default_spot_detection_catalog() -> SpotDetectionCatalog:
    """Build the built-in catalog without importing any GUI adapter."""
    from trackleed.analysis.spot_detection_runtime import (
        ClassicDetectionRuntime,
        YoloDetectionRuntime,
    )

    return SpotDetectionCatalog(
        methods=(
            DetectionMethod(
                method_id="classic",
                display_name="Classic",
                family=SpotDetectionFamily.CLASSICAL,
                capabilities=DetectionCapabilities(True, True, False),
                runtime=ClassicDetectionRuntime(),
            ),
            DetectionMethod(
                method_id="yolo",
                display_name="YOLO",
                family=SpotDetectionFamily.NEURAL,
                capabilities=DetectionCapabilities(True, True, True),
                runtime=YoloDetectionRuntime(),
            ),
        )
    )


def run_spot_detection(
    request: DetectionRequest,
    context: RunContext,
) -> DetectionResult:
    """Detect spots frame-by-frame and return only a complete batch result."""
    catalog = context.detection_catalog or default_spot_detection_catalog()
    method = catalog.method(request.method_id)
    report = catalog.preflight(request)
    if not report.ready:
        raise SpotDetectionPreflightError(report)

    completed: list[FrameDetections] = []
    total = len(request.frames)
    for frame in request.frames:
        if context.is_cancelled():
            raise SpotDetectionCancelled("Spot detection was cancelled")
        try:
            image = context.image_reader(frame.source_path)
            if image is None or image.size == 0:
                raise ValueError(f"Could not load image: {frame.source_path}")
            background = None
            if request.background_path is not None:
                background = context.image_reader(request.background_path)
                if background is None or background.size == 0:
                    raise ValueError(
                        f"Could not load background image: {request.background_path}"
                    )
            spots = method.runtime.detect(
                DetectionRuntimeInput(
                    image=image,
                    background_image=background,
                    analysis_region=request.analysis_region,
                ),
                request.config,
            )
            if context.is_cancelled():
                raise SpotDetectionCancelled("Spot detection was cancelled")
        except SpotDetectionError:
            raise
        except Exception as exc:
            raise SpotDetectionExecutionError(frame.frame_id, exc) from exc
        completed.append(FrameDetections(frame.frame_id, spots))
        context.report_progress(DetectionProgress(len(completed), total, frame.frame_id))
    return DetectionResult(tuple(completed))
