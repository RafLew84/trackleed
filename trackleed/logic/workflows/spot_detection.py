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


class DetectionFamily(str, Enum):
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
    family: DetectionFamily
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

    def preflight(self, method_id: str, config: DetectionConfig) -> PreflightReport:
        method = self.method(method_id)
        return PreflightReport(method_id, method.runtime.preflight(config))


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
                family=DetectionFamily.CLASSICAL,
                capabilities=DetectionCapabilities(True, True, False),
                runtime=ClassicDetectionRuntime(),
            ),
            DetectionMethod(
                method_id="yolo",
                display_name="YOLO",
                family=DetectionFamily.NEURAL,
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
    report = catalog.preflight(request.method_id, request.config)
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
