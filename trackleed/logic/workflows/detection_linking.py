"""Typed, headless orchestration for linking independent spot detections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, TypeAlias

from trackleed.logic.workflows.seeded_tracking import TrackingResult
from trackleed.logic.workflows.spot_detection import (
    FrameDetections,
    PreflightIssue,
    is_independent_detection,
)


@dataclass(frozen=True)
class HungarianLinkingConfig:
    max_distance_px: float = 20.0
    max_gap: int = 2
    min_track_length: int = 2


DetectionLinkingConfig: TypeAlias = HungarianLinkingConfig


@dataclass(frozen=True)
class DetectionLinkingRequest:
    method_id: str
    frames: tuple[FrameDetections, ...]
    config: DetectionLinkingConfig


@dataclass(frozen=True)
class DetectionLinkingProgress:
    completed: int
    total: int
    frame_id: int


@dataclass(frozen=True)
class DetectionLinkingCapabilities:
    consumes_detections: bool
    preserves_gaps: bool
    supports_review: bool


class DetectionLinkingFamily(str, Enum):
    ASSIGNMENT = "assignment"


class DetectionLinkingRuntime(Protocol):
    def validate_config(
        self,
        config: DetectionLinkingConfig,
    ) -> tuple[PreflightIssue, ...]: ...

    def run(
        self,
        request: DetectionLinkingRequest,
        context: RunContext,
    ) -> TrackingResult: ...


@dataclass(frozen=True)
class DetectionLinkingMethod:
    method_id: str
    display_name: str
    family: DetectionLinkingFamily
    capabilities: DetectionLinkingCapabilities
    runtime: DetectionLinkingRuntime


@dataclass(frozen=True)
class DetectionLinkingPreflightReport:
    method_id: str
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class DetectionLinkingCatalog:
    methods: tuple[DetectionLinkingMethod, ...]

    def method(self, method_id: str) -> DetectionLinkingMethod:
        matches = tuple(method for method in self.methods if method.method_id == method_id)
        if len(matches) != 1:
            raise UnknownDetectionLinkingMethod(
                f"Unknown detection linking method: {method_id}"
            )
        return matches[0]

    def preflight(
        self,
        request: DetectionLinkingRequest,
    ) -> DetectionLinkingPreflightReport:
        method = self.method(request.method_id)
        issues = list(method.runtime.validate_config(request.config))
        frame_ids = tuple(frame.frame_id for frame in request.frames)
        if len(frame_ids) < 2:
            issues.append(
                PreflightIssue(
                    "frames_required",
                    "Detection linking requires at least two frames.",
                )
            )
        elif len(set(frame_ids)) != len(frame_ids):
            issues.append(PreflightIssue("duplicate_frame_id", "Frame IDs must be unique."))
        if not any(frame.spots for frame in request.frames):
            issues.append(
                PreflightIssue(
                    "detections_required",
                    "Run spot detection before detection linking.",
                )
            )
        if any(
            not is_independent_detection(spot)
            for frame in request.frames
            for spot in frame.spots
        ):
            issues.append(
                PreflightIssue(
                    "independent_detections_required",
                    "Detection linking accepts only independent Classic or YOLO detections.",
                )
            )
        if any(not _detection_is_valid(spot) for frame in request.frames for spot in frame.spots):
            issues.append(
                PreflightIssue(
                    "invalid_detection",
                    "Every detection must have finite coordinates and a positive ROI.",
                )
            )
        return DetectionLinkingPreflightReport(request.method_id, tuple(issues))


@dataclass(frozen=True)
class RunContext:
    report_progress: Callable[[DetectionLinkingProgress], None] = lambda _value: None
    is_cancelled: Callable[[], bool] = lambda: False
    catalog: DetectionLinkingCatalog | None = None


class DetectionLinkingError(RuntimeError):
    """Base error for detection-linking workflows."""


class UnknownDetectionLinkingMethod(DetectionLinkingError):
    """Raised for method IDs outside the catalog."""


class DetectionLinkingPreflightError(DetectionLinkingError):
    def __init__(self, report: DetectionLinkingPreflightReport):
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


class DetectionLinkingCancelled(DetectionLinkingError):
    """Raised before an incomplete linking result can escape."""


class DetectionLinkingExecutionError(DetectionLinkingError):
    """Wrap an unexpected linker failure at the public workflow boundary."""


def default_detection_linking_catalog() -> DetectionLinkingCatalog:
    from trackleed.analysis.detection_linking_runtime import HungarianLinkingRuntime

    return DetectionLinkingCatalog(
        (
            DetectionLinkingMethod(
                "hungarian",
                "Hungarian",
                DetectionLinkingFamily.ASSIGNMENT,
                DetectionLinkingCapabilities(True, True, True),
                HungarianLinkingRuntime(),
            ),
        )
    )


def run_detection_linking(
    request: DetectionLinkingRequest,
    context: RunContext,
) -> TrackingResult:
    """Return a complete immutable track set or fail without publication."""
    catalog = context.catalog or default_detection_linking_catalog()
    method = catalog.method(request.method_id)
    report = catalog.preflight(request)
    if not report.ready:
        raise DetectionLinkingPreflightError(report)
    if context.is_cancelled():
        raise DetectionLinkingCancelled("Detection linking was cancelled")
    try:
        result = method.runtime.run(request, context)
    except DetectionLinkingError:
        raise
    except Exception as exc:
        raise DetectionLinkingExecutionError(str(exc)) from exc
    if context.is_cancelled():
        raise DetectionLinkingCancelled("Detection linking was cancelled")
    return result


def _detection_is_valid(spot: object) -> bool:
    roi = getattr(spot, "roi", None)
    return bool(
        roi is not None
        and getattr(roi, "width", 0) > 0
        and getattr(roi, "height", 0) > 0
        and isfinite(float(getattr(spot, "center_x", float("nan"))))
        and isfinite(float(getattr(spot, "center_y", float("nan"))))
    )
