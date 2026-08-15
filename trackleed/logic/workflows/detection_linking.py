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


@dataclass(frozen=True)
class DirectionalLinkingConfig:
    center_mode: str = "auto"
    manual_center: tuple[float, float] | None = None
    direction_mode: str = "auto"
    lateral_angle_tolerance_deg: float = 20.0
    ring_bin_px: float = 25.0
    base_max_distance_px: float = 20.0
    dynamic_max_distance: bool = True
    dynamic_margin_px: float = 5.0
    use_ring_penalty: bool = True
    max_gap: int = 2
    max_misses: int = 3
    min_track_length: int = 2
    allow_direction_flip: bool = True
    min_tracks_for_flip: int = 5
    flip_hold_frames: int = 2
    min_radial_speed_for_flip: float = 1.0


DetectionLinkingConfig: TypeAlias = HungarianLinkingConfig
DetectionLinkingMethodConfig: TypeAlias = (
    HungarianLinkingConfig | DirectionalLinkingConfig
)


@dataclass(frozen=True)
class DetectionLinkingRequest:
    method_id: str
    frames: tuple[FrameDetections, ...]
    config: DetectionLinkingMethodConfig


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
    requires_pattern_center: bool = False


class DetectionLinkingFamily(str, Enum):
    ASSIGNMENT = "assignment"


class DetectionLinkingRuntime(Protocol):
    def validate_config(
        self,
        config: DetectionLinkingMethodConfig,
    ) -> tuple[PreflightIssue, ...]: ...

    def preflight(
        self,
        request: DetectionLinkingRequest,
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
        config_issues = method.runtime.validate_config(request.config)
        common_issues = _common_preflight_issues(request)
        blockers = {
            "frames_required",
            "duplicate_frame_id",
            "detections_required",
            "invalid_detection",
        }
        runtime_issues = ()
        if not config_issues and not any(
            issue.code in blockers for issue in common_issues
        ):
            runtime_issues = method.runtime.preflight(request)
        return DetectionLinkingPreflightReport(
            request.method_id,
            config_issues + runtime_issues + common_issues,
        )


def _common_preflight_issues(
    request: DetectionLinkingRequest,
) -> tuple[PreflightIssue, ...]:
    issues: list[PreflightIssue] = []
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
    if any(
        not _detection_is_valid(spot)
        for frame in request.frames
        for spot in frame.spots
    ):
        issues.append(
            PreflightIssue(
                "invalid_detection",
                "Every detection must have finite coordinates and a positive ROI.",
            )
        )
    return tuple(issues)


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
    from trackleed.analysis.directional_linking_runtime import DirectionalLinkingRuntime

    return DetectionLinkingCatalog(
        (
            DetectionLinkingMethod(
                "hungarian",
                "Hungarian",
                DetectionLinkingFamily.ASSIGNMENT,
                DetectionLinkingCapabilities(True, True, True),
                HungarianLinkingRuntime(),
            ),
            DetectionLinkingMethod(
                "directional",
                "Directional",
                DetectionLinkingFamily.ASSIGNMENT,
                DetectionLinkingCapabilities(True, True, True, True),
                DirectionalLinkingRuntime(),
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
        and _is_finite_number(getattr(roi, "width", None), positive=True)
        and _is_finite_number(getattr(roi, "height", None), positive=True)
        and _is_finite_number(getattr(spot, "center_x", None))
        and _is_finite_number(getattr(spot, "center_y", None))
    )


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    return isfinite(numeric) and (not positive or numeric > 0.0)
