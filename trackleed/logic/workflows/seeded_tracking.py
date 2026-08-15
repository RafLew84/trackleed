"""Headless seeded spot-tracking workflow contracts and orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np


@dataclass(frozen=True)
class SeedRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SeededTrackingFrame:
    frame_id: int
    source_path: Path


@dataclass(frozen=True)
class TrackingSeed:
    seed_id: int
    center_x: float
    center_y: float
    roi: SeedRect
    intensity: float = 0.0
    source: str = "seed"


@dataclass(frozen=True)
class OpticalFlowConfig:
    roi_size: int = 32
    win_size: int = 41
    max_level: int = 4
    max_iterations: int = 50
    epsilon: float = 0.01
    min_eigen_threshold: float = 1e-5
    use_centroid_refinement: bool = True
    centroid_patch_multiplier: float = 1.5
    centroid_rel_threshold: float = 0.4
    centroid_max_expand: int = 1
    subtract_background: bool = False


class OutOfBoundsPolicy(Enum):
    SKIP = "skip"
    MARK_LOST = "mark_lost"


@dataclass(frozen=True)
class SkipAnchorOpticalFlowConfig(OpticalFlowConfig):
    error_threshold: float = 30.0
    forward_backward_threshold: float = 8.2
    use_forward_backward: bool = True
    max_skip_frames: int = 0
    out_of_bounds: OutOfBoundsPolicy = OutOfBoundsPolicy.SKIP
    use_ncc_check: bool = True
    ncc_threshold: float = 0.65
    ncc_patch_radius: int = 4


SeededTrackingConfig: TypeAlias = OpticalFlowConfig | SkipAnchorOpticalFlowConfig


class SeedRepresentation(Enum):
    POINT = "point"
    BOUNDING_BOX = "bounding_box"
    MASK = "mask"


class TrackingFamily(Enum):
    CLASSICAL = "classical"
    POINT = "point"
    BOUNDING_BOX = "bbox"
    MASK = "mask"


@dataclass(frozen=True)
class SeededTrackingRequest:
    method_id: str
    frames: tuple[SeededTrackingFrame, ...]
    reference_frame_id: int
    seeds: tuple[TrackingSeed, ...]
    config: SeededTrackingConfig
    background_path: Path | None = None
    seed_representation: SeedRepresentation = SeedRepresentation.POINT


@dataclass(frozen=True)
class TrackObservation:
    frame_id: int
    roi: SeedRect
    center_x: float
    center_y: float
    intensity: float = 0.0
    source: str = "tracking"


@dataclass(frozen=True)
class TrackedSpotIdentity:
    track_id: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class TrackingResult:
    tracks: tuple[TrackedSpotIdentity, ...]


@dataclass(frozen=True)
class TrackingProgress:
    completed: int
    total: int
    frame_id: int


@dataclass(frozen=True)
class TrackingPreflightIssue:
    code: str
    message: str


@dataclass(frozen=True)
class TrackingPreflightReport:
    issues: tuple[TrackingPreflightIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.issues


class SeededTrackingRuntime(Protocol):
    def preflight(
        self,
        config: SeededTrackingConfig,
    ) -> tuple[TrackingPreflightIssue, ...]: ...

    def run(
        self,
        request: SeededTrackingRequest,
        context: RunContext,
    ) -> TrackingResult: ...


@dataclass(frozen=True)
class SeededTrackingCapabilities:
    seed_representation: SeedRepresentation
    supports_background: bool
    preserves_gaps: bool


@dataclass(frozen=True)
class SeededTrackingMethod:
    method_id: str
    display_name: str
    family: TrackingFamily
    capabilities: SeededTrackingCapabilities
    runtime: SeededTrackingRuntime


class SeededTrackingCatalog:
    def __init__(self, methods: tuple[SeededTrackingMethod, ...]) -> None:
        self._methods = methods
        method_ids = tuple(method.method_id for method in methods)
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("Seeded-tracking method IDs must be unique")

    @property
    def methods(self) -> tuple[SeededTrackingMethod, ...]:
        return self._methods

    def method(self, method_id: str) -> SeededTrackingMethod:
        for method in self._methods:
            if method.method_id == method_id:
                return method
        raise KeyError(method_id)

    def preflight(self, request: SeededTrackingRequest) -> TrackingPreflightReport:
        issues = list(_structural_preflight(request, self))
        try:
            runtime = self.method(request.method_id).runtime
        except KeyError:
            return TrackingPreflightReport(tuple(issues))
        issues.extend(runtime.preflight(request.config))
        return TrackingPreflightReport(tuple(issues))


@dataclass(frozen=True)
class RunContext:
    image_reader: Callable[[Path], np.ndarray]
    report_progress: Callable[[TrackingProgress], None] = lambda _progress: None
    is_cancelled: Callable[[], bool] = lambda: False
    tracking_catalog: SeededTrackingCatalog | None = None


class SeededTrackingCancelled(RuntimeError):
    pass


class SeededTrackingPreflightError(RuntimeError):
    def __init__(self, report: TrackingPreflightReport) -> None:
        super().__init__("Seeded tracking is not ready")
        self.report = report


class SeededTrackingExecutionError(RuntimeError):
    pass


def default_seeded_tracking_catalog() -> SeededTrackingCatalog:
    from trackleed.analysis.seeded_tracking_runtime import (
        OpticalFlowRuntime,
        SkipAnchorOpticalFlowRuntime,
    )

    return SeededTrackingCatalog(
        (
            SeededTrackingMethod(
                "optical_flow",
                "Optical Flow Tracking",
                TrackingFamily.CLASSICAL,
                SeededTrackingCapabilities(
                    SeedRepresentation.POINT,
                    supports_background=True,
                    preserves_gaps=False,
                ),
                OpticalFlowRuntime(),
            ),
            SeededTrackingMethod(
                "optical_flow_skip_anchor",
                "Optical Flow (skip, anchor)",
                TrackingFamily.CLASSICAL,
                SeededTrackingCapabilities(
                    SeedRepresentation.POINT,
                    supports_background=True,
                    preserves_gaps=True,
                ),
                SkipAnchorOpticalFlowRuntime(),
            ),
        )
    )


def run_seeded_tracking(
    request: SeededTrackingRequest,
    context: RunContext,
) -> TrackingResult:
    """Validate and run one seeded-tracking request without publishing it."""
    catalog = context.tracking_catalog or default_seeded_tracking_catalog()
    report = catalog.preflight(request)
    if not report.ready:
        raise SeededTrackingPreflightError(report)
    if context.is_cancelled():
        raise SeededTrackingCancelled("Seeded tracking cancelled")
    try:
        return catalog.method(request.method_id).runtime.run(request, context)
    except SeededTrackingCancelled:
        raise
    except Exception as exc:
        raise SeededTrackingExecutionError(str(exc)) from exc


def _structural_preflight(
    request: SeededTrackingRequest,
    catalog: SeededTrackingCatalog,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    methods_by_id = {method.method_id: method for method in catalog.methods}
    method = methods_by_id.get(request.method_id)
    if method is None:
        issues.append(TrackingPreflightIssue("unknown_method", "Unknown method ID."))
    elif request.seed_representation is not method.capabilities.seed_representation:
        issues.append(
            TrackingPreflightIssue(
                "unsupported_seed_representation",
                "Requested seed representation is not supported by this method.",
            )
        )
    issues.extend(_frame_preflight_issues(request))
    issues.extend(_seed_preflight_issues(request))
    issues.extend(_background_preflight_issues(request, method))
    return tuple(issues)


def _frame_preflight_issues(
    request: SeededTrackingRequest,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    frame_ids = tuple(frame.frame_id for frame in request.frames)
    if len(frame_ids) < 2:
        issues.append(
            TrackingPreflightIssue(
                "frames_required", "At least two frames are required."
            )
        )
    if len(set(frame_ids)) != len(frame_ids):
        issues.append(
            TrackingPreflightIssue("duplicate_frame_id", "Frame IDs must be unique.")
        )
    if request.reference_frame_id not in frame_ids:
        issues.append(
            TrackingPreflightIssue(
                "reference_required",
                "Reference frame must belong to the request.",
            )
        )
    elif frame_ids.index(request.reference_frame_id) == len(frame_ids) - 1:
        issues.append(
            TrackingPreflightIssue(
                "following_frame_required",
                "At least one frame must follow the reference frame.",
            )
        )
    return tuple(issues)


def _seed_preflight_issues(
    request: SeededTrackingRequest,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    seed_ids = tuple(seed.seed_id for seed in request.seeds)
    if not seed_ids:
        issues.append(
            TrackingPreflightIssue("seeds_required", "Select at least one seed.")
        )
    elif len(set(seed_ids)) != len(seed_ids):
        issues.append(
            TrackingPreflightIssue("duplicate_seed_id", "Seed IDs must be unique.")
        )
    if any(
        not isfinite(seed.center_x)
        or not isfinite(seed.center_y)
        or seed.roi.width <= 0
        or seed.roi.height <= 0
        for seed in request.seeds
    ):
        issues.append(
            TrackingPreflightIssue(
                "invalid_seed",
                "Seed centers must be finite and seed ROIs must have positive size.",
            )
        )
    return tuple(issues)


def _background_preflight_issues(
    request: SeededTrackingRequest,
    method: SeededTrackingMethod | None,
) -> tuple[TrackingPreflightIssue, ...]:
    issues: list[TrackingPreflightIssue] = []
    if request.config.subtract_background and request.background_path is None:
        issues.append(
            TrackingPreflightIssue(
                "background_required",
                "Load a background image before subtracting it.",
            )
        )
    elif (
        request.config.subtract_background
        and method is not None
        and not method.capabilities.supports_background
    ):
        issues.append(
            TrackingPreflightIssue(
                "background_not_supported",
                "This tracking method does not support background subtraction.",
            )
        )
    return tuple(issues)
