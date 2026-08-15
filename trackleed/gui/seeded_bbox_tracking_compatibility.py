"""Qt/model compatibility boundary for bounding-box tracking workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from trackleed.core.data_models import Track
from trackleed.gui.seeded_tracking_legacy_models import (
    ContextCancellationSignal,
    legacy_files,
    legacy_spot,
    legacy_track,
    neutral_result,
    plain_legacy_value,
    progress_callback,
    raise_if_cancelled,
    raise_none_result,
)
from trackleed.logic.workflows.bbox_tracking import BoundingBoxTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingRequest,
    TrackingResult,
)


@dataclass(frozen=True)
class BboxTrackerLegacyBinding:
    method_id: str
    tracking_runner_name: str
    retrack_runner_name: str
    review_argument_name: str
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class BboxReviewSpec:
    tracker_id: str
    has_visualizer_retrack: bool = True


_BBOX_BINDINGS = (
    BboxTrackerLegacyBinding(
        "tomp_offline", "track_spots_tomp_offline",
        "retrack_single_spot_tomp_offline", "tomp_params",
        ("tomp_offline", "tamos_offline"),
    ),
    BboxTrackerLegacyBinding(
        "ostrack_offline", "track_spots_ostrack_offline",
        "retrack_single_spot_ostrack_offline", "ostrack_params",
        ("ostrack_offline",),
    ),
)


def bbox_tracker_legacy_bindings() -> tuple[BboxTrackerLegacyBinding, ...]:
    return _BBOX_BINDINGS


def bbox_tracker_legacy_binding(method_id: str) -> BboxTrackerLegacyBinding:
    normalized = str(method_id).strip().lower()
    for binding in _BBOX_BINDINGS:
        if binding.method_id == normalized:
            return binding
    raise KeyError(f"Unknown bounding-box tracker ID: {method_id}")


def bbox_review_spec_for_source(source_name: str) -> BboxReviewSpec | None:
    normalized = str(source_name).strip().lower()
    for binding in _BBOX_BINDINGS:
        if normalized in binding.source_names:
            return BboxReviewSpec(binding.method_id)
    return None


def legacy_bbox_tracker_params(
    method_id: str,
    config: BoundingBoxTrackerConfig,
) -> dict[str, Any]:
    params = {
        key: plain_legacy_value(value) for key, value in asdict(config).items()
    }
    repo_path = params.get("repo_path")
    if method_id == "tomp_offline":
        params["tomp_repo_path"] = repo_path
        params["pytracking_repo_path"] = repo_path
    elif method_id == "ostrack_offline":
        params["ostrack_models_dir"] = params.pop("models_dir")
        params["ostrack_repo_path"] = repo_path
    else:
        raise KeyError(f"Unknown bounding-box tracker ID: {method_id}")
    return params


def run_legacy_bbox_tracker(
    method_id: str,
    request: SeededTrackingRequest,
    context: RunContext,
    *,
    runner: Callable[..., list[Track] | None] | None = None,
) -> TrackingResult:
    """Run an existing bounding-box backend behind the GUI/model boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, BoundingBoxTrackerConfig):
        raise TypeError("Bounding-box tracking requires its typed configuration")
    files_by_id = legacy_files(request.frames, context)
    files_by_id[request.reference_frame_id].spots = [
        legacy_spot(
            seed.seed_id,
            seed.roi,
            seed.center_x,
            seed.center_y,
            seed.intensity,
            seed.source,
        )
        for seed in request.seeds
    ]
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=False)
    tracks = selected_runner(
        files_by_id,
        frame_ids,
        request.reference_frame_id,
        legacy_bbox_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if tracks is None:
        raise_none_result(method_id, context)
    return neutral_result(tracks, frame_ids, family_label="Bounding-box")


def run_legacy_bbox_retracker(
    method_id: str,
    request: SeededRetrackRequest,
    context: RunContext,
    *,
    runner: Callable[..., Track | None] | None = None,
) -> TrackingResult:
    """Run an existing bounding-box re-tracker at the GUI/model boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, BoundingBoxTrackerConfig):
        raise TypeError("Bounding-box re-tracking requires its typed configuration")
    files_by_id = legacy_files(request.frames, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=True)
    track = selected_runner(
        legacy_track(request.track),
        request.start_frame_id,
        files_by_id,
        frame_ids,
        legacy_bbox_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if track is None:
        raise_none_result(method_id, context)
    return neutral_result([track], frame_ids, family_label="Bounding-box")


def _legacy_runner(method_id: str, *, retrack: bool) -> Callable[..., Any]:
    binding = bbox_tracker_legacy_binding(method_id)
    module_name = (
        "trackleed.analysis.tracking_retrack"
        if retrack
        else "trackleed.analysis.tracking_bbox_backends"
    )
    function_name = (
        binding.retrack_runner_name if retrack else binding.tracking_runner_name
    )
    return getattr(import_module(module_name), function_name)
