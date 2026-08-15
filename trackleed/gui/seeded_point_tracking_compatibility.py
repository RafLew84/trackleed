"""Qt/model compatibility boundary for headless point tracking workflows."""

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
from trackleed.logic.workflows.point_tracking import PointTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingRequest,
    TrackingResult,
)


@dataclass(frozen=True)
class PointTrackerLegacyBinding:
    method_id: str
    tracking_runner_name: str
    retrack_runner_name: str
    models_param_name: str
    review_argument_name: str
    repo_param_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PointReviewSpec:
    tracker_id: str
    has_visualizer_retrack: bool = True


_POINT_BINDINGS = (
    PointTrackerLegacyBinding(
        "tapir_offline", "track_spots_tapir_offline",
        "retrack_single_spot_tapir_offline", "tapir_models_dir", "tapir_params",
    ),
    PointTrackerLegacyBinding(
        "tapir_online", "track_spots_tapir_online",
        "retrack_single_spot_tapir_online", "tapir_models_dir", "tapir_params",
    ),
    PointTrackerLegacyBinding(
        "locotrack_offline", "track_spots_locotrack_offline",
        "retrack_single_spot_locotrack_offline", "locotrack_models_dir",
        "locotrack_params", ("locotrack_repo_path",),
    ),
    PointTrackerLegacyBinding(
        "cotracker3_offline", "track_spots_cotracker3_offline",
        "retrack_single_spot_cotracker3_offline", "cotracker3_models_dir",
        "cotracker_params", ("cotracker3_repo_path", "cotracker_repo_path"),
    ),
    PointTrackerLegacyBinding(
        "tapnext_offline", "track_spots_tapnext_offline",
        "retrack_single_spot_tapnext_offline", "tapnext_models_dir",
        "tapnext_params", ("tapnext_repo_path",),
    ),
    PointTrackerLegacyBinding(
        "trackonr_offline", "track_spots_trackonr_offline",
        "retrack_single_spot_trackonr_offline", "trackonr_models_dir",
        "trackonr_params", ("trackonr_repo_path", "track_on_repo_path"),
    ),
    PointTrackerLegacyBinding(
        "pipspp_offline", "track_spots_pipspp_offline",
        "retrack_single_spot_pipspp_offline", "pipspp_models_dir",
        "pipspp_params", ("pipspp_repo_path",),
    ),
    PointTrackerLegacyBinding(
        "mftiq_offline", "track_spots_mftiq_offline",
        "retrack_single_spot_mftiq_offline", "mftiq_models_dir",
        "mftiq_params", ("mftiq_repo_path",),
    ),
)


def point_tracker_legacy_bindings() -> tuple[PointTrackerLegacyBinding, ...]:
    return _POINT_BINDINGS


def point_tracker_legacy_binding(method_id: str) -> PointTrackerLegacyBinding:
    normalized = str(method_id).strip().lower()
    for binding in _POINT_BINDINGS:
        if binding.method_id == normalized:
            return binding
    raise KeyError(f"Unknown point tracker ID: {method_id}")


def point_review_spec_for_source(source_name: str) -> PointReviewSpec | None:
    normalized = str(source_name).strip().lower()
    try:
        binding = point_tracker_legacy_binding(normalized)
    except KeyError:
        return None
    return PointReviewSpec(binding.method_id)


def legacy_point_tracker_params(
    method_id: str,
    config: PointTrackerConfig,
) -> dict[str, Any]:
    binding = point_tracker_legacy_binding(method_id)
    params = {
        key: plain_legacy_value(value) for key, value in asdict(config).items()
    }
    models_dir = params.pop("models_dir")
    repo_path = params.get("repo_path")
    params[binding.models_param_name] = models_dir
    for name in binding.repo_param_names:
        params[name] = repo_path
    return params


def run_legacy_point_tracker(
    method_id: str,
    request: SeededTrackingRequest,
    context: RunContext,
    *,
    runner: Callable[..., list[Track] | None] | None = None,
) -> TrackingResult:
    """Run one existing backend behind the GUI/model conversion boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, PointTrackerConfig):
        raise TypeError("Point tracking requires PointTrackerConfig")
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
        legacy_point_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if tracks is None:
        raise_none_result(method_id, context)
    return neutral_result(tracks, frame_ids, family_label="Point")


def run_legacy_point_retracker(
    method_id: str,
    request: SeededRetrackRequest,
    context: RunContext,
    *,
    runner: Callable[..., Track | None] | None = None,
) -> TrackingResult:
    """Run one existing re-tracker behind the GUI/model conversion boundary."""
    raise_if_cancelled(context)
    if not isinstance(request.config, PointTrackerConfig):
        raise TypeError("Point re-tracking requires PointTrackerConfig")
    files_by_id = legacy_files(request.frames, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=True)
    track = selected_runner(
        legacy_track(request.track),
        request.start_frame_id,
        files_by_id,
        frame_ids,
        legacy_point_tracker_params(method_id, request.config),
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )
    if track is None:
        raise_none_result(method_id, context)
    return neutral_result([track], frame_ids, family_label="Point")


def _legacy_runner(method_id: str, *, retrack: bool) -> Callable[..., Any]:
    binding = point_tracker_legacy_binding(method_id)
    module_name = (
        "trackleed.analysis.tracking_retrack"
        if retrack
        else "trackleed.analysis.tracking_point_backends"
    )
    function_name = (
        binding.retrack_runner_name if retrack else binding.tracking_runner_name
    )
    return getattr(import_module(module_name), function_name)
