"""Qt/model compatibility boundary for headless point tracking workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.workflows.point_tracking import PointTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededRetrackRequest,
    SeededTrackingCancelled,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingProgress,
    TrackingResult,
    TrackObservation,
)


@dataclass(frozen=True)
class PointTrackerLegacyBinding:
    method_id: str
    tracking_runner_name: str
    retrack_runner_name: str
    models_param_name: str
    review_argument_name: str
    settings_type_name: str
    runtime_settings_type_name: str
    collect_params_method_name: str
    settings_attr_name: str
    panel_group_attr: str
    warning_label_attr: str
    checkpoint_selector_attr: str
    repo_param_names: tuple[str, ...] = ()
    backend_mode_widget_attr: str | None = None
    repo_path_attr: str | None = None
    python_executable_attr: str | None = None
    default_conda_env_name: str | None = None


@dataclass(frozen=True)
class PointReviewSpec:
    tracker_id: str
    has_visualizer_retrack: bool = True


_POINT_BINDINGS = (
    PointTrackerLegacyBinding(
        "tapir_offline",
        "track_spots_tapir_offline",
        "retrack_single_spot_tapir_offline",
        "tapir_models_dir",
        "tapir_params",
        "TapirTrackingSettings",
        "TapirRuntimeSettings",
        "_collect_tapir_tracking_params",
        "tapir_settings",
        "tapir_group",
        "tapir_warning_label",
        "tapir_checkpoint_combo",
    ),
    PointTrackerLegacyBinding(
        "tapir_online",
        "track_spots_tapir_online",
        "retrack_single_spot_tapir_online",
        "tapir_models_dir",
        "tapir_params",
        "TapirTrackingSettings",
        "TapirOnlineRuntimeSettings",
        "_collect_tapir_tracking_params",
        "tapir_settings",
        "tapir_group",
        "tapir_warning_label",
        "tapir_checkpoint_combo",
    ),
    PointTrackerLegacyBinding(
        "locotrack_offline",
        "track_spots_locotrack_offline",
        "retrack_single_spot_locotrack_offline",
        "locotrack_models_dir",
        "locotrack_params",
        "LocotrackTrackingSettings",
        "LocotrackRuntimeSettings",
        "_collect_locotrack_tracking_params",
        "locotrack_settings",
        "locotrack_group",
        "locotrack_warning_label",
        "locotrack_checkpoint_combo",
        ("locotrack_repo_path",),
        "locotrack_backend_mode_combo",
        "locotrack_repo_path_edit",
        "locotrack_python_executable_edit",
        "locotrack-pytorch",
    ),
    PointTrackerLegacyBinding(
        "cotracker3_offline",
        "track_spots_cotracker3_offline",
        "retrack_single_spot_cotracker3_offline",
        "cotracker3_models_dir",
        "cotracker_params",
        "CotrackerTrackingSettings",
        "CotrackerRuntimeSettings",
        "_collect_cotracker_tracking_params",
        "cotracker_settings",
        "cotracker_group",
        "cotracker_warning_label",
        "cotracker_checkpoint_combo",
        ("cotracker3_repo_path", "cotracker_repo_path"),
    ),
    PointTrackerLegacyBinding(
        "tapnext_offline",
        "track_spots_tapnext_offline",
        "retrack_single_spot_tapnext_offline",
        "tapnext_models_dir",
        "tapnext_params",
        "TapnextTrackingSettings",
        "TapnextRuntimeSettings",
        "_collect_tapnext_tracking_params",
        "tapnext_settings",
        "tapnext_group",
        "tapnext_warning_label",
        "tapnext_checkpoint_combo",
        ("tapnext_repo_path",),
    ),
    PointTrackerLegacyBinding(
        "trackonr_offline",
        "track_spots_trackonr_offline",
        "retrack_single_spot_trackonr_offline",
        "trackonr_models_dir",
        "trackonr_params",
        "TrackonrTrackingSettings",
        "TrackonrRuntimeSettings",
        "_collect_trackonr_tracking_params",
        "trackonr_settings",
        "trackonr_group",
        "trackonr_warning_label",
        "trackonr_checkpoint_combo",
        ("trackonr_repo_path", "track_on_repo_path"),
        "trackonr_backend_mode_combo",
        "trackonr_repo_path_edit",
        "trackonr_python_executable_edit",
        "trackonr2",
    ),
    PointTrackerLegacyBinding(
        "pipspp_offline",
        "track_spots_pipspp_offline",
        "retrack_single_spot_pipspp_offline",
        "pipspp_models_dir",
        "pipspp_params",
        "PipsppTrackingSettings",
        "PipsppRuntimeSettings",
        "_collect_pipspp_tracking_params",
        "pipspp_settings",
        "pipspp_group",
        "pipspp_warning_label",
        "pipspp_checkpoint_combo",
        ("pipspp_repo_path",),
        "pipspp_backend_mode_combo",
        "pipspp_repo_path_edit",
        "pipspp_python_executable_edit",
        "pipspp",
    ),
    PointTrackerLegacyBinding(
        "mftiq_offline",
        "track_spots_mftiq_offline",
        "retrack_single_spot_mftiq_offline",
        "mftiq_models_dir",
        "mftiq_params",
        "MftiqTrackingSettings",
        "MftiqRuntimeSettings",
        "_collect_mftiq_tracking_params",
        "mftiq_settings",
        "mftiq_group",
        "mftiq_warning_label",
        "mftiq_checkpoint_combo",
        ("mftiq_repo_path",),
        "mftiq_backend_mode_combo",
        "mftiq_repo_path_edit",
        "mftiq_python_executable_edit",
        "mftiq",
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


def point_tracker_gui_spec(method_id: str) -> Any:
    """Expose legacy settings-panel metadata without re-registering point methods."""
    from trackleed.analysis.seeded_point_tracking_runtime import (
        point_tracking_definition,
    )

    definition = point_tracking_definition(method_id)
    binding = point_tracker_legacy_binding(method_id)
    backend = import_module(definition.backend_module)
    formatter_name = definition.probe_name.replace(
        "probe_",
        "format_",
    ).replace("_environment", "_environment_report")
    tracking_module = import_module("trackleed.analysis.tracking_point_backends")
    retrack_module = import_module("trackleed.analysis.tracking_retrack")
    return SimpleNamespace(
        tracker_id=definition.method_id,
        display_name=definition.display_name,
        scenario_name=definition.display_name,
        family="point",
        source_names=(definition.method_id,),
        has_visualizer_retrack=True,
        params_argument_name=binding.review_argument_name,
        supports_direct=True,
        supports_subprocess=definition.supports_subprocess,
        probe_environment=getattr(backend, definition.probe_name),
        format_environment_report=getattr(backend, formatter_name),
        run_tracking=getattr(tracking_module, binding.tracking_runner_name),
        run_retrack=getattr(retrack_module, binding.retrack_runner_name),
        **{
            field: getattr(binding, field)
            for field in (
                "settings_type_name",
                "runtime_settings_type_name",
                "collect_params_method_name",
                "settings_attr_name",
                "panel_group_attr",
                "warning_label_attr",
                "backend_mode_widget_attr",
                "repo_path_attr",
                "python_executable_attr",
                "default_conda_env_name",
                "checkpoint_selector_attr",
            )
        },
    )


def legacy_point_tracker_params(
    method_id: str,
    config: PointTrackerConfig,
) -> dict[str, Any]:
    binding = point_tracker_legacy_binding(method_id)
    params = {key: _plain_value(value) for key, value in asdict(config).items()}
    models_dir = params.pop("models_dir")
    repo_path = params.get("repo_path")
    params[binding.models_param_name] = models_dir
    for name in binding.repo_param_names:
        params[name] = repo_path
    return params


def _plain_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    return value


class _ContextCancellationSignal:
    def __init__(self, context: RunContext) -> None:
        self._context = context

    def is_set(self) -> bool:
        return self._context.is_cancelled()


def run_legacy_point_tracker(
    method_id: str,
    request: SeededTrackingRequest,
    context: RunContext,
    *,
    runner: Callable[..., list[Track] | None] | None = None,
) -> TrackingResult:
    """Run one existing backend behind the GUI/model conversion boundary."""
    _raise_if_cancelled(context)
    if not isinstance(request.config, PointTrackerConfig):
        raise TypeError("Point tracking requires PointTrackerConfig")
    files_by_id = _legacy_files(request.frames, context)
    files_by_id[request.reference_frame_id].spots = [
        _legacy_spot(
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
        progress_callback=_progress_callback(context, frame_ids[-1]),
        cancel_event=_ContextCancellationSignal(context),
    )
    if tracks is None:
        _raise_none_result(method_id, context)
    return _neutral_result(tracks, frame_ids)


def run_legacy_point_retracker(
    method_id: str,
    request: SeededRetrackRequest,
    context: RunContext,
    *,
    runner: Callable[..., Track | None] | None = None,
) -> TrackingResult:
    """Run one existing re-tracker behind the GUI/model conversion boundary."""
    _raise_if_cancelled(context)
    if not isinstance(request.config, PointTrackerConfig):
        raise TypeError("Point re-tracking requires PointTrackerConfig")
    files_by_id = _legacy_files(request.frames, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id, retrack=True)
    track = selected_runner(
        _legacy_track(request.track),
        request.start_frame_id,
        files_by_id,
        frame_ids,
        legacy_point_tracker_params(method_id, request.config),
        progress_callback=_progress_callback(context, frame_ids[-1]),
        cancel_event=_ContextCancellationSignal(context),
    )
    if track is None:
        _raise_none_result(method_id, context)
    return _neutral_result([track], frame_ids)


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


def _legacy_files(frames, context: RunContext) -> dict[int, LeedFile]:
    files = {
        frame.frame_id: LeedFile(
            filepath=frame.source_path,
            filename=frame.source_path.name,
            is_background=False,
            element="",
            surface=0,
            mode="",
            temperature=0.0,
            energy=0.0,
            id=frame.frame_id,
            coverage=None,
        )
        for frame in frames
    }
    for file in files.values():
        file._seeded_image_reader = context.image_reader
    return files


def _legacy_track(identity: TrackedSpotIdentity) -> Track:
    return Track(
        identity.track_id,
        {
            item.frame_id: _legacy_spot(
                identity.track_id,
                item.roi,
                item.center_x,
                item.center_y,
                item.intensity,
                item.source,
            )
            for item in identity.observations
        },
    )


def _legacy_spot(
    spot_id: int,
    roi: SeedRect,
    center_x: float,
    center_y: float,
    intensity: float,
    source: str,
) -> SpotROI:
    return SpotROI(
        spot_id,
        QRect(roi.x, roi.y, roi.width, roi.height),
        center_x,
        center_y,
        intensity,
        source,
    )


def _neutral_result(tracks: list[Track], frame_ids: list[int]) -> TrackingResult:
    known_frame_ids = set(frame_ids)
    identities = []
    for track in tracks:
        unknown = set(track.spots_by_image_id) - known_frame_ids
        if unknown:
            raise ValueError(f"Point tracker returned unknown frame IDs: {unknown}")
        observations = tuple(
            _neutral_observation(frame_id, track.spots_by_image_id[frame_id])
            for frame_id in frame_ids
            if frame_id in track.spots_by_image_id
        )
        identities.append(TrackedSpotIdentity(track.track_id, observations))
    return TrackingResult(tuple(identities))


def _neutral_observation(frame_id: int, spot: SpotROI) -> TrackObservation:
    rect = spot.roi_rect
    return TrackObservation(
        frame_id,
        SeedRect(rect.x(), rect.y(), rect.width(), rect.height()),
        float(spot.center_x),
        float(spot.center_y),
        float(spot.intensity),
        str(spot.source),
    )


def _progress_callback(context: RunContext, frame_id: int) -> Callable[[int], None]:
    return lambda value: context.report_progress(
        TrackingProgress(int(value), 100, frame_id)
    )


def _raise_if_cancelled(context: RunContext) -> None:
    if context.is_cancelled():
        raise SeededTrackingCancelled("Seeded tracking cancelled")


def _raise_none_result(method_id: str, context: RunContext) -> Never:
    _raise_if_cancelled(context)
    raise RuntimeError(f"{method_id} returned no tracking result")
