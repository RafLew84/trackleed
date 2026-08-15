"""Compatibility boundary between neutral point tracking and legacy backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.workflows.point_tracking import PointTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededTrackingCancelled,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingProgress,
    TrackingResult,
    TrackObservation,
)

_RUNNER_NAMES = {
    "tapir_offline": "track_spots_tapir_offline",
    "tapir_online": "track_spots_tapir_online",
    "locotrack_offline": "track_spots_locotrack_offline",
    "cotracker3_offline": "track_spots_cotracker3_offline",
    "tapnext_offline": "track_spots_tapnext_offline",
    "trackonr_offline": "track_spots_trackonr_offline",
    "pipspp_offline": "track_spots_pipspp_offline",
    "mftiq_offline": "track_spots_mftiq_offline",
}

_MODELS_PARAM_NAMES = {
    "tapir_offline": "tapir_models_dir",
    "tapir_online": "tapir_models_dir",
    "locotrack_offline": "locotrack_models_dir",
    "cotracker3_offline": "cotracker3_models_dir",
    "tapnext_offline": "tapnext_models_dir",
    "trackonr_offline": "trackonr_models_dir",
    "pipspp_offline": "pipspp_models_dir",
    "mftiq_offline": "mftiq_models_dir",
}

_REPO_PARAM_NAMES = {
    "locotrack_offline": ("locotrack_repo_path",),
    "cotracker3_offline": ("cotracker3_repo_path", "cotracker_repo_path"),
    "tapnext_offline": ("tapnext_repo_path",),
    "trackonr_offline": ("trackonr_repo_path", "track_on_repo_path"),
    "pipspp_offline": ("pipspp_repo_path",),
    "mftiq_offline": ("mftiq_repo_path",),
}


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
    """Run one existing backend behind a narrow model-conversion boundary."""
    if context.is_cancelled():
        raise SeededTrackingCancelled("Seeded tracking cancelled")
    if not isinstance(request.config, PointTrackerConfig):
        raise TypeError("Point tracking requires PointTrackerConfig")
    files_by_id = _legacy_files(request, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    selected_runner = runner or _legacy_runner(method_id)
    tracks = selected_runner(
        files_by_id,
        frame_ids,
        request.reference_frame_id,
        legacy_point_tracker_params(method_id, request.config),
        progress_callback=lambda value: context.report_progress(
            TrackingProgress(int(value), 100, frame_ids[-1])
        ),
        cancel_event=_ContextCancellationSignal(context),
    )
    if tracks is None:
        if context.is_cancelled():
            raise SeededTrackingCancelled("Seeded tracking cancelled")
        raise RuntimeError(f"{method_id} returned no tracking result")
    return _neutral_result(tracks, frame_ids)


def _legacy_runner(method_id: str) -> Callable[..., list[Track] | None]:
    try:
        runner_name = _RUNNER_NAMES[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown point tracker ID: {method_id}") from exc
    module = import_module("trackleed.analysis.tracking_point_backends")
    return getattr(module, runner_name)


def _legacy_files(
    request: SeededTrackingRequest,
    context: RunContext,
) -> dict[int, LeedFile]:
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
        for frame in request.frames
    }
    for file in files.values():
        file._seeded_image_reader = context.image_reader
    files[request.reference_frame_id].spots = [
        SpotROI(
            seed.seed_id,
            QRect(seed.roi.x, seed.roi.y, seed.roi.width, seed.roi.height),
            seed.center_x,
            seed.center_y,
            seed.intensity,
            seed.source,
        )
        for seed in request.seeds
    ]
    return files


def legacy_point_tracker_params(
    method_id: str,
    config: PointTrackerConfig,
) -> dict[str, Any]:
    params = {key: _plain_value(value) for key, value in asdict(config).items()}
    models_dir = params.pop("models_dir")
    repo_path = params.get("repo_path")
    params[_MODELS_PARAM_NAMES[method_id]] = models_dir
    for name in _REPO_PARAM_NAMES.get(method_id, ()):
        params[name] = repo_path
    return params


def _plain_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    return value


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
