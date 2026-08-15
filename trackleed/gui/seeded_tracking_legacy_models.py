"""Shared model-conversion seam for legacy seeded-tracker runners."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    SeededTrackingCancelled,
    SeedRect,
    TrackedSpotIdentity,
    TrackingProgress,
    TrackingResult,
    TrackObservation,
)

_GUI_BINDING_FIELDS = (
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


def legacy_tracker_gui_spec(
    definition: Any,
    binding: Any,
    *,
    family: str,
    source_names: tuple[str, ...],
    tracking_module_name: str,
    extra_binding_fields: tuple[str, ...] = (),
) -> Any:
    """Build transitional panel metadata from one canonical definition."""
    backend = import_module(definition.backend_module)
    formatter_name = definition.probe_name.replace(
        "probe_",
        "format_",
    ).replace("_environment", "_environment_report")
    tracking_module = import_module(tracking_module_name)
    retrack_module = import_module("trackleed.analysis.tracking_retrack")
    return SimpleNamespace(
        tracker_id=definition.method_id,
        display_name=definition.display_name,
        scenario_name=definition.display_name,
        family=family,
        source_names=source_names,
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
            for field in _GUI_BINDING_FIELDS + extra_binding_fields
        },
    )


def plain_legacy_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(plain_legacy_value(item) for item in value)
    return value


class ContextCancellationSignal:
    def __init__(self, context: RunContext) -> None:
        self._context = context

    def is_set(self) -> bool:
        return self._context.is_cancelled()


def legacy_files(frames: Iterable[Any], context: RunContext) -> dict[int, LeedFile]:
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
        file._seeded_image_reader = context.image_reader  # type: ignore[attr-defined]
    return files


def legacy_track(identity: TrackedSpotIdentity) -> Track:
    return Track(
        identity.track_id,
        {
            item.frame_id: legacy_spot(
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


def legacy_spot(
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


def neutral_result(
    tracks: list[Track],
    frame_ids: list[int],
    *,
    family_label: str,
) -> TrackingResult:
    known_frame_ids = set(frame_ids)
    identities = []
    for track in tracks:
        unknown = set(track.spots_by_image_id) - known_frame_ids
        if unknown:
            raise ValueError(
                f"{family_label} tracker returned unknown frame IDs: {unknown}"
            )
        observations = tuple(
            neutral_observation(frame_id, track.spots_by_image_id[frame_id])
            for frame_id in frame_ids
            if frame_id in track.spots_by_image_id
        )
        identities.append(TrackedSpotIdentity(track.track_id, observations))
    return TrackingResult(tuple(identities))


def neutral_observation(frame_id: int, spot: SpotROI) -> TrackObservation:
    rect = spot.roi_rect
    return TrackObservation(
        frame_id,
        SeedRect(rect.x(), rect.y(), rect.width(), rect.height()),
        float(spot.center_x),
        float(spot.center_y),
        float(spot.intensity),
        str(spot.source),
    )


def progress_callback(context: RunContext, frame_id: int) -> Callable[[int], None]:
    return lambda value: context.report_progress(
        TrackingProgress(int(value), 100, frame_id)
    )


def raise_if_cancelled(context: RunContext) -> None:
    if context.is_cancelled():
        raise SeededTrackingCancelled("Seeded tracking cancelled")


def raise_none_result(method_id: str, context: RunContext) -> Never:
    raise_if_cancelled(context)
    raise RuntimeError(f"{method_id} returned no tracking result")
