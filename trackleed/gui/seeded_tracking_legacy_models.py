"""Shared model-conversion seam for legacy seeded-tracker runners."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Never, Protocol, cast

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.workflows.seeded_tracking import (
    RunContext,
    DiagnosticValue,
    SeededTrackingCancelled,
    SeededRetrackRequest,
    SeededTrackingRequest,
    SeedRect,
    TrackedSpotIdentity,
    TrackingProgress,
    TrackingResult,
    TrackingSeed,
    TrackObservation,
)

_SPOT_MODEL_FIELDS = frozenset(field.name for field in fields(SpotROI))


class LegacyRunnerResolver(Protocol):
    def __call__(
        self,
        method_id: str,
        *,
        retrack: bool,
    ) -> Callable[..., object]: ...


@dataclass(frozen=True)
class LegacySeededFamily:
    label: str
    parameters: Callable[[str, Any], dict[str, Any]]
    seed: Callable[[TrackingSeed], SpotROI]
    resolve_runner: LegacyRunnerResolver


def run_legacy_seeded(
    method_id: str,
    request: SeededTrackingRequest | SeededRetrackRequest,
    context: RunContext,
    family: LegacySeededFamily,
    *,
    runner: Callable[..., object] | None = None,
) -> TrackingResult:
    """Execute one legacy full run or re-track behind a neutral boundary."""
    raise_if_cancelled(context)
    params = family.parameters(method_id, request.config)
    files_by_id = legacy_files(request.frames, context)
    frame_ids = [frame.frame_id for frame in request.frames]
    is_retrack = isinstance(request, SeededRetrackRequest)
    selected_runner = runner or family.resolve_runner(
        method_id,
        retrack=is_retrack,
    )
    raw_result = _invoke_legacy_runner(
        request,
        files_by_id,
        frame_ids,
        params,
        context,
        family,
        selected_runner,
    )
    if raw_result is None:
        raise_none_result(method_id, context)
    raise_if_cancelled(context)
    tracks = (
        [cast(Track, raw_result)]
        if is_retrack
        else cast(list[Track], raw_result)
    )
    return neutral_result(tracks, frame_ids, family_label=family.label)


def _invoke_legacy_runner(
    request: SeededTrackingRequest | SeededRetrackRequest,
    files_by_id: dict[int, LeedFile],
    frame_ids: list[int],
    params: dict[str, Any],
    context: RunContext,
    family: LegacySeededFamily,
    runner: Callable[..., object],
) -> object:
    if isinstance(request, SeededRetrackRequest):
        return _run_legacy_retrack(
            request, files_by_id, frame_ids, params, context, runner
        )
    return _run_legacy_full(
        request,
        files_by_id,
        frame_ids,
        params,
        context,
        family,
        runner,
    )


def _run_legacy_full(
    request: SeededTrackingRequest,
    files_by_id: dict[int, LeedFile],
    frame_ids: list[int],
    params: dict[str, Any],
    context: RunContext,
    family: LegacySeededFamily,
    runner: Callable[..., object],
) -> object:
    files_by_id[request.reference_frame_id].spots = [
        family.seed(seed) for seed in request.seeds
    ]
    return runner(
        files_by_id,
        frame_ids,
        request.reference_frame_id,
        params,
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )


def _run_legacy_retrack(
    request: SeededRetrackRequest,
    files_by_id: dict[int, LeedFile],
    frame_ids: list[int],
    params: dict[str, Any],
    context: RunContext,
    runner: Callable[..., object],
) -> object:
    return runner(
        legacy_track(request.track),
        request.start_frame_id,
        files_by_id,
        frame_ids,
        params,
        progress_callback=progress_callback(context, frame_ids[-1]),
        cancel_event=ContextCancellationSignal(context),
    )


def plain_legacy_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
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
                item.polygon_seed_points,
                item.diagnostics,
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
    polygon_seed_points: Iterable[tuple[float, float]] = (),
    diagnostics: Iterable[tuple[str, DiagnosticValue]] = (),
) -> SpotROI:
    spot = SpotROI(
        spot_id,
        QRect(roi.x, roi.y, roi.width, roi.height),
        center_x,
        center_y,
        intensity,
        source,
        [(float(x), float(y)) for x, y in polygon_seed_points],
    )
    return apply_spot_diagnostics(spot, diagnostics)


def spot_diagnostics(spot: SpotROI) -> tuple[tuple[str, DiagnosticValue], ...]:
    """Snapshot backend-specific attributes without coupling the workflow model."""
    return tuple(
        (name, _neutral_diagnostic_value(value))
        for name, value in sorted(vars(spot).items())
        if name not in _SPOT_MODEL_FIELDS
    )


def apply_spot_diagnostics(
    spot: SpotROI,
    diagnostics: Iterable[tuple[str, DiagnosticValue]],
) -> SpotROI:
    """Restore a diagnostics snapshot onto a fresh legacy spot model."""
    for name, value in diagnostics:
        if name not in _SPOT_MODEL_FIELDS:
            setattr(spot, name, value)
    return spot


def _neutral_diagnostic_value(value: object) -> DiagnosticValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _neutral_diagnostic_value(value.item())
    if isinstance(value, np.ndarray):
        return _neutral_diagnostic_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_neutral_diagnostic_value(item) for item in value)
    raise TypeError(
        f"Unsupported tracking diagnostic value: {type(value).__name__}"
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
        tuple((float(x), float(y)) for x, y in spot.polygon_seed_points),
        spot_diagnostics(spot),
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
