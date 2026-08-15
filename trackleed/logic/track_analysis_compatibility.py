"""Atomic projection from immutable fits to the legacy Track Analysis view."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import partial
from threading import Event
from typing import Optional

import numpy as np

from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.legacy_track_geometry import (
    GeometryObservationRoi,
    GeometryProgress,
    GeometryRunContext,
    LegacyCenterMode,
    LegacyGeometryCancelled,
    LegacyGeometryRequest,
    LegacyGeometryResult,
    derive_legacy_geometry,
    manual_ellipse_from_mapping,
)
from trackleed.logic.workflows.track_analysis import (
    FitCancelled,
    FitObservationInput,
    FitProgress,
    FitRequest,
    FitRunContext,
    FitSet,
    GaussianFitResult,
    Rect,
    fit,
)


LegacyFit = dict[str, object] | None
LegacyFitProjection = dict[int, dict[int, LegacyFit]]


@dataclass(frozen=True)
class AnalysisPublication:
    tracks: tuple[Track, ...]
    files: tuple[tuple[int, LeedFile], ...]
    fit_set: FitSet


def _event_is_set(cancel_event: Optional[Event]) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def _report_fit_progress(
    progress_callback: Callable[[int], None],
    progress: FitProgress,
) -> None:
    progress_callback(int(progress.completed * 60 / max(1, progress.total)))


def _report_geometry_progress(
    progress_callback: Callable[[int], None],
    progress: GeometryProgress,
) -> None:
    progress_callback(60 + int(progress.completed * 40 / max(1, progress.total)))


def run_full_analysis_publication(
    tracks_to_analyze: list[Track],
    analysis_config: dict[int, dict],
    all_files_map: dict[int, LeedFile],
    background_file: Optional[LeedFile],
    progress_callback: Callable[[int], None],
    center_mode: str = "per_frame",
    manual_ellipse: Optional[dict[str, float]] = None,
    cancel_event: Optional[Event] = None,
    previous_fit_set: FitSet | None = None,
) -> AnalysisPublication | None:
    """Run the cold workflow and publish only after every stage finishes."""
    is_cancelled = partial(_event_is_set, cancel_event)
    try:
        request = _build_fit_request(
            tracks_to_analyze,
            analysis_config,
            all_files_map,
            background_file,
            previous_fit_set,
            is_cancelled,
        )
        fit_set = fit(
            request,
            FitRunContext(
                image_reader=load_tiff,
                report_progress=partial(
                    _report_fit_progress,
                    progress_callback,
                ),
                is_cancelled=is_cancelled,
            ),
        )
        geometry = derive_legacy_geometry(
            fit_set,
            _geometry_request(tracks_to_analyze, center_mode, manual_ellipse),
            GeometryRunContext(
                report_progress=partial(
                    _report_geometry_progress,
                    progress_callback,
                ),
                is_cancelled=is_cancelled,
            ),
        )
        temporary_tracks = _tracks_with_projected_fits(
            tracks_to_analyze,
            fit_set,
            geometry,
        )
        temporary_files = _files_with_geometry(all_files_map, geometry)
        if not geometry.frames:
            progress_callback(100)
        if is_cancelled():
            raise FitCancelled("Track Analysis was cancelled before publication")
    except (FitCancelled, LegacyGeometryCancelled):
        print("[Analysis] Analysis cancelled; previous publication retained.")
        return None
    print("[Analysis] Analysis completed successfully.")
    return AnalysisPublication(
        tuple(temporary_tracks),
        tuple(temporary_files.items()),
        fit_set,
    )


def _build_fit_request(
    tracks: list[Track],
    analysis_config: dict[int, dict],
    files: dict[int, LeedFile],
    background_file: Optional[LeedFile],
    previous: FitSet | None,
    is_cancelled: Callable[[], bool],
) -> FitRequest:
    if is_cancelled():
        raise FitCancelled("Track Analysis was cancelled before fitting")
    observations = _fit_inputs(tracks, analysis_config, files, background_file)
    return FitRequest(
        observations=observations,
        previous=previous,
    )


def _fit_inputs(
    tracks: list[Track],
    analysis_config: dict[int, dict],
    files: dict[int, LeedFile],
    background_file: Optional[LeedFile],
) -> tuple[FitObservationInput, ...]:
    inputs = []
    frame_ids = sorted(
        {frame_id for track in tracks for frame_id in track.spots_by_image_id}
    )
    for frame_id in frame_ids:
        if frame_id not in files:
            raise ValueError(f"No source file for frame {frame_id}")
        for track in tracks:
            spot = track.spots_by_image_id.get(frame_id)
            if spot is None:
                continue
            rect = spot.roi_rect
            use_background = bool(
                background_file is not None
                and analysis_config.get(track.track_id, {}).get(
                    "subtract_bg", False
                )
            )
            source = files[frame_id]
            inputs.append(FitObservationInput(
                track_id=track.track_id,
                frame_id=frame_id,
                source_path=source.filepath,
                roi=Rect(rect.x(), rect.y(), rect.width(), rect.height()),
                subtract_background=use_background,
                background_path=(
                    background_file.filepath
                    if use_background and background_file is not None
                    else None
                ),
                energy=source.energy,
                temperature=source.temperature,
            ))
    return tuple(inputs)


def _tracks_with_projected_fits(
    tracks: list[Track],
    fit_set: FitSet,
    geometry: LegacyGeometryResult,
) -> list[Track]:
    projection = project_fit_set(fit_set)
    observation_geometry = {
        (item.track_id, item.frame_id): item
        for item in geometry.observations
    }
    temporary = []
    for source in tracks:
        target = Track(source.track_id, dict(source.spots_by_image_id))
        target.analysis_results = projection.get(source.track_id, {})
        for frame_id, result in target.analysis_results.items():
            derived = observation_geometry.get((source.track_id, frame_id))
            if result is None or derived is None:
                continue
            result["radius"] = derived.radius
            if derived.radius_error is not None:
                result["radius_error"] = derived.radius_error
        temporary.append(target)
    return temporary


def _files_with_geometry(
    files: dict[int, LeedFile],
    geometry: LegacyGeometryResult,
) -> dict[int, LeedFile]:
    temporary = {frame_id: copy(item) for frame_id, item in files.items()}
    for frame in geometry.frames:
        target = temporary[frame.frame_id]
        target.pattern_center = (
            None if frame.pattern_center is None else dict(frame.pattern_center)
        )
        target.post_analysis_results = {}
        if frame.average_radius is not None:
            target.post_analysis_results["avg_radius"] = frame.average_radius
            target.post_analysis_results["avg_radius_error"] = (
                frame.average_radius_error
            )
    return temporary


def publish_analysis(
    publication: AnalysisPublication,
    targets: list[Track],
    target_files: dict[int, LeedFile],
) -> None:
    sources = publication.tracks
    source_files = dict(publication.files)
    analyzed_frame_ids = {
        frame_id for track in sources for frame_id in track.spots_by_image_id
    }
    for target, source in zip(targets, sources, strict=True):
        target.analysis_results = source.analysis_results
    for frame_id in analyzed_frame_ids:
        target = target_files[frame_id]
        source = source_files[frame_id]
        target.pattern_center = source.pattern_center
        target.post_analysis_results = dict(
            getattr(source, "post_analysis_results", {})
        )


def _geometry_request(
    tracks: list[Track],
    center_mode: str,
    manual_ellipse: Optional[dict[str, float]],
) -> LegacyGeometryRequest:
    try:
        mode = LegacyCenterMode((center_mode or "per_frame").lower())
    except ValueError as cause:
        raise ValueError(f"Unsupported center mode: {center_mode}") from cause
    rois = tuple(
        GeometryObservationRoi(
            track.track_id,
            frame_id,
            Rect(
                spot.roi_rect.x(),
                spot.roi_rect.y(),
                spot.roi_rect.width(),
                spot.roi_rect.height(),
            ),
        )
        for track in tracks
        for frame_id, spot in track.spots_by_image_id.items()
    )
    ellipse = (
        manual_ellipse_from_mapping(manual_ellipse)
        if mode is LegacyCenterMode.MANUAL_ELLIPSE
        else None
    )
    return LegacyGeometryRequest(rois, mode, ellipse)


def project_fit_set(fit_set: FitSet) -> LegacyFitProjection:
    """Return fresh legacy dictionaries without adding geometry fields."""
    projection: LegacyFitProjection = {}
    for observation in fit_set.observations:
        track_results = projection.setdefault(observation.track_id, {})
        track_results[observation.frame_id] = (
            None if observation.result is None else _project_result(observation.result)
        )
    return projection


def _project_result(result: GaussianFitResult) -> dict[str, object]:
    parameters = result.parameters
    errors = result.errors
    intensity = result.intensity
    widths = result.widths
    quality = result.quality
    return {
        "amplitude": parameters.amplitude,
        "center_x": parameters.center_x,
        "center_y": parameters.center_y,
        "sigma_x": parameters.sigma_x,
        "sigma_y": parameters.sigma_y,
        "theta": parameters.theta,
        "offset": parameters.offset,
        "covariance_matrix": np.asarray(result.covariance_matrix, dtype=float),
        "amplitude_error": errors.amplitude,
        "center_x_error": errors.center_x,
        "center_y_error": errors.center_y,
        "sigma_x_error": errors.sigma_x,
        "sigma_y_error": errors.sigma_y,
        "theta_error": errors.theta,
        "offset_error": errors.offset,
        "volume": intensity.volume,
        "volume_error": intensity.volume_error,
        **{
            name: getattr(widths, name)
            for name in widths.__dataclass_fields__
        },
        "r_squared": quality.r_squared,
        "chi_squared_reduced": quality.chi_squared_reduced,
        "global_center_x": result.global_center_x,
        "global_center_y": result.global_center_y,
    }
