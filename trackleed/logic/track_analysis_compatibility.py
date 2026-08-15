"""Atomic projection from immutable fits to the legacy Track Analysis view."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from threading import Event
from typing import Optional

import numpy as np

from trackleed.analysis.legacy_track_geometry import (
    manual_ellipse_from_mapping,
)
from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.workflows.session_sources import (
    SourceVerificationCancelled,
    SourceVerificationContext,
    verify_analysis_sources,
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
from trackleed.logic.workflows.track_geometry import (
    GeometryCancelled,
    GeometryMethod,
    GeometryProgress,
    GeometryRequest,
    GeometryResult,
    GeometryRunContext,
    PatternCenter,
    derive_geometry,
)

LegacyFit = dict[str, object] | None
LegacyFitProjection = dict[int, dict[int, LegacyFit]]


@dataclass(frozen=True)
class AnalysisPublication:
    tracks: tuple[Track, ...]
    files: tuple[tuple[int, LeedFile], ...]
    fit_set: FitSet
    geometry_result: GeometryResult


class AnalysisSourceVerificationError(RuntimeError):
    """Raised before fitting when persisted source pixels cannot be verified."""


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


def _report_geometry_only_progress(
    progress_callback: Callable[[int], None],
    progress: GeometryProgress,
) -> None:
    progress_callback(int(progress.completed * 100 / max(1, progress.total)))


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
    previous_geometry_result: GeometryResult | None = None,
    analysis_state: ActiveAnalysisState | None = None,
) -> AnalysisPublication | None:
    """Run the cold workflow and publish only after every stage finishes."""
    is_cancelled = partial(_event_is_set, cancel_event)
    try:
        previous_fit_set, previous_geometry_result = _validated_previous_analysis(
            analysis_state,
            previous_fit_set,
            previous_geometry_result,
            is_cancelled,
        )
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
        geometry = derive_geometry(
            fit_set,
            _geometry_request(
                tracks_to_analyze,
                center_mode,
                manual_ellipse,
                previous_geometry_result,
            ),
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
        progress_callback(100)
        if is_cancelled():
            raise FitCancelled("Track Analysis was cancelled before publication")
    except (FitCancelled, GeometryCancelled, SourceVerificationCancelled):
        print("[Analysis] Analysis cancelled; previous publication retained.")
        return None
    print("[Analysis] Analysis completed successfully.")
    return _publication(
        temporary_tracks,
        temporary_files,
        fit_set,
        geometry,
    )


def _validated_previous_analysis(
    state: ActiveAnalysisState | None,
    fit_set: FitSet | None,
    geometry_result: GeometryResult | None,
    is_cancelled: Callable[[], bool],
) -> tuple[FitSet | None, GeometryResult | None]:
    if state is None:
        return fit_set, geometry_result
    verification = verify_analysis_sources(
        state,
        SourceVerificationContext(load_tiff, is_cancelled=is_cancelled),
    )
    if verification.failures:
        reasons = ", ".join(item.reason.value for item in verification.failures)
        raise AnalysisSourceVerificationError(
            f"Analysis source verification failed: {reasons}"
        )
    return verification.state.fit_set, verification.state.geometry_result


def run_geometry_analysis_publication(
    tracks_to_analyze: list[Track],
    all_files_map: dict[int, LeedFile],
    fit_set: FitSet,
    progress_callback: Callable[[int], None],
    center_mode: str = "per_frame",
    manual_ellipse: Optional[dict[str, float]] = None,
    cancel_event: Optional[Event] = None,
    previous_geometry_result: GeometryResult | None = None,
) -> AnalysisPublication | None:
    """Publish new geometry from an explicitly committed immutable ``FitSet``."""
    is_cancelled = partial(_event_is_set, cancel_event)
    try:
        geometry = derive_geometry(
            fit_set,
            _geometry_request(
                tracks_to_analyze,
                center_mode,
                manual_ellipse,
                previous_geometry_result,
            ),
            GeometryRunContext(
                report_progress=partial(
                    _report_geometry_only_progress,
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
        progress_callback(100)
        if is_cancelled():
            raise GeometryCancelled("Track geometry was cancelled before publication")
    except GeometryCancelled:
        print("[Analysis] Geometry cancelled; previous publication retained.")
        return None
    print("[Analysis] Geometry completed successfully.")
    return _publication(
        temporary_tracks,
        temporary_files,
        fit_set,
        geometry,
    )


def can_reuse_fit_set_for_geometry(
    fit_set: FitSet,
    tracks: list[Track],
    analysis_config: dict[int, dict],
    background_file: LeedFile | None,
) -> bool:
    """Return whether current GUI fit-affecting controls match ``fit_set``."""
    expected = {
        (track.track_id, frame_id): (
            Rect(
                spot.roi_rect.x(),
                spot.roi_rect.y(),
                spot.roi_rect.width(),
                spot.roi_rect.height(),
            ),
            bool(
                background_file is not None
                and analysis_config.get(track.track_id, {}).get("subtract_bg", False)
            ),
            (
                background_file.filepath
                if background_file is not None
                and analysis_config.get(track.track_id, {}).get("subtract_bg", False)
                else None
            ),
        )
        for track in tracks
        for frame_id, spot in track.spots_by_image_id.items()
    }
    actual = {
        (item.track_id, item.frame_id): (
            item.provenance.original_roi,
            item.provenance.background_used,
            item.provenance.background_path,
        )
        for item in fit_set.observations
    }
    return actual == expected


def _publication(
    tracks: list[Track],
    files: dict[int, LeedFile],
    fit_set: FitSet,
    geometry: GeometryResult,
) -> AnalysisPublication:
    return AnalysisPublication(
        tuple(tracks),
        tuple(files.items()),
        fit_set,
        geometry,
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
                and analysis_config.get(track.track_id, {}).get("subtract_bg", False)
            )
            source = files[frame_id]
            inputs.append(
                FitObservationInput(
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
                )
            )
    return tuple(inputs)


def _tracks_with_projected_fits(
    tracks: list[Track],
    fit_set: FitSet,
    geometry: GeometryResult,
) -> list[Track]:
    projection = project_fit_set(fit_set)
    observation_geometry = {
        (item.track_id, item.frame_id): item for item in geometry.observations
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
    geometry: GeometryResult,
) -> dict[int, LeedFile]:
    temporary = {frame_id: copy(item) for frame_id, item in files.items()}
    for frame in geometry.frames:
        target = temporary[frame.frame_id]
        target.pattern_center = (
            None if frame.center is None else _center_mapping(frame.center)
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
    if len(sources) != len(targets):
        raise ValueError("Analysis publication track count does not match targets")
    source_track_ids = tuple(track.track_id for track in sources)
    target_track_ids = tuple(track.track_id for track in targets)
    if len(set(source_track_ids)) != len(source_track_ids):
        raise ValueError("Analysis publication source track IDs must be unique")
    if len(set(target_track_ids)) != len(target_track_ids):
        raise ValueError("Analysis publication target track IDs must be unique")
    if set(source_track_ids) != set(target_track_ids):
        raise ValueError("Analysis publication track ID set does not match targets")
    sources_by_id = {track.track_id: track for track in sources}
    observation_frame_ids = {
        frame_id for track in sources for frame_id in track.spots_by_image_id
    }
    result_frame_ids = {
        frame_id
        for track in sources
        for frame_id in getattr(track, "analysis_results", {})
    }
    analyzed_frame_ids = observation_frame_ids | result_frame_ids
    source_frame_counts = Counter(frame_id for frame_id, _file in publication.files)
    invalid_source_frames = {
        frame_id
        for frame_id in analyzed_frame_ids
        if source_frame_counts[frame_id] != 1
    }
    source_files = dict(publication.files)
    missing_target_frames = analyzed_frame_ids - target_files.keys()
    if invalid_source_frames or missing_target_frames:
        raise ValueError(
            "Analysis publication has invalid frame inventory: "
            f"source={invalid_source_frames}, target={missing_target_frames}"
        )
    track_replacements = tuple(
        (
            target,
            deepcopy(sources_by_id[target.track_id].analysis_results),
        )
        for target in targets
    )
    file_replacements = tuple(
        (
            target_files[frame_id],
            deepcopy(source_files[frame_id].pattern_center),
            deepcopy(getattr(source_files[frame_id], "post_analysis_results", {})),
        )
        for frame_id in sorted(analyzed_frame_ids)
    )
    for target, analysis_results in track_replacements:
        target.analysis_results = analysis_results
    for target_file, pattern_center, post_analysis_results in file_replacements:
        target_file.pattern_center = pattern_center
        target_file.post_analysis_results = post_analysis_results


def _geometry_request(
    tracks: list[Track],
    center_mode: str,
    manual_ellipse: Optional[dict[str, float]],
    previous: GeometryResult | None = None,
) -> GeometryRequest:
    try:
        mode = GeometryMethod((center_mode or "per_frame").lower())
    except ValueError as cause:
        raise ValueError(f"Unsupported center mode: {center_mode}") from cause
    ellipse = (
        manual_ellipse_from_mapping(manual_ellipse)
        if mode is GeometryMethod.MANUAL_ELLIPSE
        else None
    )
    return GeometryRequest(
        method=mode,
        center_reference_track_ids=tuple(track.track_id for track in tracks),
        manual_ellipse=ellipse,
        previous=previous,
    )


def _center_mapping(center: PatternCenter) -> dict[str, float]:
    values = {
        "center_x": center.center_x,
        "center_y": center.center_y,
        "error_x": center.error_x,
        "error_y": center.error_y,
    }
    optional = {
        "circle_radius": center.circle_radius,
        "radius_x": center.radius_x,
        "radius_y": center.radius_y,
        "angle_deg": center.angle_degrees,
        "radius_scale_px": center.radius_scale,
    }
    values.update({key: value for key, value in optional.items() if value is not None})
    return values


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
        **{name: getattr(widths, name) for name in widths.__dataclass_fields__},
        "r_squared": quality.r_squared,
        "chi_squared_reduced": quality.chi_squared_reduced,
        "global_center_x": result.global_center_x,
        "global_center_y": result.global_center_y,
    }
