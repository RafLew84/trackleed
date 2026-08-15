"""Atomic projection of neutral tracking results into legacy GUI models."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from PyQt6.QtCore import QRect

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_tracking_legacy_models import (
    apply_spot_diagnostics,
    spot_diagnostics,
)
from trackleed.logic.workflows.seeded_tracking import (
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)
from trackleed.logic.active_analysis_state import ActiveAnalysisState
from trackleed.logic.session_analysis_projection import (
    retain_compatible_analysis_fits,
)
from trackleed.logic.track_analysis_compatibility import project_fit_set
from trackleed.logic.workflows.track_analysis import FitSet


class TrackingPublicationTarget(Protocol):
    tracking_results: list[Track]
    _active_analysis_state: ActiveAnalysisState


@dataclass(frozen=True)
class _PreparedTrackingPublication:
    tracks: list[Track]
    frame_spots: tuple[tuple[SpotROI, ...], ...]
    analysis_state: ActiveAnalysisState
    geometry_projections: tuple[dict, ...] | None


def publish_tracks(
    target: TrackingPublicationTarget,
    files: Sequence[LeedFile],
    edited_tracks: Sequence[Track],
) -> None:
    """Publish tracks, frame spots, and reconciled analysis as one update."""
    prepared = _prepare_tracking_publication(
        files,
        edited_tracks,
        target.tracking_results,
        target._active_analysis_state,
    )
    for file, spots in zip(files, prepared.frame_spots, strict=True):
        file.spots = list(spots)
    if prepared.geometry_projections is not None:
        _apply_geometry_invalidation(files, prepared.geometry_projections)
    target.tracking_results = prepared.tracks
    target._active_analysis_state = prepared.analysis_state


def _prepare_tracking_publication(
    files: Sequence[LeedFile],
    edited_tracks: Sequence[Track],
    previous_tracks: Sequence[Track],
    analysis_state: ActiveAnalysisState,
) -> _PreparedTrackingPublication:
    frame_ids = _stable_frame_ids(files)
    _validate_track_inventory(edited_tracks)
    active_tracks = clone_tracks_for_review(edited_tracks)
    _validate_observation_frames(frame_ids, active_tracks)
    reconciled = retain_compatible_analysis_fits(
        analysis_state,
        active_tracks,
        previous_tracks=previous_tracks,
    )
    _project_active_analysis(active_tracks, reconciled)
    geometry_projections = (
        _geometry_projections(files) if reconciled.geometry_result is None else None
    )
    return _PreparedTrackingPublication(
        active_tracks,
        _frame_spot_projections(frame_ids, active_tracks),
        reconciled,
        geometry_projections,
    )


def _validate_track_inventory(tracks: Sequence[Track]) -> None:
    track_ids = tuple(track.track_id for track in tracks)
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Tracking publication requires unique track IDs")


def _validate_observation_frames(
    frame_ids: Sequence[int],
    tracks: Sequence[Track],
) -> None:
    observation_ids = {
        frame_id for track in tracks for frame_id in track.spots_by_image_id
    }
    missing_ids = observation_ids - set(frame_ids)
    if missing_ids:
        raise ValueError(f"Tracking result contains unknown frame IDs: {missing_ids}")


def _frame_spot_projections(
    frame_ids: Sequence[int],
    tracks: Sequence[Track],
) -> tuple[tuple[SpotROI, ...], ...]:
    frame_spots: dict[int, list[SpotROI]] = {
        frame_id: [] for frame_id in frame_ids
    }
    for track in tracks:
        for frame_id, spot in track.spots_by_image_id.items():
            frame_spots[frame_id].append(_clone_spot(spot))
    return tuple(tuple(frame_spots[frame_id]) for frame_id in frame_ids)


def _project_active_analysis(
    tracks: Sequence[Track],
    state: ActiveAnalysisState,
) -> None:
    projection = (
        project_fit_set(state.fit_set)
        if isinstance(state.fit_set, FitSet)
        else {}
    )
    for track in tracks:
        track.analysis_results = deepcopy(projection.get(track.track_id, {}))
    if state.geometry_result is None:
        return
    tracks_by_id = {track.track_id: track for track in tracks}
    for item in state.geometry_result.observations:
        result = tracks_by_id[item.track_id].analysis_results.get(item.frame_id)
        if result is not None:
            result["radius"] = item.radius
            if item.radius_error is not None:
                result["radius_error"] = item.radius_error


def project_tracking_result(result: TrackingResult) -> list[Track]:
    """Translate a neutral result into detached legacy track values."""
    return [_track_from_identity(identity) for identity in result.tracks]


def clone_tracks_for_review(tracks: Sequence[Track]) -> list[Track]:
    """Create an editable copy without sharing mutable observations or fits."""
    return [_clone_track(track) for track in tracks]


def _geometry_projections(files: Sequence[LeedFile]) -> tuple[dict, ...]:
    projections: list[dict] = []
    for file in files:
        post_analysis = dict(getattr(file, "post_analysis_results", {}) or {})
        for key in (
            "avg_radius",
            "avg_radius_error",
            "average_radius",
            "average_radius_error",
        ):
            post_analysis.pop(key, None)
        projections.append(post_analysis)
    return tuple(projections)


def _apply_geometry_invalidation(
    files: Sequence[LeedFile],
    projections: Sequence[dict],
) -> None:
    for file, post_analysis in zip(files, projections, strict=True):
        file.pattern_center = None
        file.post_analysis_results = post_analysis
        if hasattr(file, "geometry_result"):
            file.geometry_result = None


def _track_from_identity(identity: TrackedSpotIdentity) -> Track:
    return Track(
        track_id=identity.track_id,
        spots_by_image_id={
            observation.frame_id: _spot_from_observation(
                identity.track_id,
                observation,
            )
            for observation in identity.observations
        },
    )


def _spot_from_observation(
    track_id: int,
    observation: TrackObservation,
) -> SpotROI:
    rect = observation.roi
    spot = SpotROI(
        id=track_id,
        roi_rect=QRect(rect.x, rect.y, rect.width, rect.height),
        center_x=observation.center_x,
        center_y=observation.center_y,
        intensity=observation.intensity,
        source=observation.source,
        polygon_seed_points=list(observation.polygon_seed_points),
    )
    return apply_spot_diagnostics(spot, observation.diagnostics)


def _clone_track(track: Track) -> Track:
    clone = Track(
        track_id=track.track_id,
        spots_by_image_id={
            frame_id: _clone_spot(spot)
            for frame_id, spot in track.spots_by_image_id.items()
        },
    )
    if hasattr(track, "analysis_results"):
        clone.analysis_results = deepcopy(track.analysis_results)
    return clone


def _clone_spot(spot: SpotROI) -> SpotROI:
    rect = spot.roi_rect
    clone = SpotROI(
        id=spot.id,
        roi_rect=QRect(rect.x(), rect.y(), rect.width(), rect.height()),
        center_x=spot.center_x,
        center_y=spot.center_y,
        intensity=spot.intensity,
        source=spot.source,
        polygon_seed_points=list(spot.polygon_seed_points),
    )
    return apply_spot_diagnostics(clone, spot_diagnostics(spot))


def _stable_frame_ids(files: Sequence[LeedFile]) -> tuple[int, ...]:
    frame_ids = tuple(file.id for file in files)
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError("Tracking publication requires every frame to have an ID")
    stable_ids = tuple(int(frame_id) for frame_id in frame_ids if frame_id is not None)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Tracking publication requires unique frame IDs")
    return stable_ids
