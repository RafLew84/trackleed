"""Atomic projection of neutral tracking results into legacy GUI models."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

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


def publish_tracking_result(
    files: Sequence[LeedFile],
    result: TrackingResult,
    *,
    previous_tracks: Sequence[Track] | None = None,
) -> list[Track]:
    """Publish one complete neutral result without mutable spot aliasing."""
    return publish_edited_tracks(
        files,
        project_tracking_result(result),
        previous_tracks=previous_tracks,
    )


def project_tracking_result(result: TrackingResult) -> list[Track]:
    """Translate a neutral result into detached legacy track values."""
    return [_track_from_identity(identity) for identity in result.tracks]


def clone_tracks_for_review(tracks: Sequence[Track]) -> list[Track]:
    """Create an editable copy without sharing mutable observations or fits."""
    return [_clone_track(track) for track in tracks]


def publish_edited_tracks(
    files: Sequence[LeedFile],
    edited_tracks: Sequence[Track],
    *,
    previous_tracks: Sequence[Track] | None = None,
) -> list[Track]:
    """Atomically publish edited tracks and independent frame projections."""
    frame_ids = _stable_frame_ids(files)
    files_by_id = dict(zip(frame_ids, files, strict=True))
    observation_ids = {
        frame_id for track in edited_tracks for frame_id in track.spots_by_image_id
    }
    missing_ids = observation_ids - files_by_id.keys()
    if missing_ids:
        raise ValueError(f"Tracking result contains unknown frame IDs: {missing_ids}")
    active_tracks = clone_tracks_for_review(edited_tracks)
    if previous_tracks is not None:
        _retain_unchanged_analysis_results(active_tracks, previous_tracks)
    frame_spots: dict[int, list[SpotROI]] = {frame_id: [] for frame_id in files_by_id}
    for track in active_tracks:
        for frame_id, spot in track.spots_by_image_id.items():
            frame_spots[frame_id].append(_clone_spot(spot))
    ordered_files = tuple(files_by_id.values())
    geometry_projections = _geometry_projections(ordered_files)
    for frame_id, file in files_by_id.items():
        file.spots = frame_spots[frame_id]
    _apply_geometry_invalidation(ordered_files, geometry_projections)
    return active_tracks


def _geometry_projections(files: Sequence[LeedFile]) -> tuple[dict, ...]:
    projections: list[dict] = []
    for file in files:
        post_analysis = dict(getattr(file, "post_analysis_results", {}) or {})
        post_analysis.pop("avg_radius", None)
        post_analysis.pop("avg_radius_error", None)
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


def _retain_unchanged_analysis_results(
    tracks: Sequence[Track],
    previous_tracks: Sequence[Track],
) -> None:
    previous_by_id = {track.track_id: track for track in previous_tracks}
    for track in tracks:
        previous = previous_by_id.get(track.track_id)
        previous_results = getattr(previous, "analysis_results", None)
        if not isinstance(previous_results, dict):
            track.analysis_results = {}
            continue
        track.analysis_results = {
            frame_id: deepcopy(result)
            for frame_id, result in previous_results.items()
            if _same_fit_input_roi(
                track.spots_by_image_id.get(frame_id),
                previous.spots_by_image_id.get(frame_id),
            )
        }


def _same_fit_input_roi(current: SpotROI | None, previous: SpotROI | None) -> bool:
    if current is None or previous is None:
        return False
    current_rect = current.roi_rect
    previous_rect = previous.roi_rect
    return (
        current_rect.x(),
        current_rect.y(),
        current_rect.width(),
        current_rect.height(),
    ) == (
        previous_rect.x(),
        previous_rect.y(),
        previous_rect.width(),
        previous_rect.height(),
    )


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
