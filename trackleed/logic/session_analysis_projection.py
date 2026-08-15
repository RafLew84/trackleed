"""Consistent session snapshot projection for accepted analysis observations."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisResultStatus,
)
from trackleed.logic.workflows.track_analysis import FitSet, Rect

LegacyFit = dict[str, object] | None


@dataclass(frozen=True)
class AnalysisSessionProjection:
    """Typed save projection aligned with the currently accepted observations."""

    tracks: tuple[Track, ...]
    files: tuple[LeedFile, ...]
    state: ActiveAnalysisState


def retain_compatible_analysis_fits(
    state: ActiveAnalysisState,
    tracks: list[Track],
) -> ActiveAnalysisState:
    """Retain fits whose active observation still has the fitted ROI."""
    if not isinstance(state.fit_set, FitSet):
        return state.invalidate_geometry()
    active_rois = {
        (track.track_id, frame_id): Rect(
            spot.roi_rect.x(),
            spot.roi_rect.y(),
            spot.roi_rect.width(),
            spot.roi_rect.height(),
        )
        for track in tracks
        for frame_id, spot in track.spots_by_image_id.items()
    }
    compatible_keys = {
        (item.track_id, item.frame_id)
        for item in state.fit_set.observations
        if active_rois.get((item.track_id, item.frame_id))
        == item.provenance.original_roi
    }
    retained = state.retain_observations(compatible_keys)
    if set(active_rois) != compatible_keys:
        return retained.invalidate_geometry()
    return retained


def project_analysis_session(
    state: ActiveAnalysisState,
    tracks: list[Track],
    files: list[LeedFile],
    *,
    synchronize_track_spots: bool = False,
) -> AnalysisSessionProjection:
    """Build a non-mutating flat projection for a versioned session snapshot."""
    retained_state = retain_compatible_analysis_fits(state, tracks)
    session_files = (
        _files_with_track_spots(files, tracks)
        if synchronize_track_spots
        else files
    )
    if (
        retained_state.result_status is not AnalysisResultStatus.CURRENT
        or retained_state.geometry_result is not None
    ):
        return AnalysisSessionProjection(
            tuple(tracks),
            tuple(session_files),
            retained_state,
        )
    compatible_keys = {
        (item.track_id, item.frame_id)
        for item in (retained_state.fit_set or FitSet(())).observations
    }
    projected_tracks = []
    for source in tracks:
        target = Track(source.track_id, dict(source.spots_by_image_id))
        target.analysis_results = {
            frame_id: _fit_only_legacy_result(result)
            for frame_id, result in getattr(source, "analysis_results", {}).items()
            if (source.track_id, frame_id) in compatible_keys
        }
        projected_tracks.append(target)
    projected_files = _files_without_geometry(session_files)
    return AnalysisSessionProjection(
        tuple(projected_tracks),
        tuple(projected_files),
        retained_state,
    )


def _files_with_track_spots(
    files: list[LeedFile],
    tracks: list[Track],
) -> list[LeedFile]:
    frame_ids = [item.id for item in files]
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError("Session track projection requires every frame to have an ID")
    stable_ids = [int(frame_id) for frame_id in frame_ids if frame_id is not None]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Session track projection requires unique frame IDs")
    frame_spots: dict[int, list[SpotROI]] = {
        frame_id: [] for frame_id in stable_ids
    }
    for track in tracks:
        for frame_id, spot in track.spots_by_image_id.items():
            if frame_id not in frame_spots:
                raise ValueError(
                    f"Session track projection contains unknown frame ID: {frame_id}"
                )
            frame_spots[frame_id].append(deepcopy(spot))
    projected = []
    for source, frame_id in zip(files, stable_ids, strict=True):
        target = copy(source)
        target.spots = frame_spots[frame_id]
        projected.append(target)
    return projected


def _files_without_geometry(files: list[LeedFile]) -> list[LeedFile]:
    projected = []
    for source in files:
        target = copy(source)
        target.pattern_center = None
        post_analysis = dict(getattr(source, "post_analysis_results", {}) or {})
        for key in (
            "avg_radius",
            "avg_radius_error",
            "average_radius",
            "average_radius_error",
        ):
            post_analysis.pop(key, None)
        target.post_analysis_results = post_analysis
        projected.append(target)
    return projected


def _fit_only_legacy_result(result: LegacyFit) -> LegacyFit:
    if not isinstance(result, dict):
        return result
    projected = dict(result)
    projected.pop("radius", None)
    projected.pop("radius_error", None)
    return projected
