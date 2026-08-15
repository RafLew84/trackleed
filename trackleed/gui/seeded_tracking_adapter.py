"""Qt/model compatibility adapter for the seeded-tracking workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QRect

from trackleed.analysis.seeded_bbox_tracking_runtime import (
    default_bbox_tracker_configs,
)
from trackleed.analysis.seeded_point_tracking_runtime import (
    default_point_tracker_configs,
)
from trackleed.analysis.seeded_mask_tracking_runtime import (
    default_mask_tracker_configs,
)
from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.gui.seeded_bbox_tracking_compatibility import (
    run_legacy_bbox_retracker,
    run_legacy_bbox_tracker,
)
from trackleed.gui.seeded_point_tracking_compatibility import (
    run_legacy_point_retracker,
    run_legacy_point_tracker,
)
from trackleed.gui.seeded_mask_tracking_compatibility import (
    run_legacy_mask_retracker,
    run_legacy_mask_tracker,
)
from trackleed.gui.seeded_tracking_legacy_models import (
    apply_spot_diagnostics,
    spot_diagnostics,
)
from trackleed.logic.workflows.bbox_tracking import BoundingBoxTrackerConfig
from trackleed.logic.workflows.mask_tracking import MaskTrackerConfig
from trackleed.logic.workflows.point_tracking import PointTrackerConfig
from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    RunContext,
    SeededRetrackRequest,
    SeededTrackingCancelled,
    SeededTrackingConfig,
    SeededTrackingFrame,
    SeededTrackingRequest,
    SeedRect,
    SeedRepresentation,
    SkipAnchorOpticalFlowConfig,
    TrackedSpotIdentity,
    TrackingProgress,
    TrackingResult,
    TrackingSeed,
    TrackObservation,
    run_seeded_retrack,
    run_seeded_tracking,
)


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class SeededTrackingDialogSettings:
    selected_method_id: str
    optical_flow: OpticalFlowConfig
    skip_anchor: SkipAnchorOpticalFlowConfig
    point_configs: tuple[tuple[str, PointTrackerConfig], ...]
    bbox_configs: tuple[tuple[str, BoundingBoxTrackerConfig], ...]
    mask_configs: tuple[tuple[str, MaskTrackerConfig], ...]

    @classmethod
    def defaults(cls, *, roi_size: int = 32) -> SeededTrackingDialogSettings:
        return cls(
            selected_method_id="optical_flow",
            optical_flow=OpticalFlowConfig(roi_size=roi_size),
            skip_anchor=SkipAnchorOpticalFlowConfig(roi_size=roi_size),
            point_configs=default_point_tracker_configs(roi_size=roi_size),
            bbox_configs=default_bbox_tracker_configs(roi_size=roi_size),
            mask_configs=default_mask_tracker_configs(roi_size=roi_size),
        )

    def point_config(self, method_id: str) -> PointTrackerConfig:
        for configured_method_id, config in self.point_configs:
            if configured_method_id == method_id:
                return config
        raise KeyError(method_id)

    def with_point_config(
        self,
        method_id: str,
        config: PointTrackerConfig,
    ) -> SeededTrackingDialogSettings:
        configs = dict(self.point_configs)
        if method_id not in configs:
            raise KeyError(method_id)
        configs[method_id] = config
        return replace(self, point_configs=tuple(configs.items()))

    def bbox_config(self, method_id: str) -> BoundingBoxTrackerConfig:
        for configured_method_id, config in self.bbox_configs:
            if configured_method_id == method_id:
                return config
        raise KeyError(method_id)

    def with_bbox_config(
        self,
        method_id: str,
        config: BoundingBoxTrackerConfig,
    ) -> SeededTrackingDialogSettings:
        configs = dict(self.bbox_configs)
        if method_id not in configs:
            raise KeyError(method_id)
        configs[method_id] = config
        return replace(self, bbox_configs=tuple(configs.items()))

    def mask_config(self, method_id: str) -> MaskTrackerConfig:
        for configured_method_id, config in self.mask_configs:
            if configured_method_id == method_id:
                return config
        raise KeyError(method_id)

    def with_mask_config(
        self,
        method_id: str,
        config: MaskTrackerConfig,
    ) -> SeededTrackingDialogSettings:
        configs = dict(self.mask_configs)
        if method_id not in configs:
            raise KeyError(method_id)
        configs[method_id] = config
        return replace(self, mask_configs=tuple(configs.items()))


class SeededTrackingSettingsBinding:
    """Own the last accepted tracking settings for this process only."""

    def __init__(self, *, roi_size: int = 32) -> None:
        self._settings = SeededTrackingDialogSettings.defaults(roi_size=roi_size)

    @property
    def dialog_settings(self) -> SeededTrackingDialogSettings:
        return self._settings

    def accept(self, settings: SeededTrackingDialogSettings) -> None:
        self._settings = settings


def build_seeded_tracking_request(
    *,
    files: Sequence[LeedFile],
    reference_file: LeedFile | None,
    method_id: str,
    config: SeededTrackingConfig,
    background_file: LeedFile | None,
) -> SeededTrackingRequest:
    """Snapshot mutable GUI frame and seed models into neutral workflow values."""
    if reference_file is None:
        raise ValueError("Reference frame must be selected")
    frame_ids = _stable_frame_ids(files)
    if reference_file.id is None or int(reference_file.id) not in frame_ids:
        raise ValueError("Reference frame must belong to the requested frame series")
    return SeededTrackingRequest(
        method_id=method_id,
        frames=tuple(
            SeededTrackingFrame(frame_id, leed_file.filepath)
            for frame_id, leed_file in zip(frame_ids, files, strict=True)
        ),
        reference_frame_id=int(reference_file.id),
        seeds=tuple(
            TrackingSeed(
                seed_id=spot.id,
                center_x=float(spot.center_x),
                center_y=float(spot.center_y),
                roi=SeedRect(
                    spot.roi_rect.x(),
                    spot.roi_rect.y(),
                    spot.roi_rect.width(),
                    spot.roi_rect.height(),
                ),
                intensity=float(spot.intensity),
                source=str(spot.source),
                polygon_seed_points=tuple(
                    (float(x), float(y)) for x, y in spot.polygon_seed_points
                ),
            )
            for spot in reference_file.spots
        ),
        config=config,
        background_path=(
            background_file.filepath if background_file is not None else None
        ),
        seed_representation=_seed_representation(config),
    )


def build_seeded_retrack_request(
    *,
    files: Sequence[LeedFile],
    track: Track,
    start_frame_id: int,
    method_id: str,
    config: PointTrackerConfig | BoundingBoxTrackerConfig | MaskTrackerConfig,
) -> SeededRetrackRequest:
    """Snapshot one mutable review track into a neutral re-track request."""
    frame_ids = _stable_frame_ids(files)
    files_by_id = dict(zip(frame_ids, files, strict=True))
    if start_frame_id not in files_by_id:
        raise ValueError("Re-track start frame must belong to the frame series")
    return SeededRetrackRequest(
        method_id=method_id,
        frames=tuple(
            SeededTrackingFrame(frame_id, files_by_id[frame_id].filepath)
            for frame_id in frame_ids
        ),
        track=TrackedSpotIdentity(
            track.track_id,
            tuple(
                _observation_from_spot(frame_id, track.spots_by_image_id[frame_id])
                for frame_id in frame_ids
                if frame_id in track.spots_by_image_id
            ),
        ),
        start_frame_id=start_frame_id,
        config=config,
        seed_representation=_seed_representation(config),
    )


def publish_tracking_result(
    files: Sequence[LeedFile],
    result: TrackingResult,
) -> list[Track]:
    """Project a complete neutral result without aliasing tracks and frame spots."""
    tracks = [_track_from_identity(identity) for identity in result.tracks]
    return publish_edited_tracks(files, tracks)


def clone_tracks_for_review(tracks: Sequence[Track]) -> list[Track]:
    """Create an editable copy without sharing mutable observations."""
    return [_clone_track(track) for track in tracks]


def publish_edited_tracks(
    files: Sequence[LeedFile],
    edited_tracks: Sequence[Track],
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
    frame_spots: dict[int, list[SpotROI]] = {frame_id: [] for frame_id in files_by_id}
    for track in edited_tracks:
        for frame_id, spot in track.spots_by_image_id.items():
            frame_spots[frame_id].append(_clone_spot(spot))
    for frame_id, file in files_by_id.items():
        file.spots = frame_spots[frame_id]
    return active_tracks


def run_seeded_tracking_job(
    request: SeededTrackingRequest,
    image_reader: Callable[[Path], np.ndarray],
    cancel_signal: CancellationSignal,
    *,
    progress_callback: Callable[[int], None],
) -> TrackingResult | None:
    """Run the headless workflow inside GenericWorker without GUI model access."""
    context = RunContext(
        image_reader=image_reader,
        report_progress=lambda progress: progress_callback(
            _progress_percentage(progress)
        ),
        is_cancelled=cancel_signal.is_set,
        point_tracker_executor=run_legacy_point_tracker,
        bbox_tracker_executor=run_legacy_bbox_tracker,
        mask_tracker_executor=run_legacy_mask_tracker,
    )
    try:
        return run_seeded_tracking(request, context)
    except SeededTrackingCancelled:
        return None


def run_seeded_retrack_job(
    request: SeededRetrackRequest,
    image_reader: Callable[[Path], np.ndarray],
    cancel_signal: CancellationSignal,
    *,
    progress_callback: Callable[[int], None],
) -> Track | None:
    """Run public re-tracking and convert its single neutral identity for review."""
    context = RunContext(
        image_reader=image_reader,
        report_progress=lambda progress: progress_callback(
            _progress_percentage(progress)
        ),
        is_cancelled=cancel_signal.is_set,
        point_tracker_retrack_executor=run_legacy_point_retracker,
        bbox_tracker_retrack_executor=run_legacy_bbox_retracker,
        mask_tracker_retrack_executor=run_legacy_mask_retracker,
    )
    try:
        result = run_seeded_retrack(request, context)
    except SeededTrackingCancelled:
        return None
    if len(result.tracks) != 1:
        raise ValueError("Re-tracking must return exactly one tracked identity")
    return _track_from_identity(result.tracks[0])


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


def _observation_from_spot(frame_id: int, spot: SpotROI) -> TrackObservation:
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
    return Track(
        track_id=track.track_id,
        spots_by_image_id={
            frame_id: _clone_spot(spot)
            for frame_id, spot in track.spots_by_image_id.items()
        },
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


def _progress_percentage(progress: TrackingProgress) -> int:
    if progress.total <= 0:
        return 100
    return round(100 * progress.completed / progress.total)


def _stable_frame_ids(files: Sequence[LeedFile]) -> tuple[int, ...]:
    frame_ids = tuple(file.id for file in files)
    if any(frame_id is None for frame_id in frame_ids):
        raise ValueError("Seeded tracking requires every frame to have an ID")
    stable_ids = tuple(int(frame_id) for frame_id in frame_ids if frame_id is not None)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Seeded tracking requires unique frame IDs")
    return stable_ids


def _seed_representation(config: SeededTrackingConfig) -> SeedRepresentation:
    if isinstance(config, BoundingBoxTrackerConfig):
        return SeedRepresentation.BOUNDING_BOX
    if isinstance(config, MaskTrackerConfig):
        return SeedRepresentation.MASK
    return SeedRepresentation.POINT
