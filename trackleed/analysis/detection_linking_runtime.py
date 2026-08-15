"""Headless runtime for detection-linking methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCancelled,
    DetectionLinkingConfig,
    DetectionLinkingProgress,
    DetectionLinkingRequest,
    HungarianLinkingConfig,
    RunContext,
)
from trackleed.logic.workflows.seeded_tracking import (
    SeedRect,
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)
from trackleed.logic.workflows.spot_detection import DetectedSpot, PreflightIssue


@dataclass
class _ActiveTrack:
    track_id: int
    last_xy: tuple[float, float]
    gap: int = 0
    observations: list[TrackObservation] = field(default_factory=list)


class HungarianLinkingRuntime:
    def validate_config(
        self,
        config: DetectionLinkingConfig,
    ) -> tuple[PreflightIssue, ...]:
        if not isinstance(config, HungarianLinkingConfig):
            return (PreflightIssue("invalid_config", "Hungarian configuration is required."),)
        if (
            not isinstance(config.max_distance_px, (int, float))
            or isinstance(config.max_distance_px, bool)
            or not isfinite(float(config.max_distance_px))
            or config.max_distance_px <= 0
            or type(config.max_gap) is not int
            or config.max_gap < 0
            or type(config.min_track_length) is not int
            or config.min_track_length < 1
        ):
            return (
                PreflightIssue(
                    "invalid_config",
                    "Hungarian settings are outside their valid ranges.",
                ),
            )
        return ()

    def run(
        self,
        request: DetectionLinkingRequest,
        context: RunContext,
    ) -> TrackingResult:
        config = request.config
        active: dict[int, _ActiveTrack] = {}
        finished: list[_ActiveTrack] = []
        next_track_id = 1
        total = len(request.frames)
        linking_source = f"detection_linking:{request.method_id}"
        for index, frame in enumerate(request.frames):
            _raise_if_cancelled(context)
            assigned_detections = self._assign(
                frame.frame_id,
                frame.spots,
                active,
                config,
                linking_source,
            )
            for detection_index, detection in enumerate(frame.spots):
                if detection_index in assigned_detections:
                    continue
                active[next_track_id] = _new_track(
                    next_track_id,
                    frame.frame_id,
                    detection,
                    linking_source,
                )
                next_track_id += 1
            finished.extend(_retire_expired(active, config))
            context.report_progress(
                DetectionLinkingProgress(index + 1, total, frame.frame_id)
            )
        _raise_if_cancelled(context)
        finished.extend(active.values())
        identities = tuple(
            TrackedSpotIdentity(track.track_id, tuple(track.observations))
            for track in sorted(finished, key=lambda item: item.track_id)
            if len(track.observations) >= config.min_track_length
        )
        return TrackingResult(identities)

    def _assign(
        self,
        frame_id: int,
        detections: tuple[DetectedSpot, ...],
        active: dict[int, _ActiveTrack],
        config: HungarianLinkingConfig,
        linking_source: str,
    ) -> set[int]:
        track_ids = tuple(active)
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        if track_ids and detections:
            track_points = np.asarray(
                [active[track_id].last_xy for track_id in track_ids],
                dtype=np.float64,
            )
            detection_points = np.asarray(
                [(spot.center_x, spot.center_y) for spot in detections],
                dtype=np.float64,
            )
            costs = np.linalg.norm(
                track_points[:, np.newaxis, :] - detection_points[np.newaxis, :, :],
                axis=2,
            )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] > config.max_distance_px:
                    continue
                track_id = track_ids[int(row)]
                detection = detections[int(column)]
                track = active[track_id]
                track.last_xy = (detection.center_x, detection.center_y)
                track.gap = 0
                track.observations.append(
                    _observation(frame_id, detection, linking_source)
                )
                assigned_tracks.add(track_id)
                assigned_detections.add(int(column))
        for track_id in track_ids:
            if track_id not in assigned_tracks:
                active[track_id].gap += 1
        return assigned_detections


def _new_track(
    track_id: int,
    frame_id: int,
    detection: DetectedSpot,
    linking_source: str,
) -> _ActiveTrack:
    return _ActiveTrack(
        track_id,
        (detection.center_x, detection.center_y),
        observations=[_observation(frame_id, detection, linking_source)],
    )


def _observation(
    frame_id: int,
    detection: DetectedSpot,
    linking_source: str,
) -> TrackObservation:
    rect = detection.roi
    return TrackObservation(
        frame_id=frame_id,
        roi=SeedRect(rect.x, rect.y, rect.width, rect.height),
        center_x=detection.center_x,
        center_y=detection.center_y,
        intensity=detection.intensity,
        source=linking_source,
        polygon_seed_points=detection.polygon_seed_points,
        diagnostics=(("detection_source", detection.source),),
    )


def _retire_expired(
    active: dict[int, _ActiveTrack],
    config: HungarianLinkingConfig,
) -> tuple[_ActiveTrack, ...]:
    expired_ids = tuple(
        track_id for track_id, track in active.items() if track.gap > config.max_gap
    )
    return tuple(active.pop(track_id) for track_id in expired_ids)


def _raise_if_cancelled(context: RunContext) -> None:
    if context.is_cancelled():
        raise DetectionLinkingCancelled("Detection linking was cancelled")
