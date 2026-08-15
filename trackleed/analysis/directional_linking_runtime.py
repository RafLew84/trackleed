"""Physics-informed directional adapter for detection linking."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackleed.analysis.tracking_geometry import (
    angular_offset_deg,
    estimate_pattern_center_from_point_sets,
    radial_unit_and_radius,
)
from trackleed.logic.workflows.detection_linking import (
    DetectionLinkingCancelled,
    DetectionLinkingMethodConfig,
    DetectionLinkingProgress,
    DetectionLinkingRequest,
    DirectionalLinkingConfig,
    RunContext,
)
from trackleed.logic.workflows.seeded_tracking import (
    SeedRect,
    TrackObservation,
    TrackedSpotIdentity,
    TrackingResult,
)
from trackleed.logic.workflows.spot_detection import (
    DetectedSpot,
    FrameDetections,
    PreflightIssue,
)


@dataclass
class _DirectionalTrack:
    track_id: int
    last_xy: tuple[float, float]
    gap: int = 0
    misses: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)
    observations: list[TrackObservation] = field(default_factory=list)


@dataclass
class _DirectionState:
    expected_sign: int | None
    flip_candidate: int | None = None
    flip_counter: int = 0


class DirectionalLinkingRuntime:
    def validate_config(
        self,
        config: DetectionLinkingMethodConfig,
    ) -> tuple[PreflightIssue, ...]:
        if not isinstance(config, DirectionalLinkingConfig):
            return (
                PreflightIssue(
                    "invalid_config",
                    "Directional configuration is required.",
                ),
            )
        if not _config_is_valid(config):
            return (
                PreflightIssue(
                    "invalid_config",
                    "Directional settings are outside their valid ranges.",
                ),
            )
        return ()

    def preflight(
        self,
        request: DetectionLinkingRequest,
    ) -> tuple[PreflightIssue, ...]:
        config = request.config
        if (
            isinstance(config, DirectionalLinkingConfig)
            and config.center_mode == "auto"
            and _estimate_pattern_center(request.frames) is None
        ):
            return (
                PreflightIssue(
                    "pattern_center_unavailable",
                    "Auto (RANSAC) could not determine the pattern center.",
                ),
            )
        return ()

    def run(
        self,
        request: DetectionLinkingRequest,
        context: RunContext,
    ) -> TrackingResult:
        config = request.config
        if not isinstance(config, DirectionalLinkingConfig):
            raise TypeError("Directional configuration is required")
        center = _configured_center(config, request.frames)
        active: dict[int, _DirectionalTrack] = {}
        finished: list[_DirectionalTrack] = []
        next_track_id = 1
        source = f"detection_linking:{request.method_id}"
        direction = _direction_state(config)
        for index, frame in enumerate(request.frames):
            _raise_if_cancelled(context)
            assigned = _assign_frame(
                frame.frame_id,
                frame.spots,
                active,
                config,
                center,
                source,
                direction.expected_sign,
            )
            for detection_index, detection in enumerate(frame.spots):
                if detection_index in assigned:
                    continue
                active[next_track_id] = _new_track(
                    next_track_id,
                    frame.frame_id,
                    detection,
                    source,
                )
                next_track_id += 1
            finished.extend(_retire_expired(active, config))
            _update_direction_state(direction, active, config, center)
            context.report_progress(
                DetectionLinkingProgress(index + 1, len(request.frames), frame.frame_id)
            )
        _raise_if_cancelled(context)
        finished.extend(active.values())
        return _result(finished, config.min_track_length)


def _config_is_valid(config: DirectionalLinkingConfig) -> bool:
    center_valid = config.center_mode in {"auto", "manual"}
    if config.center_mode == "manual":
        center_valid = center_valid and _center_is_valid(config.manual_center)
    return bool(
        center_valid
        and config.direction_mode in {"auto", "inward", "outward"}
        and _finite_between(config.lateral_angle_tolerance_deg, 0.0, 90.0)
        and _finite_above(config.ring_bin_px, 0.0)
        and _finite_above(config.base_max_distance_px, 0.0)
        and type(config.dynamic_max_distance) is bool
        and _finite_between(config.dynamic_margin_px, 0.0, float("inf"))
        and type(config.use_ring_penalty) is bool
        and type(config.max_gap) is int
        and config.max_gap >= 0
        and type(config.max_misses) is int
        and config.max_misses >= 0
        and type(config.min_track_length) is int
        and config.min_track_length >= 1
        and type(config.allow_direction_flip) is bool
        and type(config.min_tracks_for_flip) is int
        and config.min_tracks_for_flip >= 1
        and type(config.flip_hold_frames) is int
        and config.flip_hold_frames >= 1
        and _finite_between(
            config.min_radial_speed_for_flip,
            0.0,
            float("inf"),
        )
    )


def _configured_center(
    config: DirectionalLinkingConfig,
    frames: tuple[FrameDetections, ...],
) -> tuple[float, float]:
    if config.center_mode == "manual" and config.manual_center is not None:
        return (float(config.manual_center[0]), float(config.manual_center[1]))
    center = _estimate_pattern_center(frames)
    if center is None:
        raise ValueError("Could not determine pattern center for directional linking")
    return center


def _estimate_pattern_center(
    frames: tuple[FrameDetections, ...],
) -> tuple[float, float] | None:
    """Estimate one series center from per-frame RANSAC center candidates."""
    center = estimate_pattern_center_from_point_sets(
        tuple(
            tuple((spot.center_x, spot.center_y) for spot in frame.spots)
            for frame in frames
        )
    )
    if center is None:
        return None
    return (
        float(center["center_x"]),
        float(center["center_y"]),
    )


def _assign_frame(
    frame_id: int,
    detections: tuple[DetectedSpot, ...],
    active: dict[int, _DirectionalTrack],
    config: DirectionalLinkingConfig,
    center: tuple[float, float],
    source: str,
    expected_sign: int | None,
) -> set[int]:
    track_ids = tuple(active)
    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()
    if track_ids and detections:
        costs = _cost_matrix(
            track_ids,
            active,
            detections,
            config,
            center,
            expected_sign,
        )
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns, strict=True):
            if costs[row, column] >= _BLOCKED_COST:
                continue
            track_id = track_ids[int(row)]
            detection = detections[int(column)]
            _append_detection(active[track_id], frame_id, detection, source)
            assigned_tracks.add(track_id)
            assigned_detections.add(int(column))
    for track_id in track_ids:
        if track_id not in assigned_tracks:
            active[track_id].gap += 1
            active[track_id].misses += 1
    return assigned_detections


_BLOCKED_COST = 1e12


def _cost_matrix(
    track_ids: tuple[int, ...],
    active: dict[int, _DirectionalTrack],
    detections: tuple[DetectedSpot, ...],
    config: DirectionalLinkingConfig,
    center: tuple[float, float],
    expected_sign: int | None,
) -> np.ndarray:
    costs = np.full((len(track_ids), len(detections)), _BLOCKED_COST)
    for row, track_id in enumerate(track_ids):
        track = active[track_id]
        max_distance = _max_distance(track, config)
        for column, detection in enumerate(detections):
            cost = _assignment_cost(
                track,
                detection,
                config,
                center,
                max_distance,
                expected_sign,
            )
            if cost is not None:
                costs[row, column] = cost
    return costs


def _assignment_cost(
    track: _DirectionalTrack,
    detection: DetectedSpot,
    config: DirectionalLinkingConfig,
    center: tuple[float, float],
    max_distance: float,
    expected_sign: int | None,
) -> float | None:
    delta = np.asarray(
        (detection.center_x - track.last_xy[0], detection.center_y - track.last_xy[1]),
        dtype=np.float64,
    )
    distance = float(np.linalg.norm(delta))
    if distance > max_distance:
        return None
    radial_unit, last_radius = radial_unit_and_radius(center, track.last_xy)
    radial_component = float(np.dot(delta, radial_unit))
    lateral = float(max(0.0, distance**2 - radial_component**2) ** 0.5)
    radial_angle = angular_offset_deg(radial_unit, delta)
    axis_angle = min(radial_angle, 180.0 - radial_angle)
    lateral_penalty = max(0.0, axis_angle - config.lateral_angle_tolerance_deg)
    lateral_penalty *= 0.1 * (distance + 1.0)
    direction_penalty = _direction_penalty(radial_component, expected_sign)
    ring_penalty = _ring_penalty(last_radius, detection, center, config)
    return abs(radial_component) + lateral + lateral_penalty + direction_penalty + ring_penalty


def _direction_penalty(
    radial_component: float,
    expected_sign: int | None,
) -> float:
    if expected_sign is None or radial_component * expected_sign >= 0:
        return 0.0
    return abs(radial_component) * 2.0


def _direction_state(config: DirectionalLinkingConfig) -> _DirectionState:
    return _DirectionState({"inward": -1, "outward": 1}.get(config.direction_mode))


def _update_direction_state(
    state: _DirectionState,
    active: dict[int, _DirectionalTrack],
    config: DirectionalLinkingConfig,
    center: tuple[float, float],
) -> None:
    if config.direction_mode != "auto":
        return
    signs = _recent_radial_signs(active, config, center)
    if len(signs) < config.min_tracks_for_flip:
        return
    median_sign = int(np.sign(np.median(signs)))
    if median_sign == 0:
        state.flip_candidate = None
        state.flip_counter = 0
    elif state.expected_sign is None:
        state.expected_sign = median_sign
    elif config.allow_direction_flip and median_sign != state.expected_sign:
        _hold_flip_candidate(state, median_sign, config.flip_hold_frames)
    else:
        state.flip_candidate = None
        state.flip_counter = 0


def _recent_radial_signs(
    active: dict[int, _DirectionalTrack],
    config: DirectionalLinkingConfig,
    center: tuple[float, float],
) -> list[float]:
    signs: list[float] = []
    for track in active.values():
        if len(track.history) < 2:
            continue
        previous, current = track.history[-2:]
        radial_unit, _ = radial_unit_and_radius(center, previous)
        radial_speed = float(
            np.dot(np.asarray(current) - np.asarray(previous), radial_unit)
        )
        if abs(radial_speed) >= config.min_radial_speed_for_flip:
            signs.append(float(np.sign(radial_speed)))
    return signs


def _hold_flip_candidate(
    state: _DirectionState,
    candidate: int,
    hold_frames: int,
) -> None:
    if state.flip_candidate == candidate:
        state.flip_counter += 1
    else:
        state.flip_candidate = candidate
        state.flip_counter = 1
    if state.flip_counter >= hold_frames:
        state.expected_sign = candidate
        state.flip_candidate = None
        state.flip_counter = 0


def _ring_penalty(
    last_radius: float,
    detection: DetectedSpot,
    center: tuple[float, float],
    config: DirectionalLinkingConfig,
) -> float:
    if not config.use_ring_penalty:
        return 0.0
    _, detection_radius = radial_unit_and_radius(
        center,
        (detection.center_x, detection.center_y),
    )
    ring_delta = abs(
        _ring_index(detection_radius, config.ring_bin_px)
        - _ring_index(last_radius, config.ring_bin_px)
    )
    return ring_delta * max(1.0, config.ring_bin_px * 0.1)


def _max_distance(
    track: _DirectionalTrack,
    config: DirectionalLinkingConfig,
) -> float:
    if not config.dynamic_max_distance or len(track.history) < 2:
        return config.base_max_distance_px
    points = np.asarray(track.history, dtype=np.float64)
    step = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    return step + config.dynamic_margin_px if step > 0 else config.base_max_distance_px


def _append_detection(
    track: _DirectionalTrack,
    frame_id: int,
    detection: DetectedSpot,
    source: str,
) -> None:
    track.last_xy = (detection.center_x, detection.center_y)
    track.gap = 0
    track.misses = 0
    track.history.append(track.last_xy)
    track.observations.append(_observation(frame_id, detection, source))


def _new_track(
    track_id: int,
    frame_id: int,
    detection: DetectedSpot,
    source: str,
) -> _DirectionalTrack:
    point = (detection.center_x, detection.center_y)
    return _DirectionalTrack(
        track_id,
        point,
        history=[point],
        observations=[_observation(frame_id, detection, source)],
    )


def _observation(
    frame_id: int,
    detection: DetectedSpot,
    source: str,
) -> TrackObservation:
    rect = detection.roi
    return TrackObservation(
        frame_id,
        SeedRect(rect.x, rect.y, rect.width, rect.height),
        detection.center_x,
        detection.center_y,
        detection.intensity,
        source,
        detection.polygon_seed_points,
        (("detection_source", detection.source),),
    )


def _retire_expired(
    active: dict[int, _DirectionalTrack],
    config: DirectionalLinkingConfig,
) -> tuple[_DirectionalTrack, ...]:
    expired = tuple(
        track_id
        for track_id, track in active.items()
        if track.gap > config.max_gap or track.misses > config.max_misses
    )
    return tuple(active.pop(track_id) for track_id in expired)


def _result(
    tracks: list[_DirectionalTrack],
    min_track_length: int,
) -> TrackingResult:
    return TrackingResult(
        tuple(
            TrackedSpotIdentity(track.track_id, tuple(track.observations))
            for track in sorted(tracks, key=lambda item: item.track_id)
            if len(track.observations) >= min_track_length
        )
    )


def _center_is_valid(center: tuple[float, float] | None) -> bool:
    return bool(
        center is not None
        and len(center) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            for value in center
        )
    )


def _finite_above(value: object, minimum: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and float(value) > minimum
    )


def _finite_between(value: object, minimum: float, maximum: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _ring_index(radius: float, bin_size: float) -> int:
    return int(max(0.0, radius) // bin_size)


def _raise_if_cancelled(context: RunContext) -> None:
    if context.is_cancelled():
        raise DetectionLinkingCancelled("Detection linking was cancelled")
