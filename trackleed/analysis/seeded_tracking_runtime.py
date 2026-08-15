"""Headless runtime adapters for seeded optical-flow tracking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.morphology import disk, white_tophat
from skimage.util import img_as_ubyte

from trackleed.logic.workflows.seeded_tracking import (
    OpticalFlowConfig,
    OutOfBoundsPolicy,
    RunContext,
    SeededTrackingCancelled,
    SeededTrackingConfig,
    SeededTrackingRequest,
    SeedRect,
    SkipAnchorOpticalFlowConfig,
    TrackedSpotIdentity,
    TrackingPreflightIssue,
    TrackingProgress,
    TrackingResult,
    TrackObservation,
)


@dataclass
class _TrackingSetup:
    reference_index: int
    background: np.ndarray | None
    reference_image: np.ndarray
    observations: dict[int, list[TrackObservation]]


@dataclass
class _SkipAnchorState:
    anchor_indices: np.ndarray
    anchor_positions: np.ndarray
    alive: np.ndarray
    skip_counts: np.ndarray
    images: dict[int, np.ndarray]


class OpticalFlowRuntime:
    def preflight(
        self,
        config: SeededTrackingConfig,
        context: RunContext | None = None,
    ) -> tuple[TrackingPreflightIssue, ...]:
        if type(config) is not OpticalFlowConfig:
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    "Optical flow requires OpticalFlowConfig.",
                ),
            )
        if not _common_config_is_valid(config):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    "Optical-flow settings are outside their valid ranges.",
                ),
            )
        return ()

    def run(
        self,
        request: SeededTrackingRequest,
        context: RunContext,
    ) -> TrackingResult:
        config = request.config
        if type(config) is not OpticalFlowConfig:
            raise TypeError("Optical flow requires OpticalFlowConfig")
        setup = _prepare_tracking(request, context, config)
        frames = request.frames
        previous = setup.reference_image
        active_points = np.array(
            [[seed.center_x, seed.center_y] for seed in request.seeds],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        active_ids = [seed.seed_id for seed in request.seeds]
        total = len(frames) - setup.reference_index - 1
        for completed, frame in enumerate(
            frames[setup.reference_index + 1 :],
            start=1,
        ):
            _raise_if_cancelled(context)
            current = prepare_optical_flow_image(
                context.image_reader(frame.source_path),
                setup.background,
            )
            active_points, active_ids = _advance_points(
                previous,
                current,
                active_points,
                active_ids,
                frame.frame_id,
                setup.observations,
                config,
            )
            previous = current
            context.report_progress(TrackingProgress(completed, total, frame.frame_id))
        return TrackingResult(
            tuple(
                TrackedSpotIdentity(
                    seed.seed_id,
                    tuple(setup.observations[seed.seed_id]),
                )
                for seed in request.seeds
            )
        )


class SkipAnchorOpticalFlowRuntime:
    def preflight(
        self,
        config: SeededTrackingConfig,
        context: RunContext | None = None,
    ) -> tuple[TrackingPreflightIssue, ...]:
        if not isinstance(config, SkipAnchorOpticalFlowConfig):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    "Skip-anchor optical flow requires SkipAnchorOpticalFlowConfig.",
                ),
            )
        if (
            not _common_config_is_valid(config)
            or config.error_threshold < 0
            or config.forward_backward_threshold < 0
            or config.max_skip_frames < 0
            or not 0 <= config.ncc_threshold <= 1
            or config.ncc_patch_radius < 1
        ):
            return (
                TrackingPreflightIssue(
                    "invalid_config",
                    "Skip-anchor settings are outside their valid ranges.",
                ),
            )
        return ()

    def run(
        self,
        request: SeededTrackingRequest,
        context: RunContext,
    ) -> TrackingResult:
        config = request.config
        if not isinstance(config, SkipAnchorOpticalFlowConfig):
            raise TypeError("Skip-anchor optical flow requires its typed config")
        setup = _prepare_tracking(request, context, config)
        frame_subset = request.frames[setup.reference_index :]
        state = _SkipAnchorState(
            anchor_indices=np.zeros(len(request.seeds), dtype=int),
            anchor_positions=np.array(
                [[seed.center_x, seed.center_y] for seed in request.seeds],
                dtype=np.float32,
            ),
            alive=np.ones(len(request.seeds), dtype=bool),
            skip_counts=np.zeros(len(request.seeds), dtype=int),
            images={0: setup.reference_image},
        )
        total = len(frame_subset) - 1
        for target_index, frame in enumerate(frame_subset[1:], start=1):
            _raise_if_cancelled(context)
            current = prepare_optical_flow_image(
                context.image_reader(frame.source_path),
                setup.background,
            )
            state.images[target_index] = current
            _advance_skip_anchor_points(
                current,
                target_index,
                frame.frame_id,
                request,
                config,
                state,
                setup.observations,
            )
            needed = set(state.anchor_indices[state.alive]) | {target_index}
            state.images = {
                index: image for index, image in state.images.items() if index in needed
            }
            context.report_progress(
                TrackingProgress(target_index, total, frame.frame_id)
            )
        return _tracking_result(request, setup.observations)


def _prepare_tracking(
    request: SeededTrackingRequest,
    context: RunContext,
    config: OpticalFlowConfig,
) -> _TrackingSetup:
    reference_index = next(
        index
        for index, frame in enumerate(request.frames)
        if frame.frame_id == request.reference_frame_id
    )
    background = (
        context.image_reader(request.background_path)
        if config.subtract_background and request.background_path is not None
        else None
    )
    reference = prepare_optical_flow_image(
        context.image_reader(request.frames[reference_index].source_path),
        background,
    )
    observations = {
        seed.seed_id: [
            TrackObservation(
                request.reference_frame_id,
                seed.roi,
                seed.center_x,
                seed.center_y,
                seed.intensity,
                seed.source,
            )
        ]
        for seed in request.seeds
    }
    return _TrackingSetup(reference_index, background, reference, observations)


def _tracking_result(
    request: SeededTrackingRequest,
    observations: dict[int, list[TrackObservation]],
) -> TrackingResult:
    return TrackingResult(
        tuple(
            TrackedSpotIdentity(seed.seed_id, tuple(observations[seed.seed_id]))
            for seed in request.seeds
        )
    )


def prepare_optical_flow_image(
    image: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Could not load tracking image")
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        image_float = image.astype(np.float32)
        minimum, maximum = np.min(image_float), np.max(image_float)
        gray = (
            (image_float - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(image_float)
        )
    working = gray.copy()
    if background is not None:
        normalized_background = normalized_grayscale(background)
        if normalized_background.shape == working.shape:
            working -= normalized_background
            np.clip(working, 0.0, None, out=working)
    residual = gaussian_filter(working, sigma=5)
    enhanced = white_tophat(working - residual, footprint=disk(5))
    minimum, maximum = np.min(enhanced), np.max(enhanced)
    normalized = (
        (enhanced - minimum) / (maximum - minimum)
        if maximum > minimum
        else np.zeros_like(enhanced)
    )
    return img_as_ubyte(normalized)


def _advance_points(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    track_ids: list[int],
    frame_id: int,
    observations: dict[int, list[TrackObservation]],
    config: OpticalFlowConfig,
) -> tuple[np.ndarray, list[int]]:
    if not track_ids:
        return points, track_ids
    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        **_lk_parameters(config),  # type: ignore[call-overload]
    )
    if next_points is None or status is None:
        return np.empty((0, 1, 2), dtype=np.float32), []
    survivors = [index for index, value in enumerate(status.reshape(-1)) if value == 1]
    surviving_points = next_points[survivors]
    surviving_ids = [track_ids[index] for index in survivors]
    for index, (track_id, point) in enumerate(zip(surviving_ids, surviving_points)):
        x, y = map(float, point.reshape(-1)[:2])
        if config.use_centroid_refinement:
            refined = refine_center_by_centroid(
                current,
                np.array([x, y], dtype=np.float32),
                max(1.0, config.roi_size / 2 * config.centroid_patch_multiplier),
                config.centroid_rel_threshold,
                config.centroid_max_expand,
            )
            x, y = map(float, refined)
            surviving_points[index] = refined
        observations[track_id].append(
            TrackObservation(
                frame_id,
                _tracked_rect(x, y, config.roi_size),
                x,
                y,
                source="optical_flow",
            )
        )
    return surviving_points.reshape(-1, 1, 2), surviving_ids


def _advance_skip_anchor_points(
    current: np.ndarray,
    target_index: int,
    frame_id: int,
    request: SeededTrackingRequest,
    config: SkipAnchorOpticalFlowConfig,
    state: _SkipAnchorState,
    observations: dict[int, list[TrackObservation]],
) -> None:
    for seed_index, seed in enumerate(request.seeds):
        if not state.alive[seed_index]:
            continue
        anchor = state.images[int(state.anchor_indices[seed_index])]
        matched, point, out_of_bounds = _match_skip_anchor(
            anchor,
            current,
            state.anchor_positions[seed_index],
            config,
        )
        if matched:
            x, y = map(float, point)
            observations[seed.seed_id].append(
                TrackObservation(
                    frame_id,
                    _tracked_rect(x, y, config.roi_size),
                    x,
                    y,
                    source="optical_flow_skip_anchor",
                )
            )
            state.anchor_indices[seed_index] = target_index
            state.anchor_positions[seed_index] = point
            state.skip_counts[seed_index] = 0
        elif out_of_bounds and config.out_of_bounds is OutOfBoundsPolicy.MARK_LOST:
            state.alive[seed_index] = False
        elif config.max_skip_frames > 0:
            state.skip_counts[seed_index] += 1
            if state.skip_counts[seed_index] > config.max_skip_frames:
                state.alive[seed_index] = False


def _match_skip_anchor(
    anchor: np.ndarray,
    current: np.ndarray,
    anchor_position: np.ndarray,
    config: SkipAnchorOpticalFlowConfig,
) -> tuple[bool, np.ndarray, bool]:
    point0 = anchor_position.reshape(1, 1, 2).astype(np.float32)
    point1, status, error = cv2.calcOpticalFlowPyrLK(
        anchor,
        current,
        point0,
        None,
        **_lk_parameters(config),  # type: ignore[call-overload]
    )
    if point1 is None or status is None or status[0, 0] != 1:
        return False, anchor_position, False
    point = point1[0, 0]
    if error is not None and float(error[0, 0]) > config.error_threshold:
        return False, anchor_position, False
    if config.use_forward_backward and not _forward_backward_matches(
        anchor,
        current,
        point0,
        point1,
        config,
    ):
        return False, anchor_position, False
    x, y = map(float, point)
    in_bounds = 0 <= x < current.shape[1] and 0 <= y < current.shape[0]
    if not in_bounds:
        return False, point, True
    if (
        config.use_ncc_check
        and ncc_around(
            anchor,
            current,
            point0[0, 0],
            point,
            config.ncc_patch_radius,
        )
        < config.ncc_threshold
    ):
        return False, anchor_position, False
    if config.use_centroid_refinement:
        point = refine_center_by_centroid(
            current,
            point,
            max(1.0, config.roi_size / 2 * config.centroid_patch_multiplier),
            config.centroid_rel_threshold,
            config.centroid_max_expand,
        )
    return True, point, False


def _forward_backward_matches(
    anchor: np.ndarray,
    current: np.ndarray,
    point0: np.ndarray,
    point1: np.ndarray,
    config: SkipAnchorOpticalFlowConfig,
) -> bool:
    returned, status, _error = cv2.calcOpticalFlowPyrLK(
        current,
        anchor,
        point1,
        None,
        **_lk_parameters(config),  # type: ignore[call-overload]
    )
    if returned is None or status is None or status[0, 0] != 1:
        return False
    return float(np.linalg.norm(returned[0, 0] - point0[0, 0])) <= (
        config.forward_backward_threshold
    )


def ncc_around(
    first: np.ndarray,
    second: np.ndarray,
    first_point: np.ndarray,
    second_point: np.ndarray,
    radius: int,
) -> float:
    first_x, first_y = map(int, np.round(first_point))
    second_x, second_y = map(int, np.round(second_point))
    if not _patch_fits(first, first_x, first_y, radius):
        return -1.0
    if not _patch_fits(second, second_x, second_y, radius):
        return -1.0
    first_patch = first[
        first_y - radius : first_y + radius + 1,
        first_x - radius : first_x + radius + 1,
    ].astype(np.float32)
    second_patch = second[
        second_y - radius : second_y + radius + 1,
        second_x - radius : second_x + radius + 1,
    ].astype(np.float32)
    first_patch -= first_patch.mean()
    second_patch -= second_patch.mean()
    denominator = np.linalg.norm(first_patch) * np.linalg.norm(second_patch) + 1e-6
    return float((first_patch * second_patch).sum() / denominator)


def _patch_fits(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> bool:
    return (
        center_x - radius >= 0
        and center_y - radius >= 0
        and center_x + radius < image.shape[1]
        and center_y + radius < image.shape[0]
    )


def _common_config_is_valid(config: OpticalFlowConfig) -> bool:
    return (
        config.roi_size >= 1
        and config.win_size >= 1
        and config.win_size % 2 == 1
        and config.max_level >= 0
        and config.max_iterations >= 1
        and config.epsilon > 0
        and config.min_eigen_threshold >= 0
        and config.centroid_patch_multiplier > 0
        and 0 <= config.centroid_rel_threshold <= 1
        and config.centroid_max_expand >= 0
    )


def _lk_parameters(config: OpticalFlowConfig) -> dict[str, object]:
    return {
        "winSize": (config.win_size, config.win_size),
        "maxLevel": config.max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.max_iterations,
            config.epsilon,
        ),
        "minEigThreshold": config.min_eigen_threshold,
    }


def _tracked_rect(
    center_x: float,
    center_y: float,
    roi_size: int,
) -> SeedRect:
    half_size = roi_size // 2
    return SeedRect(
        int(center_x - half_size),
        int(center_y - half_size),
        roi_size,
        roi_size,
    )


def normalized_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return rgb2gray(image)
    image_float = image.astype(np.float32)
    minimum, maximum = np.min(image_float), np.max(image_float)
    if maximum > minimum:
        return (image_float - minimum) / (maximum - minimum)
    return np.zeros_like(image_float)


def refine_center_by_centroid(
    image: np.ndarray,
    approximate_xy: np.ndarray,
    patch_radius: float,
    relative_threshold: float,
    max_expand: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = map(int, np.round(approximate_xy))
    radius = max(1, int(patch_radius))
    for _attempt in range(max_expand + 1):
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        patch = image[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return approximate_xy
        threshold = max(1.0, relative_threshold * patch.max())
        mask = (patch >= threshold).astype(np.float32)
        if mask.sum() < 10:
            _, mask = cv2.threshold(
                patch.astype(np.uint8),
                0,
                1,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        if mask.sum() < 10:
            radius = int(radius * 1.5)
            continue
        ys, xs = np.nonzero(mask)
        values = patch[ys, xs]
        weights = values / (values.sum() + 1e-6)
        local_x = float((xs * weights).sum())
        local_y = float((ys * weights).sum())
        if (
            abs(local_x - (x1 - x0) / 2) > radius - 2
            or abs(local_y - (y1 - y0) / 2) > radius - 2
        ):
            radius = int(radius * 1.5)
            continue
        return np.array([x0 + local_x, y0 + local_y], dtype=np.float32)
    return approximate_xy


def _raise_if_cancelled(context: RunContext) -> None:
    if context.is_cancelled():
        raise SeededTrackingCancelled("Seeded tracking cancelled")
