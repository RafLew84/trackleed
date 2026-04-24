from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from trackleed.core.data_models import SpotROI, Track
from trackleed.analysis.tracking_geometry import (
    angular_offset_deg,
    radial_unit_and_radius,
)
from trackleed.analysis.tracking_result_builders import set_spot_attributes
from trackleed.analysis.tracking_visibility_utils import (
    append_visible_track_spot,
    commit_tracking_visibility_state,
)
from trackleed.io.image_reader import load_tiff


def normalize_motion_filter_params(
    *,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_radial_jump_px: float,
    lateral_angle_tolerance_deg: float,
    min_directional_motion_px: float,
    direction_mode: str,
) -> dict[str, Any]:
    return {
        "max_step_distance_px": float(max(0.0, max_step_distance_px)),
        "max_reacquisition_gap": int(max(0, max_reacquisition_gap)),
        "terminate_after_invisible_frames": int(max(0, terminate_after_invisible_frames)),
        "max_radial_jump_px": float(max(0.0, max_radial_jump_px)),
        "lateral_angle_tolerance_deg": float(max(0.0, lateral_angle_tolerance_deg)),
        "min_directional_motion_px": float(max(0.0, min_directional_motion_px)),
        "direction_mode": str(direction_mode or "none").strip().lower(),
    }


def normalize_bbox_mask_filter_params(
    *,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
    max_bbox_center_jump_px: float,
) -> dict[str, Any]:
    return {
        "max_mask_area_ratio_change": float(max(0.0, max_mask_area_ratio_change)),
        "max_bbox_aspect_ratio_change": float(max(0.0, max_bbox_aspect_ratio_change)),
        "max_bbox_center_jump_px": float(max(0.0, max_bbox_center_jump_px)),
    }


@dataclass
class MotionFilterContext:
    last_valid_xy: np.ndarray | None
    invisible_streak: int
    terminated: bool
    max_step_distance_px: float
    max_reacquisition_gap: int
    terminate_after_invisible_frames: int
    max_radial_jump_px: float
    lateral_angle_tolerance_deg: float
    min_directional_motion_px: float
    direction_mode: str


@dataclass
class DirectionFilterContext:
    direction_mode: str
    center_mode: str
    manual_center_xy: Tuple[float, float] | None
    direction_center_xy: Tuple[float, float] | None
    lateral_angle_tolerance_deg: float
    max_radial_jump_px: float
    min_directional_motion_px: float


@dataclass
class MaskQualityFilterContext:
    max_mask_area_ratio_change: float
    max_bbox_aspect_ratio_change: float
    max_mask_components: int
    max_bbox_center_jump_px: float


@dataclass
class BBoxQualityFilterContext:
    max_bbox_aspect_ratio_change: float
    max_bbox_center_jump_px: float
    use_presence_score_gating: bool
    presence_threshold: float


def read_basic_motion_filter_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_step_distance_px": float(params.get("max_step_distance_px", 0.0) or 0.0),
        "max_reacquisition_gap": int(params.get("max_reacquisition_gap", 0) or 0),
        "terminate_after_invisible_frames": int(
            params.get("terminate_after_invisible_frames", 0) or 0
        ),
    }


def normalize_direction_filter_params(
    *,
    direction_mode: str,
    center_mode: str,
    manual_center: Any,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
) -> dict[str, Any]:
    manual_center_xy: Tuple[float, float] | None = None
    if manual_center is not None:
        try:
            manual_center_xy = (float(manual_center[0]), float(manual_center[1]))
        except Exception:
            manual_center_xy = None
    return {
        "direction_mode": str(direction_mode or "none").strip().lower(),
        "center_mode": str(center_mode or "auto").strip().lower(),
        "manual_center_xy": manual_center_xy,
        "lateral_angle_tolerance_deg": float(max(0.0, lateral_angle_tolerance_deg)),
        "max_radial_jump_px": float(max(0.0, max_radial_jump_px)),
        "min_directional_motion_px": float(max(0.0, min_directional_motion_px)),
    }


def read_direction_filter_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_direction_filter_params(
        direction_mode=str(params.get("direction_mode", "none")),
        center_mode=str(params.get("center_mode", "auto")),
        manual_center=params.get("manual_center"),
        lateral_angle_tolerance_deg=float(params.get("lateral_angle_tolerance_deg", 20.0)),
        max_radial_jump_px=float(params.get("max_radial_jump_px", 0.0)),
        min_directional_motion_px=float(params.get("min_directional_motion_px", 1.0)),
    )


def read_mask_quality_filter_params(params: Mapping[str, Any]) -> dict[str, Any]:
    bbox_mask_params = normalize_bbox_mask_filter_params(
        max_mask_area_ratio_change=float(params.get("max_mask_area_ratio_change", 0.0) or 0.0),
        max_bbox_aspect_ratio_change=float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0),
        max_bbox_center_jump_px=float(params.get("max_bbox_center_jump_px", 0.0) or 0.0),
    )
    bbox_mask_params["max_mask_components"] = int(max(0, params.get("max_mask_components", 0) or 0))
    return bbox_mask_params


def read_bbox_quality_filter_params(params: Mapping[str, Any]) -> dict[str, Any]:
    bbox_mask_params = normalize_bbox_mask_filter_params(
        max_mask_area_ratio_change=0.0,
        max_bbox_aspect_ratio_change=float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0),
        max_bbox_center_jump_px=float(params.get("max_bbox_center_jump_px", 0.0) or 0.0),
    )
    return {
        "max_bbox_aspect_ratio_change": bbox_mask_params["max_bbox_aspect_ratio_change"],
        "max_bbox_center_jump_px": bbox_mask_params["max_bbox_center_jump_px"],
        "use_presence_score_gating": bool(params.get("use_presence_score_gating", False)),
        "presence_threshold": float(max(0.0, params.get("presence_threshold", 0.0) or 0.0)),
    }


def resolve_direction_center_xy(
    *,
    all_files_map: Mapping[int, Any],
    subset_frame_ids: Sequence[int],
    direction_mode: str,
    center_mode: str,
    manual_center_xy: Tuple[float, float] | None,
    anchor_filepath: str | None,
    estimate_pattern_center_fn: Callable[..., Mapping[str, Any] | None],
) -> Tuple[float, float] | None:
    if direction_mode == "none":
        return None
    if center_mode == "manual" and manual_center_xy is not None:
        return manual_center_xy

    anchor_image = load_tiff(anchor_filepath) if anchor_filepath else None
    fallback_shape = None
    if anchor_image is not None and np.asarray(anchor_image).size:
        fallback_shape = (int(anchor_image.shape[0]), int(anchor_image.shape[1]))

    estimated_center = estimate_pattern_center_fn(
        all_files_map,
        frame_ids=list(subset_frame_ids),
        fallback_shape=fallback_shape,
    )
    if estimated_center is None:
        return None
    return (
        float(estimated_center["center_x"]),
        float(estimated_center["center_y"]),
    )


def build_direction_filter_context(
    params: Mapping[str, Any],
    *,
    all_files_map: Mapping[int, Any],
    subset_frame_ids: Sequence[int],
    anchor_filepath: str | None,
    estimate_pattern_center_fn: Callable[..., Mapping[str, Any] | None],
) -> DirectionFilterContext:
    direction_params = read_direction_filter_params(params)
    direction_center_xy = resolve_direction_center_xy(
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        direction_mode=direction_params["direction_mode"],
        center_mode=direction_params["center_mode"],
        manual_center_xy=direction_params["manual_center_xy"],
        anchor_filepath=anchor_filepath,
        estimate_pattern_center_fn=estimate_pattern_center_fn,
    )
    return DirectionFilterContext(
        direction_mode=direction_params["direction_mode"],
        center_mode=direction_params["center_mode"],
        manual_center_xy=direction_params["manual_center_xy"],
        direction_center_xy=direction_center_xy,
        lateral_angle_tolerance_deg=direction_params["lateral_angle_tolerance_deg"],
        max_radial_jump_px=direction_params["max_radial_jump_px"],
        min_directional_motion_px=direction_params["min_directional_motion_px"],
    )


def build_motion_tracking_state(
    xy: Tuple[float, float] | np.ndarray,
) -> Dict[str, Any]:
    return {
        "last_valid_xy": np.asarray(
            [float(xy[0]), float(xy[1])],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }


def build_bbox_mask_motion_tracking_state(
    xy: Tuple[float, float] | np.ndarray,
    *,
    last_valid_mask_area: float,
    last_valid_bbox_aspect_ratio: float | None,
    last_valid_bbox_center_xy: Tuple[float, float] | np.ndarray | None = None,
) -> Dict[str, Any]:
    state = build_motion_tracking_state(xy)
    center_xy = last_valid_bbox_center_xy
    if center_xy is None:
        center_xy = (float(xy[0]), float(xy[1]))
    state.update(
        {
            "last_valid_mask_area": float(last_valid_mask_area),
            "last_valid_bbox_aspect_ratio": last_valid_bbox_aspect_ratio,
            "last_valid_bbox_center_xy": np.asarray(center_xy, dtype=np.float32),
        }
    )
    return state


def build_mask_quality_filter_context(
    *,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
    max_mask_components: int,
    max_bbox_center_jump_px: float,
) -> MaskQualityFilterContext:
    normalized = normalize_bbox_mask_filter_params(
        max_mask_area_ratio_change=max_mask_area_ratio_change,
        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
    )
    return MaskQualityFilterContext(
        max_mask_area_ratio_change=normalized["max_mask_area_ratio_change"],
        max_bbox_aspect_ratio_change=normalized["max_bbox_aspect_ratio_change"],
        max_mask_components=int(max(0, max_mask_components)),
        max_bbox_center_jump_px=normalized["max_bbox_center_jump_px"],
    )


def build_bbox_quality_filter_context(
    *,
    max_bbox_aspect_ratio_change: float,
    max_bbox_center_jump_px: float,
    use_presence_score_gating: bool,
    presence_threshold: float,
) -> BBoxQualityFilterContext:
    normalized = normalize_bbox_mask_filter_params(
        max_mask_area_ratio_change=0.0,
        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
    )
    return BBoxQualityFilterContext(
        max_bbox_aspect_ratio_change=normalized["max_bbox_aspect_ratio_change"],
        max_bbox_center_jump_px=normalized["max_bbox_center_jump_px"],
        use_presence_score_gating=bool(use_presence_score_gating),
        presence_threshold=float(max(0.0, presence_threshold)),
    )


def build_motion_filter_context(
    state: Mapping[str, Any],
    *,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_radial_jump_px: float,
    lateral_angle_tolerance_deg: float,
    min_directional_motion_px: float,
    direction_mode: str,
) -> MotionFilterContext:
    filter_params = normalize_motion_filter_params(
        max_step_distance_px=max_step_distance_px,
        max_reacquisition_gap=max_reacquisition_gap,
        terminate_after_invisible_frames=terminate_after_invisible_frames,
        max_radial_jump_px=max_radial_jump_px,
        lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
        min_directional_motion_px=min_directional_motion_px,
        direction_mode=direction_mode,
    )
    last_valid_xy = state.get("last_valid_xy")
    if last_valid_xy is not None:
        last_valid_xy = np.asarray(last_valid_xy, dtype=np.float32)
    return MotionFilterContext(
        last_valid_xy=last_valid_xy,
        invisible_streak=int(state.get("invisible_streak", 0) or 0),
        terminated=bool(state.get("terminated", False)),
        max_step_distance_px=filter_params["max_step_distance_px"],
        max_reacquisition_gap=filter_params["max_reacquisition_gap"],
        terminate_after_invisible_frames=filter_params["terminate_after_invisible_frames"],
        max_radial_jump_px=filter_params["max_radial_jump_px"],
        lateral_angle_tolerance_deg=filter_params["lateral_angle_tolerance_deg"],
        min_directional_motion_px=filter_params["min_directional_motion_px"],
        direction_mode=filter_params["direction_mode"],
    )


def should_terminate_before_visible_candidate(
    state: MutableMapping[str, Any],
    context: MotionFilterContext,
) -> bool:
    if context.terminated:
        return True
    if exceeds_reacquisition_gap(
        invisible_streak=context.invisible_streak,
        max_reacquisition_gap=context.max_reacquisition_gap,
    ):
        context.terminated = True
        state["terminated"] = True
        return True
    return False


def register_motion_rejection(
    state: MutableMapping[str, Any],
    context: MotionFilterContext,
) -> bool:
    context.invisible_streak, should_terminate = advance_invisible_streak(
        invisible_streak=context.invisible_streak,
        terminate_after_invisible_frames=context.terminate_after_invisible_frames,
    )
    if should_terminate:
        context.terminated = True
        state["terminated"] = True
        return True
    return False


def append_motion_accepted_spot(
    track: Track,
    *,
    frame_id: int,
    spot: SpotROI,
    context: MotionFilterContext,
) -> np.ndarray:
    context.last_valid_xy = append_visible_track_spot(
        track,
        frame_id=frame_id,
        spot=spot,
    )
    context.invisible_streak = 0
    return context.last_valid_xy


def commit_motion_filter_context(
    state: MutableMapping[str, Any],
    context: MotionFilterContext,
    *,
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    commit_tracking_visibility_state(
        state,
        last_valid_xy=context.last_valid_xy,
        invisible_streak=context.invisible_streak,
        terminated=context.terminated,
        extra_state=extra_state,
    )


def exceeds_reacquisition_gap(
    *,
    invisible_streak: int,
    max_reacquisition_gap: int,
) -> bool:
    return max_reacquisition_gap > 0 and int(invisible_streak) > int(max_reacquisition_gap)


def advance_invisible_streak(
    *,
    invisible_streak: int,
    terminate_after_invisible_frames: int,
) -> tuple[int, bool]:
    next_streak = int(invisible_streak) + 1
    should_terminate = (
        int(terminate_after_invisible_frames) > 0
        and next_streak > int(terminate_after_invisible_frames)
    )
    return next_streak, should_terminate


def filter_rejection_attributes(prefix: str, reason: str | None) -> Dict[str, Any]:
    return {
        f"{prefix}_filter_rejected": True,
        f"{prefix}_filter_rejected_reason": str(reason) if reason else None,
    }


def mark_filter_rejected(
    spot: SpotROI | None,
    *,
    prefix: str,
    reason: str,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    if spot is None:
        return
    attributes = filter_rejection_attributes(prefix, reason)
    if extra_attributes:
        attributes.update(extra_attributes)
    set_spot_attributes(spot, attributes)


def gate_by_max_step_distance(
    candidate_spot: SpotROI | None,
    *,
    last_valid_xy: np.ndarray | None,
    max_step_distance_px: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if (
        candidate_spot is None
        or max_step_distance_px <= 0.0
        or last_valid_xy is None
    ):
        return True, None

    candidate_xy = np.array(
        [float(candidate_spot.center_x), float(candidate_spot.center_y)],
        dtype=np.float32,
    )
    if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
        reason = "max_step_distance"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason
    return True, None


def gate_by_direction(
    candidate_spot: SpotROI | None,
    state: Dict[str, Any],
    *,
    last_valid_xy: np.ndarray | None,
    direction_center_xy: Tuple[float, float] | None,
    direction_mode: str,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if (
        candidate_spot is None
        or direction_center_xy is None
        or last_valid_xy is None
        or direction_mode == "none"
    ):
        return True, None

    candidate_xy = np.array(
        [float(candidate_spot.center_x), float(candidate_spot.center_y)],
        dtype=np.float32,
    )
    delta = candidate_xy - last_valid_xy
    step_norm = float(np.linalg.norm(delta))
    radial_unit, last_radius = radial_unit_and_radius(
        direction_center_xy,
        (float(last_valid_xy[0]), float(last_valid_xy[1])),
    )
    _, candidate_radius = radial_unit_and_radius(
        direction_center_xy,
        (float(candidate_xy[0]), float(candidate_xy[1])),
    )

    if max_radial_jump_px > 0.0 and abs(candidate_radius - last_radius) > max_radial_jump_px:
        reason = "max_radial_jump"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason

    if step_norm < min_directional_motion_px:
        return True, None

    expected_sign = state.get("expected_direction_sign")
    if direction_mode == "inward":
        expected_sign = -1.0
    elif direction_mode == "outward":
        expected_sign = 1.0
    else:
        if expected_sign is None:
            radial_comp_seed = float(np.dot(delta, radial_unit))
            if abs(radial_comp_seed) >= min_directional_motion_px:
                expected_sign = float(np.sign(radial_comp_seed))
                state["expected_direction_sign"] = expected_sign

    if expected_sign is None:
        return True, None

    expected_vec = radial_unit * float(expected_sign)
    angle_deg = angular_offset_deg(expected_vec, delta)
    if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
        reason = "direction_gating"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason
    return True, None


def gate_by_mask_area_ratio(
    candidate_spot: SpotROI | None,
    *,
    current_mask_area: float,
    last_valid_mask_area: float | None,
    max_mask_area_ratio_change: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if (
        candidate_spot is None
        or max_mask_area_ratio_change <= 1.0
        or last_valid_mask_area is None
        or not np.isfinite(last_valid_mask_area)
        or last_valid_mask_area <= 0.0
        or not np.isfinite(current_mask_area)
        or current_mask_area <= 0.0
    ):
        return True, None

    area_ratio = max(float(current_mask_area), float(last_valid_mask_area)) / min(
        float(current_mask_area),
        float(last_valid_mask_area),
    )
    if area_ratio > max_mask_area_ratio_change:
        reason = "mask_area_drift"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason
    return True, None


def gate_by_bbox_aspect_ratio(
    candidate_spot: SpotROI | None,
    *,
    bbox_aspect_ratio: float | None,
    last_valid_bbox_aspect_ratio: float | None,
    max_bbox_aspect_ratio_change: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if (
        candidate_spot is None
        or max_bbox_aspect_ratio_change <= 1.0
        or last_valid_bbox_aspect_ratio is None
        or not np.isfinite(last_valid_bbox_aspect_ratio)
        or last_valid_bbox_aspect_ratio <= 0.0
        or bbox_aspect_ratio is None
        or not np.isfinite(bbox_aspect_ratio)
        or bbox_aspect_ratio <= 0.0
    ):
        return True, None

    aspect_ratio_change = max(
        float(bbox_aspect_ratio),
        float(last_valid_bbox_aspect_ratio),
    ) / min(
        float(bbox_aspect_ratio),
        float(last_valid_bbox_aspect_ratio),
    )
    if aspect_ratio_change > max_bbox_aspect_ratio_change:
        reason = "bbox_aspect_ratio_drift"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason
    return True, None


def gate_by_bbox_center_jump(
    candidate_spot: SpotROI | None,
    *,
    bbox_center_xy: np.ndarray | None,
    last_valid_bbox_center_xy: np.ndarray | None,
    max_bbox_center_jump_px: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if (
        candidate_spot is None
        or max_bbox_center_jump_px <= 0.0
        or last_valid_bbox_center_xy is None
        or bbox_center_xy is None
    ):
        return True, None

    if float(np.linalg.norm(bbox_center_xy - last_valid_bbox_center_xy)) > max_bbox_center_jump_px:
        reason = "bbox_jump"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason
    return True, None


def presence_score_is_accepted(
    score: float,
    *,
    use_presence_score_gating: bool,
    presence_threshold: float,
) -> bool:
    if not use_presence_score_gating:
        return True
    return bool(np.isfinite(score) and float(score) >= float(presence_threshold))


def gate_by_presence_score(
    candidate_spot: SpotROI | None,
    *,
    score: float,
    use_presence_score_gating: bool,
    presence_threshold: float,
    prefix: str,
) -> tuple[bool, str | None]:
    if presence_score_is_accepted(
        score,
        use_presence_score_gating=use_presence_score_gating,
        presence_threshold=presence_threshold,
    ):
        return True, None
    reason = "presence_score"
    mark_filter_rejected(
        candidate_spot,
        prefix=prefix,
        reason=reason,
        extra_attributes={f"{prefix}_bbox_score": float(score) if np.isfinite(score) else float("nan")},
    )
    return False, reason


def gate_mask_quality(
    candidate_spot: SpotROI | None,
    *,
    context: MaskQualityFilterContext,
    current_mask_area: float,
    last_valid_mask_area: float | None,
    mask_component_count: int,
    bbox_aspect_ratio: float | None,
    last_valid_bbox_aspect_ratio: float | None,
    bbox_center_xy: np.ndarray | None,
    last_valid_bbox_center_xy: np.ndarray | None,
    prefix: str,
) -> tuple[bool, str | None]:
    if candidate_spot is not None and context.max_mask_components > 0 and mask_component_count > context.max_mask_components:
        reason = "mask_fragmentation"
        mark_filter_rejected(candidate_spot, prefix=prefix, reason=reason)
        return False, reason

    accepted, reason = gate_by_mask_area_ratio(
        candidate_spot,
        current_mask_area=current_mask_area,
        last_valid_mask_area=last_valid_mask_area,
        max_mask_area_ratio_change=context.max_mask_area_ratio_change,
        prefix=prefix,
    )
    if not accepted:
        return False, reason

    accepted, reason = gate_by_bbox_aspect_ratio(
        candidate_spot,
        bbox_aspect_ratio=bbox_aspect_ratio,
        last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
        max_bbox_aspect_ratio_change=context.max_bbox_aspect_ratio_change,
        prefix=prefix,
    )
    if not accepted:
        return False, reason

    return gate_by_bbox_center_jump(
        candidate_spot,
        bbox_center_xy=bbox_center_xy,
        last_valid_bbox_center_xy=last_valid_bbox_center_xy,
        max_bbox_center_jump_px=context.max_bbox_center_jump_px,
        prefix=prefix,
    )


def gate_bbox_quality(
    candidate_spot: SpotROI | None,
    *,
    context: BBoxQualityFilterContext,
    score: float,
    bbox_aspect_ratio: float | None,
    last_valid_bbox_aspect_ratio: float | None,
    bbox_center_xy: np.ndarray | None,
    last_valid_bbox_center_xy: np.ndarray | None,
    prefix: str,
) -> tuple[bool, str | None]:
    accepted, reason = gate_by_presence_score(
        candidate_spot,
        score=score,
        use_presence_score_gating=context.use_presence_score_gating,
        presence_threshold=context.presence_threshold,
        prefix=prefix,
    )
    if not accepted:
        return False, reason

    accepted, reason = gate_by_bbox_aspect_ratio(
        candidate_spot,
        bbox_aspect_ratio=bbox_aspect_ratio,
        last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
        max_bbox_aspect_ratio_change=context.max_bbox_aspect_ratio_change,
        prefix=prefix,
    )
    if not accepted:
        return False, reason

    return gate_by_bbox_center_jump(
        candidate_spot,
        bbox_center_xy=bbox_center_xy,
        last_valid_bbox_center_xy=last_valid_bbox_center_xy,
        max_bbox_center_jump_px=context.max_bbox_center_jump_px,
        prefix=prefix,
    )
