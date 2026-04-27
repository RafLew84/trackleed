from __future__ import annotations

from importlib import import_module
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trackleed.analysis.cutie_backend import (
    DEFAULT_CUTIE_MODELS_DIR,
    CutieRuntimeSettings,
    discover_cutie_checkpoints,
    run_cutie_offline,
    select_default_cutie_checkpoint,
)
from trackleed.analysis.deva_backend import (
    DEFAULT_DEVA_MODELS_DIR,
    DevaRuntimeSettings,
    discover_deva_checkpoints,
    run_deva_offline,
    select_default_deva_checkpoint,
)
from trackleed.analysis.dam4sam_backend import (
    DEFAULT_DAM4SAM_MODELS_DIR,
    DEFAULT_DAM4SAM_TRACKER_NAME,
    Dam4SamRuntimeSettings,
    discover_dam4sam_checkpoints,
    run_dam4sam_offline,
    select_default_dam4sam_checkpoint,
)
from trackleed.analysis.samurai_backend import (
    DEFAULT_SAMURAI_MODEL_VARIANT,
    DEFAULT_SAMURAI_MODELS_DIR,
    SamuraiRuntimeSettings,
    discover_samurai_checkpoints,
    run_samurai_offline,
    select_default_samurai_checkpoint,
)
from trackleed.analysis.polygon_mask import polygon_points_are_valid
from trackleed.analysis.tracking_family_adapters import (
    build_mask_family_adapter_context,
)
from trackleed.analysis.sam2_backend import (
    DEFAULT_SAM2_MODELS_DIR,
    Sam2RuntimeSettings,
    discover_sam2_checkpoints,
    run_sam2_offline,
    select_default_sam2_checkpoint,
)
from trackleed.analysis.tracking_filter_utils import (
    append_motion_accepted_spot,
    build_mask_quality_filter_context,
    build_bbox_mask_motion_tracking_state,
    build_motion_filter_context,
    commit_motion_filter_context,
    gate_by_direction,
    gate_by_max_step_distance,
    gate_mask_quality,
    register_motion_rejection,
    should_terminate_before_visible_candidate,
)
from trackleed.analysis.tracking_retrack_utils import (
    finish_retrack_if_too_short,
    prepare_retrack_start_context,
)
from trackleed.analysis.tracking_result_builders import (
    apply_bbox_xywh_metadata,
    apply_bbox_xyxy_metadata,
    apply_prefixed_point_metadata,
    bbox_center_and_aspect_ratio_from_xyxy as _bbox_center_and_aspect_ratio_from_xyxy,
    build_spot_roi,
    set_spot_attributes,
)
from trackleed.analysis.tracking_visibility_utils import append_invisible_track_frame
from trackleed.analysis.tracking_windowing_utils import (
    advance_window_coverage,
    build_reseed_tracking_window_context,
)
from trackleed.analysis.xmem2_backend import (
    DEFAULT_XMEM2_MODELS_DIR,
    XMem2RuntimeSettings,
    discover_xmem2_checkpoints,
    run_xmem2_offline,
    select_default_xmem2_checkpoint,
)
from trackleed.core.data_models import LeedFile, SpotROI, Track


def _tracking_support():
    return import_module("trackleed.analysis.tracking")


def _build_reference_mask_tracking_state(reference_spot: SpotROI) -> Dict[str, Any]:
    rect = reference_spot.roi_rect
    return build_bbox_mask_motion_tracking_state(
        (float(reference_spot.center_x), float(reference_spot.center_y)),
        last_valid_mask_area=float(rect.width() * rect.height()),
        last_valid_bbox_aspect_ratio=(
            float(rect.width()) / float(rect.height())
            if rect.height() > 0
            else None
        ),
        last_valid_bbox_center_xy=(
            float(reference_spot.center_x),
            float(reference_spot.center_y),
        ),
    )


def _build_mask_tracking_state_from_values(
    spot: SpotROI,
    *,
    mask_area: float,
    bbox_aspect_ratio: float | None,
    bbox_center_xy: Tuple[float, float] | np.ndarray | None,
) -> Dict[str, Any]:
    return build_bbox_mask_motion_tracking_state(
        (float(spot.center_x), float(spot.center_y)),
        last_valid_mask_area=float(mask_area),
        last_valid_bbox_aspect_ratio=bbox_aspect_ratio,
        last_valid_bbox_center_xy=bbox_center_xy,
    )


def _select_dam4sam_checkpoint_path(
    params: dict,
    *,
    repo_path: str | Path | None,
    tracker_name: str,
) -> str:
    checkpoint_path = params.get("checkpoint_path") or params.get("dam4sam_checkpoint_path")
    if checkpoint_path:
        return str(checkpoint_path)

    models_dir = Path(params.get("dam4sam_models_dir", DEFAULT_DAM4SAM_MODELS_DIR))
    detected_checkpoints = discover_dam4sam_checkpoints(
        models_dir,
        repo_path=repo_path,
    )
    selected_checkpoint = select_default_dam4sam_checkpoint(
        detected_checkpoints,
        tracker_name=tracker_name,
        models_dir=models_dir,
    )
    if selected_checkpoint is None:
        raise ValueError(
            "No DAM4SAM checkpoint selected. Place a SAM2.1 checkpoint such as "
            "`sam2.1_hiera_large.pt` in tracking_models/dam4sam or the local "
            "repo/checkpoints and pass checkpoint_path in params."
        )
    return str(selected_checkpoint)


def _build_dam4sam_runtime_settings(
    params: dict,
    *,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    tracker_name: str,
) -> Dam4SamRuntimeSettings:
    return Dam4SamRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        tracker_name=tracker_name,
        device=params.get("device", "auto"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )


def _select_samurai_checkpoint_path(
    params: dict,
    *,
    repo_path: str | Path | None,
    model_variant: str,
) -> str:
    checkpoint_path = params.get("checkpoint_path") or params.get("samurai_checkpoint_path")
    if checkpoint_path:
        return str(checkpoint_path)

    models_dir = Path(params.get("samurai_models_dir", DEFAULT_SAMURAI_MODELS_DIR))
    detected_checkpoints = discover_samurai_checkpoints(
        models_dir,
        repo_path=repo_path,
    )
    selected_checkpoint = select_default_samurai_checkpoint(
        detected_checkpoints,
        model_variant=model_variant,
        models_dir=models_dir,
    )
    if selected_checkpoint is None:
        raise ValueError(
            "No SAMURAI checkpoint selected. Place a SAM2.1 checkpoint such as "
            "`sam2.1_hiera_base_plus.pt` in tracking_models/samurai or the local "
            "repo/sam2/checkpoints and pass checkpoint_path in params."
        )
    return str(selected_checkpoint)


def _build_samurai_runtime_settings(
    params: dict,
    *,
    repo_path: str | Path | None,
    checkpoint_path: str | Path,
    model_variant: str,
) -> SamuraiRuntimeSettings:
    return SamuraiRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_variant=model_variant,
        device=params.get("device", "auto"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )


def _build_dam4sam_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_bbox_xywh: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    mask_component_count: int,
    representative_point_mode: str,
    tracker_name: str,
    checkpoint_name: str,
    config_name: str,
) -> SpotROI:
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _tracking_support()._roi_rect_from_bbox_xyxy(
        mask_bbox_xyxy if visible else None,
        image_shape,
        fallback_center_x=x,
        fallback_center_y=y,
        fallback_roi_size=roi_size,
    )
    spot = build_spot_roi(
        spot_id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        source="dam4sam_offline",
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix="dam4sam",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "dam4sam_mask_area": float(mask_area),
            "dam4sam_mask_score": float(mask_score),
            "dam4sam_mask_component_count": int(mask_component_count),
            "dam4sam_repr_mode": str(representative_point_mode),
            "dam4sam_tracker_name": str(tracker_name),
            "dam4sam_checkpoint_name": str(checkpoint_name),
            "dam4sam_config_name": str(config_name),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="dam4sam_mask_bbox_xyxy",
        aspect_ratio_attribute_name="dam4sam_mask_bbox_aspect_ratio",
    )
    apply_bbox_xywh_metadata(
        spot,
        bbox_xywh=mask_bbox_xywh,
        bbox_attribute_name="dam4sam_mask_bbox_xywh",
    )
    return spot


def _build_samurai_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_bbox_xywh: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    mask_component_count: int,
    representative_point_mode: str,
    model_variant: str,
    checkpoint_name: str,
    config_name: str,
) -> SpotROI:
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _tracking_support()._roi_rect_from_bbox_xyxy(
        mask_bbox_xyxy if visible else None,
        image_shape,
        fallback_center_x=x,
        fallback_center_y=y,
        fallback_roi_size=roi_size,
    )
    spot = build_spot_roi(
        spot_id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        source="samurai_offline",
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix="samurai",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "samurai_mask_area": float(mask_area),
            "samurai_mask_score": float(mask_score),
            "samurai_mask_component_count": int(mask_component_count),
            "samurai_repr_mode": str(representative_point_mode),
            "samurai_model_variant": str(model_variant),
            "samurai_checkpoint_name": str(checkpoint_name),
            "samurai_config_name": str(config_name),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="samurai_mask_bbox_xyxy",
        aspect_ratio_attribute_name="samurai_mask_bbox_aspect_ratio",
    )
    apply_bbox_xywh_metadata(
        spot,
        bbox_xywh=mask_bbox_xywh,
        bbox_attribute_name="samurai_mask_bbox_xywh",
    )
    return spot


def _count_dam4sam_mask_components(mask: np.ndarray) -> int:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not bool(np.any(mask_bool)):
        return 0

    try:
        import cv2  # type: ignore

        if hasattr(cv2, "connectedComponents"):
            component_count, _labels = cv2.connectedComponents(
                mask_bool.astype(np.uint8),
                connectivity=8,
            )
            return int(max(0, int(component_count) - 1))
    except Exception:
        pass

    visited = np.zeros(mask_bool.shape, dtype=bool)
    height, width = mask_bool.shape
    components = 0
    ys, xs = np.nonzero(mask_bool)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        components += 1
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if (
                        not visited[next_y, next_x]
                        and bool(mask_bool[next_y, next_x])
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
    return int(components)


def _append_dam4sam_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_bboxes_xywh: np.ndarray,
    query_mask_scores: np.ndarray,
    query_masks: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    tracker_name: str,
    checkpoint_name: str,
    config_name: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
    max_mask_components: int,
    max_bbox_center_jump_px: float,
    direction_center_xy: Tuple[float, float] | None,
    direction_mode: str,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
    cancel_event: Optional[Event] = None,
    start_local_idx: int = 1,
    min_global_write_idx: int | None = None,
    global_start_idx: int = 0,
) -> bool:
    motion_context = build_motion_filter_context(
        state,
        max_step_distance_px=max_step_distance_px,
        max_reacquisition_gap=max_reacquisition_gap,
        terminate_after_invisible_frames=terminate_after_invisible_frames,
        max_radial_jump_px=max_radial_jump_px,
        lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
        min_directional_motion_px=min_directional_motion_px,
        direction_mode=direction_mode,
    )
    last_valid_xy = motion_context.last_valid_xy
    last_valid_mask_area = state["last_valid_mask_area"]
    last_valid_bbox_aspect_ratio = state["last_valid_bbox_aspect_ratio"]
    last_valid_bbox_center_xy = state["last_valid_bbox_center_xy"]
    quality_context = build_mask_quality_filter_context(
        max_mask_area_ratio_change=max_mask_area_ratio_change,
        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
        max_mask_components=max_mask_components,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
    )

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        image_h, image_w = frame_shapes[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        mask_area = float(query_mask_areas[local_idx])
        mask_bbox_xyxy = query_mask_bboxes_xyxy[local_idx]
        mask_bbox_xywh = query_mask_bboxes_xywh[local_idx]
        mask_score = float(query_mask_scores[local_idx])
        mask_component_count = (
            _count_dam4sam_mask_components(query_masks[local_idx])
            if visible
            else 0
        )
        bbox_center_xy, bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(mask_bbox_xyxy)

        candidate_spot: SpotROI | None = None
        rejection_reason = "invalid_mask"
        accepted = visible
        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                accepted = False
                rejection_reason = "invalid_mask"
            elif not (0.0 <= raw_x < float(image_w) and 0.0 <= raw_y < float(image_h)):
                accepted = False
                rejection_reason = "out_of_bounds"

        if accepted:
            candidate_spot = _build_dam4sam_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_bbox_xywh,
                mask_score,
                mask_component_count,
                representative_point_mode,
                tracker_name,
                checkpoint_name,
                config_name,
            )

            accepted, rejection_reason = gate_mask_quality(
                candidate_spot,
                context=quality_context,
                current_mask_area=mask_area,
                last_valid_mask_area=last_valid_mask_area,
                mask_component_count=mask_component_count,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="dam4sam",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="dam4sam",
                )
            if accepted:
                accepted, rejection_reason = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="dam4sam",
                )

        if keep_invisible_as_gaps:
            if not accepted and register_motion_rejection(state, motion_context):
                return True
            if not accepted:
                continue

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            fallback_x = raw_x if np.isfinite(raw_x) else (
                float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
            )
            fallback_y = raw_y if np.isfinite(raw_y) else (
                float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
            )
            forced_spot = candidate_spot or _build_dam4sam_spot(
                reference_spot,
                frame_shapes[local_idx],
                fallback_x,
                fallback_y,
                False,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_bbox_xywh,
                mask_score,
                mask_component_count,
                representative_point_mode,
                tracker_name,
                checkpoint_name,
                config_name,
            )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="dam4sam_visibility",
                extra_attributes={
                    "dam4sam_filter_rejected": True,
                    "dam4sam_filter_rejected_reason": rejection_reason,
                },
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )
        if np.isfinite(mask_area) and mask_area > 0.0:
            last_valid_mask_area = float(mask_area)
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)

    commit_motion_filter_context(
        state,
        motion_context,
        extra_state={
            "last_valid_mask_area": last_valid_mask_area,
            "last_valid_bbox_aspect_ratio": last_valid_bbox_aspect_ratio,
            "last_valid_bbox_center_xy": last_valid_bbox_center_xy,
        },
    )
    return True


def _append_samurai_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_bboxes_xywh: np.ndarray,
    query_mask_scores: np.ndarray,
    query_masks: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    model_variant: str,
    checkpoint_name: str,
    config_name: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
    max_mask_components: int,
    max_bbox_center_jump_px: float,
    direction_center_xy: Tuple[float, float] | None,
    direction_mode: str,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
    cancel_event: Optional[Event] = None,
    start_local_idx: int = 1,
    min_global_write_idx: int | None = None,
    global_start_idx: int = 0,
) -> bool:
    motion_context = build_motion_filter_context(
        state,
        max_step_distance_px=max_step_distance_px,
        max_reacquisition_gap=max_reacquisition_gap,
        terminate_after_invisible_frames=terminate_after_invisible_frames,
        max_radial_jump_px=max_radial_jump_px,
        lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
        min_directional_motion_px=min_directional_motion_px,
        direction_mode=direction_mode,
    )
    last_valid_xy = motion_context.last_valid_xy
    last_valid_mask_area = state["last_valid_mask_area"]
    last_valid_bbox_aspect_ratio = state["last_valid_bbox_aspect_ratio"]
    last_valid_bbox_center_xy = state["last_valid_bbox_center_xy"]
    quality_context = build_mask_quality_filter_context(
        max_mask_area_ratio_change=max_mask_area_ratio_change,
        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
        max_mask_components=max_mask_components,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
    )

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        image_h, image_w = frame_shapes[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        mask_area = float(query_mask_areas[local_idx])
        mask_bbox_xyxy = query_mask_bboxes_xyxy[local_idx]
        mask_bbox_xywh = query_mask_bboxes_xywh[local_idx]
        mask_score = float(query_mask_scores[local_idx])
        mask_component_count = (
            _count_dam4sam_mask_components(query_masks[local_idx])
            if visible
            else 0
        )
        bbox_center_xy, bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(mask_bbox_xyxy)

        candidate_spot: SpotROI | None = None
        rejection_reason = "invalid_mask"
        accepted = visible
        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                accepted = False
                rejection_reason = "invalid_mask"
            elif not (0.0 <= raw_x < float(image_w) and 0.0 <= raw_y < float(image_h)):
                accepted = False
                rejection_reason = "out_of_bounds"

        if accepted:
            candidate_spot = _build_samurai_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_bbox_xywh,
                mask_score,
                mask_component_count,
                representative_point_mode,
                model_variant,
                checkpoint_name,
                config_name,
            )

            accepted, rejection_reason = gate_mask_quality(
                candidate_spot,
                context=quality_context,
                current_mask_area=mask_area,
                last_valid_mask_area=last_valid_mask_area,
                mask_component_count=mask_component_count,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="samurai",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="samurai",
                )
            if accepted:
                accepted, rejection_reason = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="samurai",
                )

        if keep_invisible_as_gaps:
            if not accepted and register_motion_rejection(state, motion_context):
                return True
            if not accepted:
                continue

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            fallback_x = raw_x if np.isfinite(raw_x) else (
                float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
            )
            fallback_y = raw_y if np.isfinite(raw_y) else (
                float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
            )
            forced_spot = candidate_spot or _build_samurai_spot(
                reference_spot,
                frame_shapes[local_idx],
                fallback_x,
                fallback_y,
                False,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_bbox_xywh,
                mask_score,
                mask_component_count,
                representative_point_mode,
                model_variant,
                checkpoint_name,
                config_name,
            )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="samurai_visibility",
                extra_attributes={
                    "samurai_filter_rejected": True,
                    "samurai_filter_rejected_reason": rejection_reason,
                },
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )
        if np.isfinite(mask_area) and mask_area > 0.0:
            last_valid_mask_area = float(mask_area)
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)

    commit_motion_filter_context(
        state,
        motion_context,
        extra_state={
            "last_valid_mask_area": last_valid_mask_area,
            "last_valid_bbox_aspect_ratio": last_valid_bbox_aspect_ratio,
            "last_valid_bbox_center_xy": last_valid_bbox_center_xy,
        },
    )
    return True


def track_spots_sam2_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline SAM 2 requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("sam2_repo_path")
    sam2_models_dir = Path(params.get("sam2_models_dir", DEFAULT_SAM2_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_sam2_checkpoints(
            sam2_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_sam2_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No SAM 2 checkpoint selected. Place a checkpoint such as `sam2.1_hiera_small.pt` "
            "in tracking_models/sam2 or the local repo/checkpoints and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )
    query_boxes = np.array(
        [
            [
                float(spot.roi_rect.left()),
                float(spot.roi_rect.top()),
                float(spot.roi_rect.left() + spot.roi_rect.width()),
                float(spot.roi_rect.top() + spot.roi_rect.height()),
            ]
            for spot in ref_file.spots
        ],
        dtype=np.float32,
    )
    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    sam2_settings = Sam2RuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        config_path=params.get("config_path"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        prompt_mode=params.get("prompt_mode", "point+bbox"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        mask_threshold=float(params.get("mask_threshold", 0.5)),
        min_mask_area=float(params.get("min_mask_area", 1.0)),
        max_mask_area=(
            None if params.get("max_mask_area") in {None, ""} else float(params.get("max_mask_area"))
        ),
        use_vos_optimized=bool(params.get("use_vos_optimized", False)),
        apply_postprocessing=bool(params.get("apply_postprocessing", True)),
        offload_video_to_cpu=bool(params.get("offload_video_to_cpu", True)),
        offload_state_to_cpu=bool(params.get("offload_state_to_cpu", False)),
        async_loading_frames=bool(params.get("async_loading_frames", False)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_mask_components = quality_filter_params["max_mask_components"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = family_context.roi_size
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_sam2_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=sam2_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = support._append_sam2_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_scores[query_idx],
                result.mask_component_counts[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                prompt_mode=result.prompt_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_mask_components=max_mask_components,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_box_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                seed_rect = seed_spot.roi_rect
                query_rows.append(
                    [float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]
                )
                query_box_rows.append(
                    [
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_sam2_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=sam2_settings,
                query_boxes_xyxy=np.asarray(query_box_rows, dtype=np.float32),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = support._append_sam2_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.mask_areas[query_idx],
                    result.mask_bboxes_xyxy[query_idx],
                    result.mask_scores[query_idx],
                    result.mask_component_counts[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    prompt_mode=result.prompt_mode,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_mask_area_ratio_change=max_mask_area_ratio_change,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_mask_components=max_mask_components,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def track_spots_dam4sam_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline DAM4SAM requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("dam4sam_repo_path")
    tracker_name = str(
        params.get("tracker_name")
        or params.get("dam4sam_tracker_name")
        or params.get("model_variant")
        or DEFAULT_DAM4SAM_TRACKER_NAME
    )
    checkpoint_path = _select_dam4sam_checkpoint_path(
        params,
        repo_path=repo_path,
        tracker_name=tracker_name,
    )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )
    query_boxes = np.array(
        [
            [
                float(spot.roi_rect.left()),
                float(spot.roi_rect.top()),
                float(spot.roi_rect.left() + spot.roi_rect.width()),
                float(spot.roi_rect.top() + spot.roi_rect.height()),
            ]
            for spot in ref_file.spots
        ],
        dtype=np.float32,
    )
    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    dam4sam_settings = _build_dam4sam_runtime_settings(
        params,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        tracker_name=tracker_name,
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_mask_components = quality_filter_params["max_mask_components"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    direction_filter = family_context.direction_filter
    direction_center_xy = direction_filter.direction_center_xy
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_dam4sam_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=dam4sam_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = _append_dam4sam_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_bboxes_xywh[query_idx],
                result.mask_scores[query_idx],
                result.masks[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                tracker_name=result.tracker_name,
                checkpoint_name=result.checkpoint_name,
                config_name=result.config_name,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_mask_components=max_mask_components,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_box_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                seed_rect = seed_spot.roi_rect
                query_rows.append(
                    [float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]
                )
                query_box_rows.append(
                    [
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_dam4sam_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=dam4sam_settings,
                query_boxes_xyxy=np.asarray(query_box_rows, dtype=np.float32),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_dam4sam_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.mask_areas[query_idx],
                    result.mask_bboxes_xyxy[query_idx],
                    result.mask_bboxes_xywh[query_idx],
                    result.mask_scores[query_idx],
                    result.masks[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    tracker_name=result.tracker_name,
                    checkpoint_name=result.checkpoint_name,
                    config_name=result.config_name,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_mask_area_ratio_change=max_mask_area_ratio_change,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_mask_components=max_mask_components,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def track_spots_samurai_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline SAMURAI requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("samurai_repo_path")
    model_variant = str(
        params.get("model_variant")
        or params.get("samurai_model_variant")
        or DEFAULT_SAMURAI_MODEL_VARIANT
    )
    checkpoint_path = _select_samurai_checkpoint_path(
        params,
        repo_path=repo_path,
        model_variant=model_variant,
    )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )
    query_boxes = np.array(
        [
            [
                float(spot.roi_rect.left()),
                float(spot.roi_rect.top()),
                float(spot.roi_rect.left() + spot.roi_rect.width()),
                float(spot.roi_rect.top() + spot.roi_rect.height()),
            ]
            for spot in ref_file.spots
        ],
        dtype=np.float32,
    )
    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    samurai_settings = _build_samurai_runtime_settings(
        params,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        model_variant=model_variant,
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_mask_components = quality_filter_params["max_mask_components"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    direction_filter = family_context.direction_filter
    direction_center_xy = direction_filter.direction_center_xy
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_samurai_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=samurai_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = _append_samurai_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_bboxes_xywh[query_idx],
                result.mask_scores[query_idx],
                result.masks[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                model_variant=result.model_variant,
                checkpoint_name=result.checkpoint_name,
                config_name=result.config_name,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_mask_components=max_mask_components,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_box_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                seed_rect = seed_spot.roi_rect
                query_rows.append(
                    [float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]
                )
                query_box_rows.append(
                    [
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_samurai_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=samurai_settings,
                query_boxes_xyxy=np.asarray(query_box_rows, dtype=np.float32),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_samurai_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.mask_areas[query_idx],
                    result.mask_bboxes_xyxy[query_idx],
                    result.mask_bboxes_xywh[query_idx],
                    result.mask_scores[query_idx],
                    result.masks[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    model_variant=result.model_variant,
                    checkpoint_name=result.checkpoint_name,
                    config_name=result.config_name,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_mask_area_ratio_change=max_mask_area_ratio_change,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_mask_components=max_mask_components,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def retrack_single_spot_dam4sam_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="DAM4SAM re-tracking",
    )
    start_spot = retrack_start.start_spot
    subset_frame_ids = retrack_start.subset_frame_ids
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    repo_path = params.get("repo_path") or params.get("dam4sam_repo_path")
    tracker_name = str(
        params.get("tracker_name")
        or params.get("dam4sam_tracker_name")
        or params.get("model_variant")
        or DEFAULT_DAM4SAM_TRACKER_NAME
    )
    checkpoint_path = _select_dam4sam_checkpoint_path(
        params,
        repo_path=repo_path,
        tracker_name=tracker_name,
    )

    start_rect = start_spot.roi_rect
    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )
    query_boxes = np.array(
        [[
            float(start_rect.left()),
            float(start_rect.top()),
            float(start_rect.left() + start_rect.width()),
            float(start_rect.top() + start_rect.height()),
        ]],
        dtype=np.float32,
    )

    dam4sam_settings = _build_dam4sam_runtime_settings(
        params,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        tracker_name=tracker_name,
    )
    start_file = all_files_map.get(start_frame_id)
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_mask_components = quality_filter_params["max_mask_components"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    direction_filter = family_context.direction_filter
    direction_center_xy = direction_filter.direction_center_xy
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap

    start_mask_area = getattr(start_spot, "dam4sam_mask_area", None)
    if not (
        start_mask_area is not None
        and np.isfinite(float(start_mask_area))
        and float(start_mask_area) > 0.0
    ):
        start_mask_area = float(start_rect.width() * start_rect.height())
    start_bbox_xyxy = getattr(start_spot, "dam4sam_mask_bbox_xyxy", None)
    start_bbox_center_xy, start_bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(
        start_bbox_xyxy
    )
    if start_bbox_aspect_ratio is None:
        start_bbox_aspect_ratio = (
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        )
    if start_bbox_center_xy is None:
        start_bbox_center_xy = np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        )

    state = _build_mask_tracking_state_from_values(
        start_spot,
        mask_area=float(start_mask_area),
        bbox_aspect_ratio=start_bbox_aspect_ratio,
        bbox_center_xy=np.asarray(start_bbox_center_xy, dtype=np.float32),
    )

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_dam4sam_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=dam4sam_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_dam4sam_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.mask_areas[0],
            result.mask_bboxes_xyxy[0],
            result.mask_bboxes_xywh[0],
            result.mask_scores[0],
            result.masks[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            tracker_name=result.tracker_name,
            checkpoint_name=result.checkpoint_name,
            config_name=result.config_name,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_mask_area_ratio_change=max_mask_area_ratio_change,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_mask_components=max_mask_components,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=1,
            min_global_write_idx=None,
            global_start_idx=0,
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            seed_local_idx = None
            seed_spot = None
            query_seed_ids = window.seed_frame_ids
            for local_idx, frame_id in enumerate(window_frame_ids):
                if frame_id not in query_seed_ids:
                    break
                candidate = track_to_update.spots_by_image_id.get(frame_id)
                if candidate is None:
                    continue
                seed_local_idx = local_idx
                seed_spot = candidate
                break

            if seed_spot is None or seed_local_idx is None:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            seed_rect = seed_spot.roi_rect
            result = run_dam4sam_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=dam4sam_settings,
                query_boxes_xyxy=np.asarray(
                    [[
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]],
                    dtype=np.float32,
                ),
            )

            ok = _append_dam4sam_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.mask_areas[0],
                result.mask_bboxes_xyxy[0],
                result.mask_bboxes_xywh[0],
                result.mask_scores[0],
                result.masks[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                tracker_name=result.tracker_name,
                checkpoint_name=result.checkpoint_name,
                config_name=result.config_name,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_mask_components=max_mask_components,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
            )
            if not ok:
                return None

            covered_until = advance_window_coverage(covered_until, end_idx)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def track_spots_deva_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline DEVA requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("deva_repo_path")
    deva_models_dir = Path(params.get("deva_models_dir", DEFAULT_DEVA_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_deva_checkpoints(
            deva_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_deva_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No DEVA checkpoint selected. Place `DEVA-propagation.pth` in tracking_models/deva "
            "or the local repo/saves and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )
    query_boxes = np.array(
        [
            [
                float(spot.roi_rect.left()),
                float(spot.roi_rect.top()),
                float(spot.roi_rect.left() + spot.roi_rect.width()),
                float(spot.roi_rect.top() + spot.roi_rect.height()),
            ]
            for spot in ref_file.spots
        ],
        dtype=np.float32,
    )
    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    deva_settings = DevaRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_size=params.get("inference_size", params.get("inference_resolution", 480)),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 0.0)),
        initial_point_radius_px=float(params.get("initial_point_radius_px", 4.0)),
        amp=bool(params.get("amp", True)),
        chunk_size=int(params.get("chunk_size", 4)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    direction_filter = family_context.direction_filter
    direction_center_xy = direction_filter.direction_center_xy
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_deva_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=deva_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = support._append_deva_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_scores[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_groups: Dict[int, List[Tuple[SpotROI, SpotROI]]] = {}
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                query_groups.setdefault(int(seed_local_idx), []).append(
                    (reference_spot, seed_spot)
                )

            if not query_groups:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            write_after_global_idx = covered_until
            for seed_local_idx, query_specs in sorted(query_groups.items()):
                if seed_local_idx >= len(window_frame_ids) - 1:
                    continue

                sliced_frames = window_frames[seed_local_idx:]
                sliced_frame_ids = window_frame_ids[seed_local_idx:]
                sliced_frame_shapes = window_shapes[seed_local_idx:]

                query_rows: List[List[float]] = []
                query_box_rows: List[List[float]] = []
                for _reference_spot, seed_spot in query_specs:
                    seed_rect = seed_spot.roi_rect
                    query_rows.append(
                        [0.0, float(seed_spot.center_y), float(seed_spot.center_x)]
                    )
                    query_box_rows.append(
                        [
                            float(seed_rect.left()),
                            float(seed_rect.top()),
                            float(seed_rect.left() + seed_rect.width()),
                            float(seed_rect.top() + seed_rect.height()),
                        ]
                    )

                result = run_deva_offline(
                    frames=sliced_frames,
                    query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                    settings=deva_settings,
                    query_boxes_xyxy=np.asarray(query_box_rows, dtype=np.float32),
                )

                for query_idx, (reference_spot, _seed_spot) in enumerate(query_specs):
                    if cancel_event and cancel_event.is_set():
                        return None

                    track = track_map[reference_spot.id]
                    ok = support._append_deva_sequence_to_track(
                        track,
                        track_states[reference_spot.id],
                        reference_spot,
                        sliced_frame_ids,
                        sliced_frame_shapes,
                        result.tracks_xy[query_idx],
                        result.visible_mask[query_idx],
                        result.mask_areas[query_idx],
                        result.mask_bboxes_xyxy[query_idx],
                        result.mask_scores[query_idx],
                        roi_size=roi_size,
                        keep_invisible_as_gaps=keep_invisible_as_gaps,
                        representative_point_mode=result.representative_point_mode,
                        initial_mask_mode=result.initial_mask_mode,
                        max_step_distance_px=max_step_distance_px,
                        max_reacquisition_gap=max_reacquisition_gap,
                        terminate_after_invisible_frames=terminate_after_invisible_frames,
                        max_mask_area_ratio_change=max_mask_area_ratio_change,
                        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                        max_bbox_center_jump_px=max_bbox_center_jump_px,
                        direction_center_xy=direction_center_xy,
                        direction_mode=direction_mode,
                        lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                        max_radial_jump_px=max_radial_jump_px,
                        min_directional_motion_px=min_directional_motion_px,
                        cancel_event=cancel_event,
                        start_local_idx=1,
                        min_global_write_idx=write_after_global_idx,
                        global_start_idx=start_idx + seed_local_idx,
                    )
                    if not ok:
                        return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def track_spots_cutie_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline Cutie requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("cutie_repo_path")
    cutie_models_dir = Path(params.get("cutie_models_dir", DEFAULT_CUTIE_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_cutie_checkpoints(
            cutie_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_cutie_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No Cutie checkpoint selected. Place a checkpoint such as `cutie-base-mega.pth` "
            "in tracking_models/cutie or the local repo/weights and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    mem_every_value = params.get("mem_every")
    cutie_settings = CutieRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "direct"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        initial_mask_mode=params.get("initial_mask_mode", "threshold"),
        initial_mask_threshold_rel=float(params.get("initial_mask_threshold_rel", 0.7)),
        initial_mask_disk_radius_ratio=float(params.get("initial_mask_disk_radius_ratio", 0.35)),
        mem_every=None if mem_every_value in {None, ""} else int(mem_every_value),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [support._spot_polygon_seed_mask(spot, frames[0].shape) for spot in ref_file.spots],
            dtype=bool,
        )

        if progress_callback:
            progress_callback(45)
        result = run_cutie_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=cutie_settings,
            query_initial_masks=query_initial_masks,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = support._append_cutie_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_scores[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_mask_rows: List[np.ndarray] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                query_rows.append(
                    [float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]
                )
                query_mask_rows.append(
                    support._spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_cutie_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=cutie_settings,
                query_initial_masks=np.asarray(query_mask_rows, dtype=bool),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = support._append_cutie_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.mask_areas[query_idx],
                    result.mask_bboxes_xyxy[query_idx],
                    result.mask_scores[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    initial_mask_mode=result.initial_mask_mode,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_mask_area_ratio_change=max_mask_area_ratio_change,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def track_spots_xmem2_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")
    missing_polygon_seed_ids = support._collect_missing_polygon_seed_spot_ids(ref_file.spots)
    if missing_polygon_seed_ids:
        raise ValueError(
            "Offline XMem2 requires a polygon seed for every reference spot. "
            f"Missing polygon for spot IDs: {', '.join(str(v) for v in missing_polygon_seed_ids)}."
        )

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline XMem2 requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("xmem2_repo_path")
    xmem2_models_dir = Path(params.get("xmem2_models_dir", DEFAULT_XMEM2_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_xmem2_checkpoints(
            xmem2_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_xmem2_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No XMem2 checkpoint selected. Place a checkpoint such as `XMem.pth` "
            "in tracking_models/xmem2 or the local repo/saves and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )
    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    mem_every_value = params.get("mem_every")
    xmem2_settings = XMem2RuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "direct"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        initial_mask_mode=params.get("initial_mask_mode", "threshold"),
        initial_mask_threshold_rel=float(params.get("initial_mask_threshold_rel", 0.7)),
        initial_mask_disk_radius_ratio=float(params.get("initial_mask_disk_radius_ratio", 0.35)),
        mem_every=None if mem_every_value in {None, ""} else int(mem_every_value),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [support._spot_polygon_seed_mask(spot, frames[0].shape) for spot in ref_file.spots],
            dtype=bool,
        )

        if progress_callback:
            progress_callback(45)
        result = run_xmem2_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=xmem2_settings,
            query_initial_masks=query_initial_masks,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            state = _build_reference_mask_tracking_state(reference_spot)
            ok = support._append_xmem2_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.mask_areas[query_idx],
                result.mask_bboxes_xyxy[query_idx],
                result.mask_scores[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
            )
            if not ok:
                return None
            converted_tracks.append(track)
    else:
        if progress_callback:
            progress_callback(10)

        track_map: Dict[int, Track] = {
            reference_spot.id: Track(
                track_id=reference_spot.id,
                spots_by_image_id={reference_frame_id: reference_spot},
            )
            for reference_spot in ref_file.spots
        }
        reference_spots_by_id = {spot.id: spot for spot in ref_file.spots}
        track_states: Dict[int, Dict[str, Any]] = {
            reference_spot.id: _build_reference_mask_tracking_state(reference_spot)
            for reference_spot in ref_file.spots
        }
        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_mask_rows: List[np.ndarray] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = window.seed_frame_ids

            for track_id, track in track_map.items():
                reference_spot = reference_spots_by_id[track_id]
                seed_local_idx = None
                seed_spot = None
                for local_idx, frame_id in enumerate(window_frame_ids):
                    if frame_id not in query_seed_ids:
                        break
                    candidate = track.spots_by_image_id.get(frame_id)
                    if candidate is None:
                        continue
                    seed_local_idx = local_idx
                    seed_spot = candidate
                    break
                if seed_spot is None or seed_local_idx is None:
                    continue

                query_rows.append(
                    [float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]
                )
                query_mask_rows.append(
                    support._spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_xmem2_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=xmem2_settings,
                query_initial_masks=np.asarray(query_mask_rows, dtype=bool),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = support._append_xmem2_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.mask_areas[query_idx],
                    result.mask_bboxes_xyxy[query_idx],
                    result.mask_scores[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    initial_mask_mode=result.initial_mask_mode,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_mask_area_ratio_change=max_mask_area_ratio_change,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = advance_window_coverage(covered_until, end_idx)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def retrack_single_spot_deva_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="DEVA re-tracking",
    )
    start_spot = retrack_start.start_spot
    subset_frame_ids = retrack_start.subset_frame_ids
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    repo_path = params.get("repo_path") or params.get("deva_repo_path")
    deva_models_dir = Path(params.get("deva_models_dir", DEFAULT_DEVA_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_deva_checkpoints(
            deva_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_deva_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No DEVA checkpoint selected. Place `DEVA-propagation.pth` in tracking_models/deva "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )
    start_rect = start_spot.roi_rect
    query_boxes = np.array(
        [[
            float(start_rect.left()),
            float(start_rect.top()),
            float(start_rect.left() + start_rect.width()),
            float(start_rect.top() + start_rect.height()),
        ]],
        dtype=np.float32,
    )

    deva_settings = DevaRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_size=params.get("inference_size", params.get("inference_resolution", 480)),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 0.0)),
        initial_point_radius_px=float(params.get("initial_point_radius_px", 4.0)),
        amp=bool(params.get("amp", True)),
        chunk_size=int(params.get("chunk_size", 4)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    start_file = all_files_map.get(start_frame_id)
    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    roi_size = family_context.roi_size
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    direction_filter = family_context.direction_filter
    direction_center_xy = direction_filter.direction_center_xy
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap

    start_mask_area = getattr(start_spot, "deva_mask_area", None)
    if not (
        start_mask_area is not None
        and np.isfinite(float(start_mask_area))
        and float(start_mask_area) > 0.0
    ):
        start_mask_area = float(start_rect.width() * start_rect.height())
    start_bbox_xyxy = getattr(start_spot, "deva_mask_bbox_xyxy", None)
    start_bbox_center_xy, start_bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(
        start_bbox_xyxy
    )
    if start_bbox_aspect_ratio is None:
        start_bbox_aspect_ratio = (
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        )
    if start_bbox_center_xy is None:
        start_bbox_center_xy = np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        )

    state = _build_mask_tracking_state_from_values(
        start_spot,
        mask_area=float(start_mask_area),
        bbox_aspect_ratio=start_bbox_aspect_ratio,
        bbox_center_xy=np.asarray(start_bbox_center_xy, dtype=np.float32),
    )

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_deva_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=deva_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = support._append_deva_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.mask_areas[0],
            result.mask_bboxes_xyxy[0],
            result.mask_scores[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            initial_mask_mode=result.initial_mask_mode,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_mask_area_ratio_change=max_mask_area_ratio_change,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=1,
            min_global_write_idx=None,
            global_start_idx=0,
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            seed_local_idx = None
            seed_spot = None
            query_seed_ids = window.seed_frame_ids
            for local_idx, frame_id in enumerate(window_frame_ids):
                if frame_id not in query_seed_ids:
                    break
                candidate = track_to_update.spots_by_image_id.get(frame_id)
                if candidate is None:
                    continue
                seed_local_idx = local_idx
                seed_spot = candidate
                break

            if seed_spot is None or seed_local_idx is None:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue
            if seed_local_idx >= len(window_frame_ids) - 1:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            seed_rect = seed_spot.roi_rect
            sliced_frames = window_frames[seed_local_idx:]
            sliced_frame_ids = window_frame_ids[seed_local_idx:]
            sliced_frame_shapes = window_shapes[seed_local_idx:]
            result = run_deva_offline(
                frames=sliced_frames,
                query_points_tyx=np.asarray(
                    [[0.0, float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=deva_settings,
                query_boxes_xyxy=np.asarray(
                    [[
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]],
                    dtype=np.float32,
                ),
            )

            ok = support._append_deva_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                sliced_frame_ids,
                sliced_frame_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.mask_areas[0],
                result.mask_bboxes_xyxy[0],
                result.mask_scores[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx + seed_local_idx,
            )
            if not ok:
                return None

            covered_until = advance_window_coverage(covered_until, end_idx)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def retrack_single_spot_sam2_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="SAM 2 re-tracking",
    )
    start_spot = retrack_start.start_spot
    subset_frame_ids = retrack_start.subset_frame_ids
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    sam2_models_dir = Path(params.get("sam2_models_dir", DEFAULT_SAM2_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_sam2_checkpoints(
            sam2_models_dir,
            repo_path=params.get("repo_path") or params.get("sam2_repo_path"),
        )
        selected_checkpoint = select_default_sam2_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No SAM 2 checkpoint selected. Place a checkpoint such as `sam2.1_hiera_small.pt` "
            "in tracking_models/sam2 or the local repo/checkpoints and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )
    start_rect = start_spot.roi_rect
    query_boxes = np.array(
        [[
            float(start_rect.left()),
            float(start_rect.top()),
            float(start_rect.left() + start_rect.width()),
            float(start_rect.top() + start_rect.height()),
        ]],
        dtype=np.float32,
    )

    sam2_settings = Sam2RuntimeSettings(
        repo_path=params.get("repo_path") or params.get("sam2_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        config_path=params.get("config_path"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        prompt_mode=params.get("prompt_mode", "point+bbox"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        mask_threshold=float(params.get("mask_threshold", 0.5)),
        min_mask_area=float(params.get("min_mask_area", 1.0)),
        max_mask_area=(
            None if params.get("max_mask_area") in {None, ""} else float(params.get("max_mask_area"))
        ),
        use_vos_optimized=bool(params.get("use_vos_optimized", False)),
        apply_postprocessing=bool(params.get("apply_postprocessing", True)),
        offload_video_to_cpu=bool(params.get("offload_video_to_cpu", True)),
        offload_state_to_cpu=bool(params.get("offload_state_to_cpu", False)),
        async_loading_frames=bool(params.get("async_loading_frames", False)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_mask_components = quality_filter_params["max_mask_components"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = family_context.roi_size
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    direction_center_xy = direction_filter.direction_center_xy

    start_mask_area = getattr(start_spot, "sam2_mask_area", None)
    if not (start_mask_area is not None and np.isfinite(float(start_mask_area)) and float(start_mask_area) > 0.0):
        start_mask_area = float(start_rect.width() * start_rect.height())
    start_bbox_xyxy = getattr(start_spot, "sam2_mask_bbox_xyxy", None)
    start_bbox_center_xy, start_bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(start_bbox_xyxy)
    if start_bbox_aspect_ratio is None:
        start_bbox_aspect_ratio = (
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        )
    if start_bbox_center_xy is None:
        start_bbox_center_xy = np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        )

    state = _build_mask_tracking_state_from_values(
        start_spot,
        mask_area=float(start_mask_area),
        bbox_aspect_ratio=start_bbox_aspect_ratio,
        bbox_center_xy=np.asarray(start_bbox_center_xy, dtype=np.float32),
    )

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded

        if progress_callback:
            progress_callback(45)
        result = run_sam2_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=sam2_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = support._append_sam2_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.mask_areas[0],
            result.mask_bboxes_xyxy[0],
            result.mask_scores[0],
            result.mask_component_counts[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            prompt_mode=result.prompt_mode,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_mask_area_ratio_change=max_mask_area_ratio_change,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_mask_components=max_mask_components,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=1,
            min_global_write_idx=None,
            global_start_idx=0,
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            seed_local_idx = None
            seed_spot = None
            query_seed_ids = window.seed_frame_ids
            for local_idx, frame_id in enumerate(window_frame_ids):
                if frame_id not in query_seed_ids:
                    break
                candidate = track_to_update.spots_by_image_id.get(frame_id)
                if candidate is None:
                    continue
                seed_local_idx = local_idx
                seed_spot = candidate
                break

            if seed_spot is None or seed_local_idx is None:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            seed_rect = seed_spot.roi_rect
            result = run_sam2_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=sam2_settings,
                query_boxes_xyxy=np.asarray(
                    [[
                        float(seed_rect.left()),
                        float(seed_rect.top()),
                        float(seed_rect.left() + seed_rect.width()),
                        float(seed_rect.top() + seed_rect.height()),
                    ]],
                    dtype=np.float32,
                ),
            )

            ok = support._append_sam2_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.mask_areas[0],
                result.mask_bboxes_xyxy[0],
                result.mask_scores[0],
                result.mask_component_counts[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                prompt_mode=result.prompt_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_mask_components=max_mask_components,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
            )
            if not ok:
                return None

            covered_until = advance_window_coverage(covered_until, end_idx)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def retrack_single_spot_cutie_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="Cutie re-tracking",
    )
    start_spot = retrack_start.start_spot
    subset_frame_ids = retrack_start.subset_frame_ids
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    cutie_models_dir = Path(params.get("cutie_models_dir", DEFAULT_CUTIE_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_cutie_checkpoints(
            cutie_models_dir,
            repo_path=params.get("repo_path") or params.get("cutie_repo_path"),
        )
        selected_checkpoint = select_default_cutie_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No Cutie checkpoint selected. Place a checkpoint such as `cutie-base-mega.pth` "
            "in tracking_models/cutie or the local repo/weights and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    mem_every_value = params.get("mem_every")
    cutie_settings = CutieRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("cutie_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "direct"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        initial_mask_mode=params.get("initial_mask_mode", "threshold"),
        initial_mask_threshold_rel=float(params.get("initial_mask_threshold_rel", 0.7)),
        initial_mask_disk_radius_ratio=float(params.get("initial_mask_disk_radius_ratio", 0.35)),
        mem_every=None if mem_every_value in {None, ""} else int(mem_every_value),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = family_context.roi_size
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    direction_center_xy = direction_filter.direction_center_xy

    start_rect = start_spot.roi_rect
    start_mask_area = getattr(start_spot, "cutie_mask_area", None)
    if not (start_mask_area is not None and np.isfinite(float(start_mask_area)) and float(start_mask_area) > 0.0):
        start_mask_area = float(start_rect.width() * start_rect.height())
    start_bbox_xyxy = getattr(start_spot, "cutie_mask_bbox_xyxy", None)
    start_bbox_center_xy, start_bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(start_bbox_xyxy)
    if start_bbox_aspect_ratio is None:
        start_bbox_aspect_ratio = (
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        )
    if start_bbox_center_xy is None:
        start_bbox_center_xy = np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        )

    state = _build_mask_tracking_state_from_values(
        start_spot,
        mask_area=float(start_mask_area),
        bbox_aspect_ratio=start_bbox_aspect_ratio,
        bbox_center_xy=np.asarray(start_bbox_center_xy, dtype=np.float32),
    )

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [support._spot_polygon_seed_mask(start_spot, frames[0].shape)],
            dtype=bool,
        )

        if progress_callback:
            progress_callback(45)
        result = run_cutie_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=cutie_settings,
            query_initial_masks=query_initial_masks,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = support._append_cutie_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.mask_areas[0],
            result.mask_bboxes_xyxy[0],
            result.mask_scores[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            initial_mask_mode=result.initial_mask_mode,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_mask_area_ratio_change=max_mask_area_ratio_change,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=1,
            min_global_write_idx=None,
            global_start_idx=0,
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            seed_local_idx = None
            seed_spot = None
            query_seed_ids = window.seed_frame_ids
            for local_idx, frame_id in enumerate(window_frame_ids):
                if frame_id not in query_seed_ids:
                    break
                candidate = track_to_update.spots_by_image_id.get(frame_id)
                if candidate is None:
                    continue
                seed_local_idx = local_idx
                seed_spot = candidate
                break

            if seed_spot is None or seed_local_idx is None:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_cutie_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=cutie_settings,
                query_initial_masks=np.asarray(
                    [support._spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)],
                    dtype=bool,
                ),
            )

            ok = support._append_cutie_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.mask_areas[0],
                result.mask_bboxes_xyxy[0],
                result.mask_scores[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
            )
            if not ok:
                return None

            covered_until = advance_window_coverage(covered_until, end_idx)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _collect_xmem2_permanent_reference_spots(
    track: Track,
    upto_frame_id: int,
    sorted_frame_ids: List[int],
) -> List[Tuple[int, SpotROI]]:
    references: List[Tuple[int, SpotROI]] = []
    seen_frame_ids: set[int] = set()

    for frame_id in sorted_frame_ids:
        if frame_id > upto_frame_id:
            break
        spot = track.spots_by_image_id.get(frame_id)
        if spot is None:
            continue

        is_first_available = not references
        is_current_frame = frame_id == upto_frame_id
        is_marked_reference = bool(getattr(spot, "xmem2_permanent_reference", False))
        if not (is_first_available or is_current_frame or is_marked_reference):
            continue
        if frame_id in seen_frame_ids:
            continue

        references.append((frame_id, spot))
        seen_frame_ids.add(frame_id)

    if not references and upto_frame_id in track.spots_by_image_id:
        references.append((upto_frame_id, track.spots_by_image_id[upto_frame_id]))
    return references


def retrack_single_spot_xmem2_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="XMem2 re-tracking",
    )
    start_spot = retrack_start.start_spot
    start_frame_index = retrack_start.start_frame_index
    if not polygon_points_are_valid(start_spot.polygon_seed_points):
        raise ValueError(
            f"XMem2 re-tracking requires a polygon seed on the start spot {start_spot.id}."
        )
    reference_entries = _collect_xmem2_permanent_reference_spots(
        track_to_update,
        start_frame_id,
        sorted_frame_ids,
    )
    missing_reference_polygon_ids = [
        int(reference_spot.id)
        for _, reference_spot in reference_entries
        if not polygon_points_are_valid(reference_spot.polygon_seed_points)
    ]
    if missing_reference_polygon_ids:
        raise ValueError(
            "XMem2 re-tracking requires polygon seeds for all active references. "
            f"Missing polygon for spot IDs: {', '.join(str(v) for v in missing_reference_polygon_ids)}."
        )
    earliest_reference_frame_id = reference_entries[0][0] if reference_entries else start_frame_id
    subset_start_index = sorted_frame_ids.index(earliest_reference_frame_id)
    subset_frame_ids = sorted_frame_ids[subset_start_index:]
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    xmem2_models_dir = Path(params.get("xmem2_models_dir", DEFAULT_XMEM2_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_xmem2_checkpoints(
            xmem2_models_dir,
            repo_path=params.get("repo_path") or params.get("xmem2_repo_path"),
        )
        selected_checkpoint = select_default_xmem2_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No XMem2 checkpoint selected. Place a checkpoint such as `XMem.pth` "
            "in tracking_models/xmem2 or the local repo/saves and pass checkpoint_path in params."
        )

    query_points = np.array(
        [
            [
                float(sorted_frame_ids.index(reference_frame_id) - subset_start_index),
                float(reference_spot.center_y),
                float(reference_spot.center_x),
            ]
            for reference_frame_id, reference_spot in reference_entries
        ],
        dtype=np.float32,
    )
    query_object_ids = np.ones((len(reference_entries),), dtype=np.int32)
    start_rect = start_spot.roi_rect

    mem_every_value = params.get("mem_every")
    xmem2_settings = XMem2RuntimeSettings(
        repo_path=params.get("repo_path") or params.get("xmem2_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "direct"),
        representative_point_mode=params.get("representative_point_mode", "mask_centroid"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 12.0)),
        initial_mask_mode=params.get("initial_mask_mode", "threshold"),
        initial_mask_threshold_rel=float(params.get("initial_mask_threshold_rel", 0.7)),
        initial_mask_disk_radius_ratio=float(params.get("initial_mask_disk_radius_ratio", 0.35)),
        mem_every=None if mem_every_value in {None, ""} else int(mem_every_value),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    family_context = build_mask_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    quality_filter_params = family_context.quality_filter_params
    max_mask_area_ratio_change = quality_filter_params["max_mask_area_ratio_change"]
    max_bbox_aspect_ratio_change = quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = quality_filter_params["max_bbox_center_jump_px"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = family_context.roi_size
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    force_full_sequence = len(reference_entries) > 1
    start_local_idx = int(start_frame_index - subset_start_index + 1)

    direction_center_xy = direction_filter.direction_center_xy

    start_mask_area = getattr(start_spot, "xmem2_mask_area", None)
    if not (start_mask_area is not None and np.isfinite(float(start_mask_area)) and float(start_mask_area) > 0.0):
        start_mask_area = float(start_rect.width() * start_rect.height())
    start_bbox_xyxy = getattr(start_spot, "xmem2_mask_bbox_xyxy", None)
    start_bbox_center_xy, start_bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(start_bbox_xyxy)
    if start_bbox_aspect_ratio is None:
        start_bbox_aspect_ratio = (
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        )
    if start_bbox_center_xy is None:
        start_bbox_center_xy = np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        )

    state = _build_mask_tracking_state_from_values(
        start_spot,
        mask_area=float(start_mask_area),
        bbox_aspect_ratio=start_bbox_aspect_ratio,
        bbox_center_xy=np.asarray(start_bbox_center_xy, dtype=np.float32),
    )

    if force_full_sequence or window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = support._load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [support._spot_polygon_seed_mask(reference_spot, frames[0].shape) for _, reference_spot in reference_entries],
            dtype=bool,
        )

        if progress_callback:
            progress_callback(45)
        result = run_xmem2_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=xmem2_settings,
            query_object_ids=query_object_ids,
            query_initial_masks=query_initial_masks,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = support._append_xmem2_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.mask_areas[0],
            result.mask_bboxes_xyxy[0],
            result.mask_scores[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            initial_mask_mode=result.initial_mask_mode,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_mask_area_ratio_change=max_mask_area_ratio_change,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=max(1, start_local_idx),
            min_global_write_idx=start_frame_index,
            global_start_idx=subset_start_index,
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_context = build_reseed_tracking_window_context(
            subset_frame_ids,
            window_size,
            window_overlap,
        )
        window_plans = window_context.plans
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = support._load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            seed_local_idx = None
            seed_spot = None
            query_seed_ids = window.seed_frame_ids
            for local_idx, frame_id in enumerate(window_frame_ids):
                if frame_id not in query_seed_ids:
                    break
                candidate = track_to_update.spots_by_image_id.get(frame_id)
                if candidate is None:
                    continue
                seed_local_idx = local_idx
                seed_spot = candidate
                break

            if seed_spot is None or seed_local_idx is None:
                covered_until = advance_window_coverage(covered_until, end_idx)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_plans))) * 75.0
                progress_callback(int(base))

            result = run_xmem2_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=xmem2_settings,
                query_initial_masks=np.asarray(
                    [support._spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)],
                    dtype=bool,
                ),
            )

            ok = support._append_xmem2_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.mask_areas[0],
                result.mask_bboxes_xyxy[0],
                result.mask_scores[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                initial_mask_mode=result.initial_mask_mode,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_mask_area_ratio_change=max_mask_area_ratio_change,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
            )
            if not ok:
                return None

            covered_until = advance_window_coverage(covered_until, end_idx)

    if progress_callback:
        progress_callback(100)
    return track_to_update
