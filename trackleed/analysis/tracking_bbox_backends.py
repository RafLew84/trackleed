from __future__ import annotations

from importlib import import_module
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trackleed.analysis.tomp_backend import (
    DEFAULT_TAMOS_MODELS_DIR,
    DEFAULT_TOMP_MODELS_DIR,
    TompRuntimeSettings,
    discover_tamos_checkpoints,
    discover_tomp_checkpoints,
    expected_tomp_checkpoint_name_for_variant,
    run_tomp_offline,
    select_default_tamos_checkpoint_for_variant,
    select_default_tomp_checkpoint_for_variant,
)
from trackleed.analysis.tracking_family_adapters import (
    build_bbox_family_adapter_context,
)
from trackleed.analysis.tracking_filter_utils import (
    build_bbox_mask_motion_tracking_state,
)
from trackleed.analysis.tracking_retrack_utils import (
    finish_retrack_if_too_short,
    prepare_retrack_start_context,
)
from trackleed.analysis.tracking_windowing_utils import (
    advance_window_coverage,
    build_reseed_tracking_window_context,
)
from trackleed.core.data_models import LeedFile, SpotROI, Track


def _tracking_support():
    return import_module("trackleed.analysis.tracking")


def _normalize_tomp_model_variant_key(model_variant: str | None) -> str:
    normalized = str(model_variant or "tomp50").strip().lower().replace("-", "_")
    if normalized in {"tomp", "tomp50", "tomp_50"}:
        return "tomp50"
    if normalized in {"tomp101", "tomp_101"}:
        return "tomp101"
    if normalized in {"tamos", "tamos50", "tamos_resnet50"}:
        return "tamos_resnet50"
    if normalized in {"tamos_swin", "tamos_swin_base", "tamos_swinbase"}:
        return "tamos_swin_base"
    raise ValueError(
        "Unsupported ToMP / TaMOs model variant. "
        "Expected `tomp50`/`tomp`, `tomp101`, `tamos_resnet50`/`tamos`, or `tamos_swin_base`."
    )


def _select_tomp_checkpoint_path(
    params: dict,
    model_variant_key: str,
    repo_path: str | None,
) -> str:
    checkpoint_path = params.get("checkpoint_path")
    if model_variant_key in {"tamos_resnet50", "tamos_swin_base"}:
        tamos_models_dir = Path(params.get("tamos_models_dir", DEFAULT_TAMOS_MODELS_DIR))
        if not checkpoint_path:
            detected_checkpoints = discover_tamos_checkpoints(
                tamos_models_dir,
                repo_path=repo_path,
            )
            selected_checkpoint = select_default_tamos_checkpoint_for_variant(
                detected_checkpoints,
                model_variant_key,
            )
            checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
        if not checkpoint_path:
            raise ValueError(
                "No TaMOs checkpoint selected. Place "
                f"`{expected_tomp_checkpoint_name_for_variant(model_variant_key)}` in "
                "tracking_models/tamos or pytracking/networks and pass checkpoint_path in params."
            )
        return str(checkpoint_path)

    tomp_models_dir = Path(params.get("tomp_models_dir", DEFAULT_TOMP_MODELS_DIR))
    if not checkpoint_path:
        detected_checkpoints = discover_tomp_checkpoints(
            tomp_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_tomp_checkpoint_for_variant(
            detected_checkpoints,
            model_variant_key,
        )
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No ToMP checkpoint selected. Place "
            f"`{expected_tomp_checkpoint_name_for_variant(model_variant_key)}` in "
            "tracking_models/tomp or pytracking/networks and pass checkpoint_path in params."
        )
    return str(checkpoint_path)


def _build_tomp_runtime_settings(
    params: dict,
    repo_path: str | None,
    checkpoint_path: str,
    model_variant_key: str,
) -> TompRuntimeSettings:
    bbox_quality_filter_params = read_bbox_quality_filter_params(params)
    return TompRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_variant=model_variant_key,
        device=params.get("device", "auto"),
        representative_point_mode=params.get("representative_point_mode", "bbox_center"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 0.0)),
        use_presence_score_gating=bbox_quality_filter_params["use_presence_score_gating"],
        presence_threshold=bbox_quality_filter_params["presence_threshold"],
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )


def track_spots_tomp_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline ToMP / TaMOs.

    Each reference spot ROI becomes one seeded bbox. The returned bbox centers
    are converted to Track / SpotROI objects while preserving bbox metadata.
    """
    support = _tracking_support()

    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline ToMP / TaMOs requires at least two frames starting from the reference.")

    model_variant_key = _normalize_tomp_model_variant_key(params.get("model_variant", "tomp50"))
    repo_path = (
        params.get("repo_path")
        or params.get("tomp_repo_path")
        or params.get("pytracking_repo_path")
    )
    checkpoint_path = _select_tomp_checkpoint_path(params, model_variant_key, repo_path)

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

    tomp_settings = _build_tomp_runtime_settings(
        params,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        model_variant_key=model_variant_key,
    )
    family_context = build_bbox_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    roi_size = family_context.roi_size
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    bbox_quality_filter_params = family_context.quality_filter_params
    max_bbox_aspect_ratio_change = bbox_quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = bbox_quality_filter_params["max_bbox_center_jump_px"]
    use_presence_score_gating = bbox_quality_filter_params["use_presence_score_gating"]
    presence_threshold = bbox_quality_filter_params["presence_threshold"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
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
        result = run_tomp_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tomp_settings,
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
            rect = reference_spot.roi_rect
            state = build_bbox_mask_motion_tracking_state(
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
            ok = support._append_tomp_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.tracked_bboxes_xywh[query_idx],
                result.tracked_bboxes_xyxy[query_idx],
                result.bbox_scores[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                model_variant=result.model_variant,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                use_presence_score_gating=use_presence_score_gating,
                presence_threshold=presence_threshold,
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
            reference_spot.id: build_bbox_mask_motion_tracking_state(
                (float(reference_spot.center_x), float(reference_spot.center_y)),
                last_valid_mask_area=float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                last_valid_bbox_aspect_ratio=(
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                last_valid_bbox_center_xy=(
                    float(reference_spot.center_x),
                    float(reference_spot.center_y),
                ),
            )
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

            result = run_tomp_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=tomp_settings,
                query_boxes_xyxy=np.asarray(query_box_rows, dtype=np.float32),
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = support._append_tomp_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.tracked_bboxes_xywh[query_idx],
                    result.tracked_bboxes_xyxy[query_idx],
                    result.bbox_scores[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    representative_point_mode=result.representative_point_mode,
                    model_variant=result.model_variant,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                    max_bbox_center_jump_px=max_bbox_center_jump_px,
                    use_presence_score_gating=use_presence_score_gating,
                    presence_threshold=presence_threshold,
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


def retrack_single_spot_tomp_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using ToMP / TaMOs.

    The track is trimmed after `start_frame_id`, then the selected bbox tracker
    is rerun from that point using the current ROI rectangle as the seed bbox.
    """
    support = _tracking_support()

    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="ToMP / TaMOs re-tracking",
    )
    start_spot = retrack_start.start_spot
    start_frame_index = retrack_start.start_frame_index
    subset_frame_ids = retrack_start.subset_frame_ids
    early_return = finish_retrack_if_too_short(
        track_to_update,
        subset_frame_ids=subset_frame_ids,
        progress_callback=progress_callback,
    )
    if early_return is not None:
        return early_return

    start_variant = (
        params.get("model_variant")
        or getattr(start_spot, "tomp_model_variant", None)
        or (
            "tamos_resnet50"
            if str(getattr(start_spot, "source", "")).strip().lower() == "tamos_offline"
            else "tomp50"
        )
    )
    model_variant_key = _normalize_tomp_model_variant_key(start_variant)
    repo_path = (
        params.get("repo_path")
        or params.get("tomp_repo_path")
        or params.get("pytracking_repo_path")
    )
    checkpoint_path = _select_tomp_checkpoint_path(params, model_variant_key, repo_path)

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

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    tomp_settings = _build_tomp_runtime_settings(
        params,
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        model_variant_key=model_variant_key,
    )
    family_context = build_bbox_family_adapter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
        estimate_pattern_center_fn=_tracking_support().estimate_pattern_center_from_detections,
    )
    roi_size = family_context.roi_size
    keep_invisible_as_gaps = family_context.keep_invisible_as_gaps
    window_size = family_context.window_size
    window_overlap = family_context.window_overlap
    motion_filter_params = family_context.motion_filter_params
    max_step_distance_px = motion_filter_params["max_step_distance_px"]
    max_reacquisition_gap = motion_filter_params["max_reacquisition_gap"]
    terminate_after_invisible_frames = motion_filter_params["terminate_after_invisible_frames"]
    bbox_quality_filter_params = family_context.quality_filter_params
    max_bbox_aspect_ratio_change = bbox_quality_filter_params["max_bbox_aspect_ratio_change"]
    max_bbox_center_jump_px = bbox_quality_filter_params["max_bbox_center_jump_px"]
    use_presence_score_gating = bbox_quality_filter_params["use_presence_score_gating"]
    presence_threshold = bbox_quality_filter_params["presence_threshold"]
    support = _tracking_support()
    direction_filter = family_context.direction_filter
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    start_rect = start_spot.roi_rect
    state = build_bbox_mask_motion_tracking_state(
        (float(start_spot.center_x), float(start_spot.center_y)),
        last_valid_mask_area=float(start_rect.width() * start_rect.height()),
        last_valid_bbox_aspect_ratio=(
            float(start_rect.width()) / float(start_rect.height())
            if start_rect.height() > 0
            else None
        ),
        last_valid_bbox_center_xy=(
            float(start_spot.center_x),
            float(start_spot.center_y),
        ),
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
        result = run_tomp_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tomp_settings,
            query_boxes_xyxy=query_boxes,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = support._append_tomp_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.tracked_bboxes_xywh[0],
            result.tracked_bboxes_xyxy[0],
            result.bbox_scores[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            representative_point_mode=result.representative_point_mode,
            model_variant=result.model_variant,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
            max_bbox_center_jump_px=max_bbox_center_jump_px,
            use_presence_score_gating=use_presence_score_gating,
            presence_threshold=presence_threshold,
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
            result = run_tomp_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=tomp_settings,
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

            ok = support._append_tomp_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.tracked_bboxes_xywh[0],
                result.tracked_bboxes_xyxy[0],
                result.bbox_scores[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                representative_point_mode=result.representative_point_mode,
                model_variant=result.model_variant,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
                max_bbox_center_jump_px=max_bbox_center_jump_px,
                use_presence_score_gating=use_presence_score_gating,
                presence_threshold=presence_threshold,
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
