from __future__ import annotations

# stal/analysis/tracking.py
# Compatibility shim:
# public Trackleed tracking entrypoints are resolved lazily from family modules,
# while the legacy runtime remains below during the staged refactor.

from importlib import import_module
from pathlib import Path
from typing import Any, List, Dict, Callable, Tuple, Optional
from threading import Event

import cv2
import numpy as np
from PyQt6.QtCore import QRect
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2gray

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.peak_analyzer import find_pattern_center, fit_spot_to_gaussian
from trackleed.analysis.tracking_config import default_directional_hybrid_params
from trackleed.analysis.tapir_backend import (
    DEFAULT_TAPIR_MODELS_DIR,
    TapirOnlineRuntimeSettings,
    TapirRuntimeSettings,
    discover_online_tapir_checkpoints,
    discover_tapir_checkpoints,
    run_tapir_offline,
    run_tapir_online,
    select_default_online_tapir_checkpoint,
    select_default_tapir_checkpoint,
)
from trackleed.analysis.locotrack_backend import (
    DEFAULT_LOCOTRACK_MODELS_DIR,
    LocotrackRuntimeSettings,
    discover_locotrack_checkpoints,
    run_locotrack_offline,
    select_default_locotrack_checkpoint,
)
from trackleed.analysis.cotracker_backend import (
    DEFAULT_COTRACKER_MODELS_DIR,
    CotrackerRuntimeSettings,
    discover_cotracker_checkpoints,
    run_cotracker3_offline,
    select_default_cotracker_checkpoint,
)
from trackleed.analysis.tapnext_backend import (
    DEFAULT_TAPNEXT_MODELS_DIR,
    TapnextRuntimeSettings,
    discover_tapnext_checkpoints,
    run_tapnext_offline,
    select_default_tapnext_checkpoint,
)
from trackleed.analysis.trackonr_backend import (
    DEFAULT_TRACKONR_MODELS_DIR,
    TrackonrRuntimeSettings,
    discover_trackonr_checkpoints,
    run_trackonr_offline,
    select_default_trackonr_checkpoint,
)
from trackleed.analysis.pipspp_backend import (
    DEFAULT_PIPSPP_MODELS_DIR,
    PipsppRuntimeSettings,
    discover_pipspp_checkpoints,
    run_pipspp_offline,
    select_default_pipspp_checkpoint,
)
from trackleed.analysis.mftiq_backend import (
    DEFAULT_MFTIQ_MODELS_DIR,
    MftiqRuntimeSettings,
    discover_mftiq_checkpoints,
    run_mftiq_offline,
    select_default_mftiq_checkpoint,
)
from trackleed.analysis.sam2_backend import (
    DEFAULT_SAM2_MODELS_DIR,
    Sam2RuntimeSettings,
    discover_sam2_checkpoints,
    run_sam2_offline,
    select_default_sam2_checkpoint,
)
from trackleed.analysis.cutie_backend import (
    DEFAULT_CUTIE_MODELS_DIR,
    CutieRuntimeSettings,
    discover_cutie_checkpoints,
    run_cutie_offline,
    select_default_cutie_checkpoint,
)
from trackleed.analysis.xmem2_backend import (
    DEFAULT_XMEM2_MODELS_DIR,
    XMem2RuntimeSettings,
    discover_xmem2_checkpoints,
    run_xmem2_offline,
    select_default_xmem2_checkpoint,
)
from trackleed.analysis.polygon_mask import (
    normalize_polygon_points,
    polygon_points_are_valid,
    polygon_points_to_mask,
)
from trackleed.analysis.tracking_result_builders import (
    apply_bbox_xywh_metadata,
    apply_bbox_xyxy_metadata,
    apply_prefixed_point_metadata,
    bbox_center_and_aspect_ratio_from_xyxy as _bbox_center_and_aspect_ratio_from_xyxy,
    build_spot_roi,
    build_track,
    build_track_with_initial_spot,
    set_spot_attributes,
)
from trackleed.analysis.tracking_retrack_utils import (
    finish_retrack_if_too_short,
    prepare_retrack_start_context,
)
from trackleed.analysis.tracking_filter_utils import (
    append_motion_accepted_spot,
    build_bbox_quality_filter_context,
    build_direction_filter_context,
    build_mask_quality_filter_context,
    build_motion_filter_context,
    commit_motion_filter_context,
    gate_bbox_quality,
    gate_by_direction,
    gate_by_max_step_distance,
    gate_mask_quality,
    register_motion_rejection,
    should_terminate_before_visible_candidate,
)
from trackleed.analysis.tracking_visibility_utils import (
    append_invisible_track_frame,
)
from trackleed.analysis.tracking_windowing_utils import (
    advance_window_coverage,
    build_tracking_window_plans,
    build_plain_tracking_window_context,
    build_reseed_tracking_window_context,
)
from trackleed.analysis.seeded_tracking_runtime import (
    ncc_around as _ncc_around,
    normalized_grayscale,
    prepare_optical_flow_image,
    refine_center_by_centroid,
)

_PUBLIC_TRACKING_SHIM_EXPORTS = {
    "track_spots_tapir_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_tapir_offline",
    ),
    "track_spots_tapir_online": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_tapir_online",
    ),
    "track_spots_locotrack_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_locotrack_offline",
    ),
    "track_spots_cotracker3_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_cotracker3_offline",
    ),
    "track_spots_tapnext_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_tapnext_offline",
    ),
    "track_spots_trackonr_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_trackonr_offline",
    ),
    "track_spots_pipspp_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_pipspp_offline",
    ),
    "track_spots_mftiq_offline": (
        "trackleed.analysis.tracking_point_backends",
        "track_spots_mftiq_offline",
    ),
    "track_spots_tomp_offline": (
        "trackleed.analysis.tracking_bbox_backends",
        "track_spots_tomp_offline",
    ),
    "track_spots_ostrack_offline": (
        "trackleed.analysis.tracking_bbox_backends",
        "track_spots_ostrack_offline",
    ),
    "track_spots_cutie_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_cutie_offline",
    ),
    "track_spots_xmem2_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_xmem2_offline",
    ),
    "track_spots_sam2_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_sam2_offline",
    ),
    "track_spots_deva_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_deva_offline",
    ),
    "track_spots_dam4sam_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_dam4sam_offline",
    ),
    "track_spots_samurai_offline": (
        "trackleed.analysis.tracking_mask_backends",
        "track_spots_samurai_offline",
    ),
}

_PUBLIC_RETRACK_SHIM_EXPORTS = {
    "retrack_single_spot_tapir_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_tapir_offline",
    ),
    "retrack_single_spot_tapir_online": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_tapir_online",
    ),
    "retrack_single_spot_locotrack_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_locotrack_offline",
    ),
    "retrack_single_spot_cotracker3_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_cotracker3_offline",
    ),
    "retrack_single_spot_tapnext_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_tapnext_offline",
    ),
    "retrack_single_spot_trackonr_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_trackonr_offline",
    ),
    "retrack_single_spot_pipspp_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_pipspp_offline",
    ),
    "retrack_single_spot_mftiq_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_mftiq_offline",
    ),
    "retrack_single_spot_tomp_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_tomp_offline",
    ),
    "retrack_single_spot_ostrack_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_ostrack_offline",
    ),
    "retrack_single_spot_cutie_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_cutie_offline",
    ),
    "retrack_single_spot_xmem2_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_xmem2_offline",
    ),
    "retrack_single_spot_sam2_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_sam2_offline",
    ),
    "retrack_single_spot_deva_offline": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_deva_offline",
    ),
    "retrack_single_spot_optical_flow": (
        "trackleed.analysis.tracking_retrack",
        "retrack_single_spot_optical_flow",
    ),
}

_PUBLIC_SHIM_EXPORTS = {
    **_PUBLIC_TRACKING_SHIM_EXPORTS,
    **_PUBLIC_RETRACK_SHIM_EXPORTS,
}

__all__ = [
    "track_spots_optical_flow",
    "track_spots_optical_flow_skip_anchor",
    "track_spots_hungarian",
    "track_spots_hungarian_directional",
    *_PUBLIC_SHIM_EXPORTS.keys(),
]


def __getattr__(name: str):
    target = _PUBLIC_SHIM_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def _load_and_prep_image(filepath, background_image: Optional[np.ndarray] = None, subtract_background: bool = False):
    """
    Loads a TIFF image, normalizes it, and returns an 8-bit grayscale array suitable for LK tracking.
    """
    image_raw = load_tiff(filepath)
    try:
        return prepare_optical_flow_image(
            image_raw,
            background_image if subtract_background else None,
        )
    except ValueError:
        return None


def _load_background_image(filepath: Optional[str | Path]) -> Optional[np.ndarray]:
    if not filepath:
        return None
    path = Path(filepath)
    bg_raw = load_tiff(path)
    if bg_raw is None or bg_raw.size == 0:
        return None
    return normalized_grayscale(bg_raw)


def _centered_roi_rect(
    center_x: float,
    center_y: float,
    roi_size: int,
    image_shape: Tuple[int, int],
) -> QRect:
    """Build a square QRect centered on a point and clamp it to image bounds."""
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    size = max(1, min(int(roi_size), max(1, image_w), max(1, image_h)))
    left = float(center_x) - size / 2.0
    top = float(center_y) - size / 2.0
    max_left = float(max(0, image_w - size))
    max_top = float(max(0, image_h - size))
    left = min(max(0.0, left), max_left)
    top = min(max(0.0, top), max_top)
    return QRect(int(round(left)), int(round(top)), int(size), int(size))


def _roi_rect_from_bbox_xyxy(
    bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    image_shape: Tuple[int, int],
    *,
    fallback_center_x: float,
    fallback_center_y: float,
    fallback_roi_size: int,
) -> QRect:
    """Build a QRect from an xyxy bbox and clamp it to image bounds."""
    if bbox_xyxy is None:
        return _centered_roi_rect(
            fallback_center_x,
            fallback_center_y,
            fallback_roi_size,
            image_shape,
        )

    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return _centered_roi_rect(
            fallback_center_x,
            fallback_center_y,
            fallback_roi_size,
            image_shape,
        )

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    x0 = float(np.clip(bbox[0], 0.0, max(0.0, float(image_w - 1))))
    y0 = float(np.clip(bbox[1], 0.0, max(0.0, float(image_h - 1))))
    x1 = float(np.clip(bbox[2], 0.0, float(image_w)))
    y1 = float(np.clip(bbox[3], 0.0, float(image_h)))
    if x1 <= x0 or y1 <= y0:
        return _centered_roi_rect(
            fallback_center_x,
            fallback_center_y,
            fallback_roi_size,
            image_shape,
        )

    width = max(1, int(round(x1 - x0)))
    height = max(1, int(round(y1 - y0)))
    left = min(max(0, int(round(x0))), max(0, image_w - width))
    top = min(max(0, int(round(y0))), max(0, image_h - height))
    return QRect(left, top, width, height)


def _spot_polygon_seed_mask(
    spot: SpotROI,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    if not polygon_points_are_valid(spot.polygon_seed_points, image_shape=image_shape):
        raise ValueError(
            f"Spot {spot.id} requires a valid polygon seed for mask-based tracking."
        )
    return polygon_points_to_mask(image_shape, spot.polygon_seed_points)


def _translate_polygon_seed_points(
    polygon_points: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None,
    *,
    delta_x: float,
    delta_y: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[tuple[float, float]]:
    if not polygon_points:
        return []
    shifted = [
        (float(point[0]) + float(delta_x), float(point[1]) + float(delta_y))
        for point in polygon_points
    ]
    return normalize_polygon_points(shifted, image_shape=image_shape)


def _collect_missing_polygon_seed_spot_ids(
    spots: List[SpotROI],
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> list[int]:
    missing: list[int] = []
    for spot in spots:
        if not polygon_points_are_valid(spot.polygon_seed_points, image_shape=image_shape):
            missing.append(int(spot.id))
    return missing


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


def _tomp_tracker_source_name(model_variant: str | None) -> str:
    if _normalize_tomp_model_variant_key(model_variant) in {"tamos_resnet50", "tamos_swin_base"}:
        return "tamos_offline"
    return "tomp_offline"


def _frame_to_normalized_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an arbitrary loaded frame to normalized 2D float grayscale."""
    if frame.ndim == 3:
        gray = rgb2gray(frame).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        if max_val > min_val:
            gray = (gray - min_val) / (max_val - min_val)
        else:
            gray = np.zeros_like(gray, dtype=np.float32)
    return gray


def _build_recentered_point_tracker_spot(
    *,
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    source: str,
    metadata_prefix: str,
    extra_attributes: dict[str, Any] | None = None,
) -> SpotROI:
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)
    if visible and np.isfinite(x) and np.isfinite(y):
        x, y = _refine_tapir_point(
            image_frame,
            x,
            y,
            roi_size,
            recenter_mode,
            centroid_patch_multiplier,
            centroid_rel_threshold,
            centroid_max_expand,
            centroid_max_shift_ratio,
        )
    rect = _centered_roi_rect(x, y, roi_size, image_shape)
    spot = build_spot_roi(
        spot_id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        source=source,
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix=metadata_prefix,
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    if extra_attributes:
        set_spot_attributes(spot, extra_attributes)
    return spot


def _refine_tapir_point_by_centroid(
    frame: np.ndarray,
    approx_x: float,
    approx_y: float,
    roi_size: int,
    patch_multiplier: float,
    rel_threshold: float,
    max_expand: int,
    max_shift_ratio: float,
) -> tuple[float, float]:
    """
    Recenters a TAPIR point on the local bright spot inside a bounded patch.

    The refinement is intentionally local so we do not jump to a different spot
    when the raw TAPIR point is still reasonably close to the intended target.
    """
    gray = _frame_to_normalized_gray(frame)
    approx_xy = np.array([approx_x, approx_y], dtype=np.float32)
    patch_radius = max(1.0, (float(roi_size) / 2.0) * float(patch_multiplier))
    refined_xy = refine_center_by_centroid(
        gray,
        approx_xy,
        patch_radius,
        float(rel_threshold),
        int(max(0, max_expand)),
    )
    max_shift = max(1.0, float(roi_size) * float(max_shift_ratio))
    delta = refined_xy - approx_xy
    shift_norm = float(np.linalg.norm(delta))
    if shift_norm > max_shift and shift_norm > 1e-6:
        refined_xy = approx_xy + (delta / shift_norm) * max_shift
    return float(refined_xy[0]), float(refined_xy[1])


def _refine_tapir_point(
    frame: np.ndarray,
    approx_x: float,
    approx_y: float,
    roi_size: int,
    recenter_mode: str,
    patch_multiplier: float,
    rel_threshold: float,
    max_expand: int,
    max_shift_ratio: float,
) -> tuple[float, float]:
    """
    Recenter a TAPIR point using one of several local refinement strategies.

    Supported modes:
    - none/off/raw
    - peak
    - centroid
    - gaussian
    """
    mode = str(recenter_mode or "none").strip().lower()
    if mode in {"", "none", "off", "raw"}:
        return float(approx_x), float(approx_y)

    gray = _frame_to_normalized_gray(frame)
    height, width = gray.shape[:2]
    approx_xy = np.array([approx_x, approx_y], dtype=np.float32)
    patch_radius = max(1.0, (float(roi_size) / 2.0) * float(patch_multiplier))
    r = max(1, int(round(patch_radius)))
    cx = int(round(float(approx_x)))
    cy = int(round(float(approx_y)))
    x0 = max(0, cx - r)
    x1 = min(width, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(height, cy + r + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return float(approx_x), float(approx_y)

    refined_xy = approx_xy.copy()

    if mode == "centroid":
        return _refine_tapir_point_by_centroid(
            frame,
            approx_x,
            approx_y,
            roi_size,
            patch_multiplier,
            rel_threshold,
            max_expand,
            max_shift_ratio,
        )

    if mode == "peak":
        smoothed = gaussian_filter(patch.astype(np.float64, copy=False), sigma=1.0)
        peak_index = int(np.nanargmax(smoothed))
        peak_y, peak_x = np.unravel_index(peak_index, smoothed.shape)
        refined_xy = np.array([x0 + peak_x, y0 + peak_y], dtype=np.float32)
    elif mode == "gaussian":
        fit = fit_spot_to_gaussian(
            patch,
            center_preference_factor=0.35,
        )
        if fit is None:
            return float(approx_x), float(approx_y)
        fit_x = float(fit.get("center_x", (patch.shape[1] - 1) / 2.0))
        fit_y = float(fit.get("center_y", (patch.shape[0] - 1) / 2.0))
        if not np.isfinite(fit_x) or not np.isfinite(fit_y):
            return float(approx_x), float(approx_y)
        refined_xy = np.array([x0 + fit_x, y0 + fit_y], dtype=np.float32)
    else:
        return float(approx_x), float(approx_y)

    max_shift = max(1.0, float(roi_size) * float(max_shift_ratio))
    delta = refined_xy - approx_xy
    shift_norm = float(np.linalg.norm(delta))
    if shift_norm > max_shift and shift_norm > 1e-6:
        refined_xy = approx_xy + (delta / shift_norm) * max_shift
    return float(refined_xy[0]), float(refined_xy[1])


def _load_frames_for_tapir(
    all_files_map: Dict[int, LeedFile],
    frame_ids: List[int],
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """
    Load raw image frames for TAPIR inference.

    Returns:
        tuple (frames, per_frame_shapes) where frames is a stacked ndarray
        and per_frame_shapes is a list of (height, width).
    """
    frames: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    total = len(frame_ids)
    for idx, frame_id in enumerate(frame_ids):
        if cancel_event and cancel_event.is_set():
            return None
        leed_file = all_files_map.get(frame_id)
        if not leed_file:
            raise ValueError(f"Missing LeedFile for frame {frame_id}.")
        image = load_tiff(leed_file.filepath)
        if image is None or image.size == 0:
            raise ValueError(f"Could not load image for frame {frame_id}: {leed_file.filepath}")
        frames.append(np.asarray(image))
        shapes.append((int(image.shape[0]), int(image.shape[1])))
        if progress_callback:
            progress_callback(int(((idx + 1) / max(1, total)) * 35))
    return np.stack(frames, axis=0), shapes


def _build_tapir_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    source: str = "tapir_offline",
) -> SpotROI:
    """Create a SpotROI from TAPIR output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source=source,
        metadata_prefix="tapir",
    )


def _build_locotrack_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
) -> SpotROI:
    """Create a SpotROI from LocoTrack output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="locotrack_offline",
        metadata_prefix="locotrack",
    )


def _build_cotracker3_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
) -> SpotROI:
    """Create a SpotROI from CoTracker3 output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="cotracker3_offline",
        metadata_prefix="cotracker3",
    )


def _build_tapnext_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    visibility_score: float | None = None,
    certainty_score: float | None = None,
    combined_score: float | None = None,
) -> SpotROI:
    """Create a SpotROI from TAPNext output, optionally recentered on the local spot."""
    extra_attributes: dict[str, Any] = {}
    if visibility_score is not None:
        extra_attributes["tapnext_visibility_score"] = float(visibility_score)
    if certainty_score is not None:
        extra_attributes["tapnext_certainty_score"] = float(certainty_score)
    if combined_score is not None:
        extra_attributes["tapnext_combined_score"] = float(combined_score)
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="tapnext_offline",
        metadata_prefix="tapnext",
        extra_attributes=extra_attributes,
    )


def _build_trackonr_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
) -> SpotROI:
    """Create a SpotROI from Track-On-R output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="trackonr_offline",
        metadata_prefix="trackonr",
    )


def _build_pipspp_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
) -> SpotROI:
    """Create a SpotROI from PIPs++ output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="pipspp_offline",
        metadata_prefix="pipspp",
    )


def _build_mftiq_spot(
    reference_spot: SpotROI,
    image_frame: np.ndarray,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    occlusion_score: float,
    sigma_score: float,
) -> SpotROI:
    """Create a SpotROI from MFTIQ output, optionally recentered on the local spot."""
    return _build_recentered_point_tracker_spot(
        reference_spot=reference_spot,
        image_frame=image_frame,
        image_shape=image_shape,
        raw_x=raw_x,
        raw_y=raw_y,
        visible=visible,
        roi_size=roi_size,
        recenter_mode=recenter_mode,
        centroid_patch_multiplier=centroid_patch_multiplier,
        centroid_rel_threshold=centroid_rel_threshold,
        centroid_max_expand=centroid_max_expand,
        centroid_max_shift_ratio=centroid_max_shift_ratio,
        source="mftiq_offline",
        metadata_prefix="mftiq",
        extra_attributes={
            "mftiq_occlusion_score": float(occlusion_score),
            "mftiq_sigma_score": float(sigma_score),
        },
    )


def _build_sam2_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    mask_component_count: int,
    representative_point_mode: str,
    prompt_mode: str,
) -> SpotROI:
    """Create a SpotROI from SAM 2 output using the representative point and mask bbox."""
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _roi_rect_from_bbox_xyxy(
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
        source="sam2_offline",
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix="sam2",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "sam2_mask_area": float(mask_area),
            "sam2_mask_score": float(mask_score),
            "sam2_mask_component_count": int(mask_component_count),
            "sam2_repr_mode": str(representative_point_mode),
            "sam2_prompt_mode": str(prompt_mode),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="sam2_mask_bbox_xyxy",
        aspect_ratio_attribute_name="sam2_mask_bbox_aspect_ratio",
    )
    return spot


def _build_cutie_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    representative_point_mode: str,
    initial_mask_mode: str,
) -> SpotROI:
    """Create a SpotROI from Cutie output using the representative point and mask bbox."""
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _roi_rect_from_bbox_xyxy(
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
        source="cutie_offline",
        visible=visible,
    )
    if reference_spot.has_polygon_seed:
        delta_x = float(x - float(reference_spot.center_x))
        delta_y = float(y - float(reference_spot.center_y))
        spot.polygon_seed_points = _translate_polygon_seed_points(
            reference_spot.polygon_seed_points,
            delta_x=delta_x,
            delta_y=delta_y,
            image_shape=image_shape,
        )
    apply_prefixed_point_metadata(
        spot,
        prefix="cutie",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "cutie_mask_area": float(mask_area),
            "cutie_mask_score": float(mask_score),
            "cutie_repr_mode": str(representative_point_mode),
            "cutie_initial_mask_mode": str(initial_mask_mode),
            "polygon_seed_used": bool(reference_spot.has_polygon_seed),
            "polygon_vertex_count": int(len(spot.polygon_seed_points)),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="cutie_mask_bbox_xyxy",
        aspect_ratio_attribute_name="cutie_mask_bbox_aspect_ratio",
    )
    return spot


def _build_xmem2_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    representative_point_mode: str,
    initial_mask_mode: str,
) -> SpotROI:
    """Create a SpotROI from XMem2 output using the representative point and mask bbox."""
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _roi_rect_from_bbox_xyxy(
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
        source="xmem2_offline",
        visible=visible,
    )
    if reference_spot.has_polygon_seed:
        delta_x = float(x - float(reference_spot.center_x))
        delta_y = float(y - float(reference_spot.center_y))
        spot.polygon_seed_points = _translate_polygon_seed_points(
            reference_spot.polygon_seed_points,
            delta_x=delta_x,
            delta_y=delta_y,
            image_shape=image_shape,
        )
    apply_prefixed_point_metadata(
        spot,
        prefix="xmem2",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "xmem2_mask_area": float(mask_area),
            "xmem2_mask_score": float(mask_score),
            "xmem2_repr_mode": str(representative_point_mode),
            "xmem2_initial_mask_mode": str(initial_mask_mode),
            "polygon_seed_used": bool(reference_spot.has_polygon_seed),
            "polygon_vertex_count": int(len(spot.polygon_seed_points)),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="xmem2_mask_bbox_xyxy",
        aspect_ratio_attribute_name="xmem2_mask_bbox_aspect_ratio",
    )
    return spot


def _build_deva_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    mask_area: float,
    mask_bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    mask_score: float,
    representative_point_mode: str,
    initial_mask_mode: str,
) -> SpotROI:
    """Create a SpotROI from DEVA output using the representative point and mask bbox."""
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _roi_rect_from_bbox_xyxy(
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
        source="deva_offline",
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix="deva",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "deva_mask_area": float(mask_area),
            "deva_mask_score": float(mask_score),
            "deva_repr_mode": str(representative_point_mode),
            "deva_initial_mask_mode": str(initial_mask_mode),
        },
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=mask_bbox_xyxy,
        bbox_attribute_name="deva_mask_bbox_xyxy",
        aspect_ratio_attribute_name="deva_mask_bbox_aspect_ratio",
    )
    return spot


def _build_tomp_spot(
    reference_spot: SpotROI,
    image_shape: Tuple[int, int],
    raw_x: float,
    raw_y: float,
    visible: bool,
    roi_size: int,
    bbox_xywh: np.ndarray | tuple[float, float, float, float] | None,
    bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
    bbox_score: float,
    representative_point_mode: str,
    model_variant: str,
) -> SpotROI:
    """Create a SpotROI from ToMP / TaMOs bbox output."""
    if np.isfinite(raw_x) and np.isfinite(raw_y):
        x = float(raw_x)
        y = float(raw_y)
    else:
        x = float(reference_spot.center_x)
        y = float(reference_spot.center_y)

    rect = _roi_rect_from_bbox_xyxy(
        bbox_xyxy if visible else None,
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
        source=_tomp_tracker_source_name(model_variant),
        visible=visible,
    )
    apply_prefixed_point_metadata(
        spot,
        prefix="tomp",
        raw_x=raw_x,
        raw_y=raw_y,
        refined_x=x,
        refined_y=y,
        visible=visible,
    )
    set_spot_attributes(
        spot,
        {
            "tomp_bbox_score": float(bbox_score) if np.isfinite(bbox_score) else float("nan"),
            "tomp_model_variant": str(model_variant),
            "tomp_repr_mode": str(representative_point_mode),
        },
    )
    apply_bbox_xywh_metadata(
        spot,
        bbox_xywh=bbox_xywh,
        bbox_attribute_name="tomp_bbox_xywh",
    )
    apply_bbox_xyxy_metadata(
        spot,
        bbox_xyxy=bbox_xyxy,
        bbox_attribute_name="tomp_bbox_xyxy",
        aspect_ratio_attribute_name="tomp_bbox_aspect_ratio",
    )
    return spot


def _compute_point_tracker_window_starts(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> List[int]:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_plain_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return [window.start_idx for window in window_context.plans]


def _compute_tapir_window_starts(total_frames: int, window_size: int, window_overlap: int) -> List[int]:
    """Backward-compatible alias for the generic point-tracker window helper."""
    return _compute_point_tracker_window_starts(total_frames, window_size, window_overlap)


def _resolve_tapnext_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_trackonr_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_pipspp_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_mftiq_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_sam2_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_cutie_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_xmem2_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_tomp_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _resolve_tapir_online_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    frame_ids = list(range(max(0, int(total_frames))))
    window_context = build_reseed_tracking_window_context(
        frame_ids,
        window_size,
        window_overlap,
    )
    return window_context.overlap


def _infer_locotrack_model_size(
    checkpoint_path: str | Path | None,
    fallback: str = "small",
) -> str:
    if checkpoint_path is None:
        return fallback
    name = Path(checkpoint_path).name.lower()
    if "base" in name:
        return "base"
    if "small" in name:
        return "small"
    return fallback


def _append_tapir_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    direction_center_xy: Tuple[float, float] | None,
    direction_mode: str,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
    cancel_event: Optional[Event] = None,
    start_local_idx: int = 1,
    min_global_write_idx: int | None = None,
    global_start_idx: int = 0,
    source: str = "tapir_offline",
) -> bool:
    """
    Append TAPIR predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_tapir_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
                source=source,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="tapir",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="tapir",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                forced_spot = candidate_spot or _build_tapir_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    raw_x,
                    raw_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                    source=source,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="tapir_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_locotrack_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
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
    """
    Append LocoTrack predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_locotrack_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="locotrack",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="locotrack",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                forced_spot = candidate_spot or _build_locotrack_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    raw_x,
                    raw_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="locotrack_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_cotracker3_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
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
    """
    Append CoTracker3 predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_cotracker3_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="cotracker3",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="cotracker3",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                forced_spot = candidate_spot or _build_cotracker3_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    raw_x,
                    raw_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="cotracker3_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_tapnext_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    direction_center_xy: Tuple[float, float] | None,
    direction_mode: str,
    lateral_angle_tolerance_deg: float,
    max_radial_jump_px: float,
    min_directional_motion_px: float,
    query_visibility_scores: np.ndarray | None = None,
    query_certainty_scores: np.ndarray | None = None,
    query_combined_scores: np.ndarray | None = None,
    cancel_event: Optional[Event] = None,
    start_local_idx: int = 1,
    min_global_write_idx: int | None = None,
    global_start_idx: int = 0,
) -> bool:
    """
    Append TAPNext predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        visibility_score = (
            float(query_visibility_scores[local_idx]) if query_visibility_scores is not None else None
        )
        certainty_score = (
            float(query_certainty_scores[local_idx]) if query_certainty_scores is not None else None
        )
        combined_score = (
            float(query_combined_scores[local_idx]) if query_combined_scores is not None else None
        )

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_tapnext_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
                visibility_score=visibility_score,
                certainty_score=certainty_score,
                combined_score=combined_score,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="tapnext",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="tapnext",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                forced_spot = candidate_spot or _build_tapnext_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    raw_x,
                    raw_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                    visibility_score=visibility_score,
                    certainty_score=certainty_score,
                    combined_score=combined_score,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="tapnext_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_trackonr_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
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
    """
    Append Track-On-R predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_trackonr_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="trackonr",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="trackonr",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                forced_spot = candidate_spot or _build_trackonr_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    raw_x,
                    raw_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="trackonr_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_pipspp_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
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
    """
    Append PIPs++ predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_pipspp_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="pipspp",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="pipspp",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                fallback_x = raw_x if np.isfinite(raw_x) else (
                    float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
                )
                fallback_y = raw_y if np.isfinite(raw_y) else (
                    float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
                )
                forced_spot = candidate_spot or _build_pipspp_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    fallback_x,
                    fallback_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="pipspp_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_mftiq_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frames: np.ndarray,
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_occlusion_scores: np.ndarray,
    query_sigma_scores: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    recenter_mode: str,
    centroid_patch_multiplier: float,
    centroid_rel_threshold: float,
    centroid_max_expand: int,
    centroid_max_shift_ratio: float,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
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
    """
    Append MFTIQ predictions to one track, applying filtering and gap logic.

    Returns False only when canceled.
    """
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

    if motion_context.terminated:
        return True

    for local_idx in range(start_local_idx, len(frame_ids)):
        if cancel_event and cancel_event.is_set():
            return False

        global_idx = global_start_idx + local_idx
        if min_global_write_idx is not None and global_idx <= min_global_write_idx:
            continue

        frame_id = frame_ids[local_idx]
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        occlusion_score = float(query_occlusion_scores[local_idx])
        sigma_score = float(query_sigma_scores[local_idx])

        candidate_spot: SpotROI | None = None
        accepted = visible

        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            candidate_spot = _build_mftiq_spot(
                reference_spot,
                frames[local_idx],
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                recenter_mode,
                centroid_patch_multiplier,
                centroid_rel_threshold,
                centroid_max_expand,
                centroid_max_shift_ratio,
                occlusion_score=occlusion_score,
                sigma_score=sigma_score,
            )

            accepted, _ = gate_by_max_step_distance(
                candidate_spot,
                last_valid_xy=motion_context.last_valid_xy,
                max_step_distance_px=motion_context.max_step_distance_px,
                prefix="mftiq",
            )
            if accepted:
                accepted, _ = gate_by_direction(
                    candidate_spot,
                    state,
                    last_valid_xy=motion_context.last_valid_xy,
                    direction_center_xy=direction_center_xy,
                    direction_mode=motion_context.direction_mode,
                    lateral_angle_tolerance_deg=motion_context.lateral_angle_tolerance_deg,
                    max_radial_jump_px=motion_context.max_radial_jump_px,
                    min_directional_motion_px=motion_context.min_directional_motion_px,
                    prefix="mftiq",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                fallback_x = raw_x if np.isfinite(raw_x) else (
                    float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
                )
                fallback_y = raw_y if np.isfinite(raw_y) else (
                    float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
                )
                forced_spot = candidate_spot or _build_mftiq_spot(
                    reference_spot,
                    frames[local_idx],
                    frame_shapes[local_idx],
                    fallback_x,
                    fallback_y,
                    False,
                    roi_size,
                    "none",
                    centroid_patch_multiplier,
                    centroid_rel_threshold,
                    centroid_max_expand,
                    centroid_max_shift_ratio,
                    occlusion_score=occlusion_score,
                    sigma_score=sigma_score,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="mftiq_visibility",
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )

    commit_motion_filter_context(state, motion_context)
    return True


def _append_sam2_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_scores: np.ndarray,
    query_mask_component_counts: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    prompt_mode: str,
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
    """
    Append SAM 2 representative points and mask summaries to one track.

    Returns False only when canceled.
    """
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
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        mask_area = float(query_mask_areas[local_idx])
        mask_score = float(query_mask_scores[local_idx])
        mask_bbox_xyxy = query_mask_bboxes_xyxy[local_idx]
        mask_component_count = int(query_mask_component_counts[local_idx])
        image_h, image_w = frame_shapes[local_idx]
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
            candidate_spot = _build_sam2_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                mask_component_count,
                representative_point_mode,
                prompt_mode,
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
                prefix="sam2",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="sam2",
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
                    prefix="sam2",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                fallback_x = raw_x if np.isfinite(raw_x) else (
                    float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
                )
                fallback_y = raw_y if np.isfinite(raw_y) else (
                    float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
                )
                forced_spot = candidate_spot or _build_sam2_spot(
                    reference_spot,
                    frame_shapes[local_idx],
                    fallback_x,
                    fallback_y,
                    False,
                    roi_size,
                    mask_area,
                    mask_bbox_xyxy,
                    mask_score,
                    mask_component_count,
                    representative_point_mode,
                    prompt_mode,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="sam2_visibility",
                extra_attributes={
                    "sam2_filter_rejected": True,
                    "sam2_filter_rejected_reason": rejection_reason,
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


def _append_cutie_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_scores: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    initial_mask_mode: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
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
    """Append Cutie representative points and mask summaries to one track."""
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
        max_mask_components=0,
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
        visible = bool(query_visible[local_idx])
        raw_x = float(query_tracks[local_idx, 0])
        raw_y = float(query_tracks[local_idx, 1])
        mask_area = float(query_mask_areas[local_idx])
        mask_bbox_xyxy = query_mask_bboxes_xyxy[local_idx]
        mask_score = float(query_mask_scores[local_idx])
        image_h, image_w = frame_shapes[local_idx]
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
            candidate_spot = _build_cutie_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                representative_point_mode,
                initial_mask_mode,
            )

            accepted, rejection_reason = gate_mask_quality(
                candidate_spot,
                context=quality_context,
                current_mask_area=mask_area,
                last_valid_mask_area=last_valid_mask_area,
                mask_component_count=0,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="cutie",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="cutie",
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
                    prefix="cutie",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            forced_spot = None
            if not keep_invisible_as_gaps:
                fallback_x = raw_x if np.isfinite(raw_x) else (
                    float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
                )
                fallback_y = raw_y if np.isfinite(raw_y) else (
                    float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
                )
                forced_spot = candidate_spot or _build_cutie_spot(
                    reference_spot,
                    frame_shapes[local_idx],
                    fallback_x,
                    fallback_y,
                    False,
                    roi_size,
                    mask_area,
                    mask_bbox_xyxy,
                    mask_score,
                    representative_point_mode,
                    initial_mask_mode,
                )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="cutie_visibility",
                extra_attributes={
                    "cutie_filter_rejected": True,
                    "cutie_filter_rejected_reason": rejection_reason,
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


def _append_xmem2_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_scores: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    initial_mask_mode: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
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
    """Append XMem2 representative points and mask summaries to one track."""
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
        max_mask_components=0,
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
        mask_score = float(query_mask_scores[local_idx])
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
            candidate_spot = _build_xmem2_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                representative_point_mode,
                initial_mask_mode,
            )

            accepted, rejection_reason = gate_mask_quality(
                candidate_spot,
                context=quality_context,
                current_mask_area=mask_area,
                last_valid_mask_area=last_valid_mask_area,
                mask_component_count=0,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="xmem2",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="xmem2",
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
                    prefix="xmem2",
                )

        if keep_invisible_as_gaps:
            if not accepted:
                if register_motion_rejection(state, motion_context):
                    return True
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
            forced_spot = candidate_spot or _build_xmem2_spot(
                reference_spot,
                frame_shapes[local_idx],
                fallback_x,
                fallback_y,
                False,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                representative_point_mode,
                initial_mask_mode,
            )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="xmem2_visibility",
                extra_attributes={
                    "xmem2_filter_rejected": True,
                    "xmem2_filter_rejected_reason": rejection_reason,
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


def _append_deva_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_mask_areas: np.ndarray,
    query_mask_bboxes_xyxy: np.ndarray,
    query_mask_scores: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    initial_mask_mode: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_mask_area_ratio_change: float,
    max_bbox_aspect_ratio_change: float,
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
    """Append DEVA representative points and mask summaries to one track."""
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
        max_mask_components=0,
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
        mask_score = float(query_mask_scores[local_idx])
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
            candidate_spot = _build_deva_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                representative_point_mode,
                initial_mask_mode,
            )

            accepted, rejection_reason = gate_mask_quality(
                candidate_spot,
                context=quality_context,
                current_mask_area=mask_area,
                last_valid_mask_area=last_valid_mask_area,
                mask_component_count=0,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="deva",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="deva",
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
                    prefix="deva",
                )

        if keep_invisible_as_gaps:
            if not accepted:
                if register_motion_rejection(state, motion_context):
                    return True
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
            forced_spot = candidate_spot or _build_deva_spot(
                reference_spot,
                frame_shapes[local_idx],
                fallback_x,
                fallback_y,
                False,
                roi_size,
                mask_area,
                mask_bbox_xyxy,
                mask_score,
                representative_point_mode,
                initial_mask_mode,
            )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="deva_visibility",
                extra_attributes={
                    "deva_filter_rejected": True,
                    "deva_filter_rejected_reason": rejection_reason,
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


def _append_tomp_sequence_to_track(
    track: Track,
    state: Dict[str, Any],
    reference_spot: SpotROI,
    frame_ids: List[int],
    frame_shapes: List[Tuple[int, int]],
    query_tracks: np.ndarray,
    query_visible: np.ndarray,
    query_bboxes_xywh: np.ndarray,
    query_bboxes_xyxy: np.ndarray,
    query_bbox_scores: np.ndarray,
    *,
    roi_size: int,
    keep_invisible_as_gaps: bool,
    representative_point_mode: str,
    model_variant: str,
    max_step_distance_px: float,
    max_reacquisition_gap: int,
    terminate_after_invisible_frames: int,
    max_bbox_aspect_ratio_change: float,
    max_bbox_center_jump_px: float,
    use_presence_score_gating: bool,
    presence_threshold: float,
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
    """Append ToMP / TaMOs bbox summaries to one track."""
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
    last_valid_bbox_aspect_ratio = state.get("last_valid_bbox_aspect_ratio")
    last_valid_bbox_center_xy = state.get("last_valid_bbox_center_xy")
    quality_context = build_bbox_quality_filter_context(
        max_bbox_aspect_ratio_change=max_bbox_aspect_ratio_change,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
        use_presence_score_gating=use_presence_score_gating,
        presence_threshold=presence_threshold,
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
        bbox_xywh = query_bboxes_xywh[local_idx]
        bbox_xyxy = query_bboxes_xyxy[local_idx]
        bbox_score = float(query_bbox_scores[local_idx])
        bbox_center_xy, bbox_aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(bbox_xyxy)

        candidate_spot: SpotROI | None = None
        rejection_reason = "not_visible"
        accepted = visible
        if accepted:
            if should_terminate_before_visible_candidate(state, motion_context):
                return True

            if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                accepted = False
                rejection_reason = "invalid_bbox"
            elif not (0.0 <= raw_x < float(image_w) and 0.0 <= raw_y < float(image_h)):
                accepted = False
                rejection_reason = "out_of_bounds"

        if accepted:
            candidate_spot = _build_tomp_spot(
                reference_spot,
                frame_shapes[local_idx],
                raw_x,
                raw_y,
                True,
                roi_size,
                bbox_xywh,
                bbox_xyxy,
                bbox_score,
                representative_point_mode,
                model_variant,
            )

            accepted, rejection_reason = gate_bbox_quality(
                candidate_spot,
                context=quality_context,
                score=bbox_score,
                bbox_aspect_ratio=bbox_aspect_ratio,
                last_valid_bbox_aspect_ratio=last_valid_bbox_aspect_ratio,
                bbox_center_xy=bbox_center_xy,
                last_valid_bbox_center_xy=last_valid_bbox_center_xy,
                prefix="tomp",
            )
            if accepted:
                accepted, rejection_reason = gate_by_max_step_distance(
                    candidate_spot,
                    last_valid_xy=motion_context.last_valid_xy,
                    max_step_distance_px=motion_context.max_step_distance_px,
                    prefix="tomp",
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
                    prefix="tomp",
                )

        if not accepted:
            if register_motion_rejection(state, motion_context):
                return True
            if keep_invisible_as_gaps:
                continue
            fallback_x = raw_x if np.isfinite(raw_x) else (
                float(last_valid_xy[0]) if last_valid_xy is not None else float(reference_spot.center_x)
            )
            fallback_y = raw_y if np.isfinite(raw_y) else (
                float(last_valid_xy[1]) if last_valid_xy is not None else float(reference_spot.center_y)
            )
            forced_spot = candidate_spot or _build_tomp_spot(
                reference_spot,
                frame_shapes[local_idx],
                fallback_x,
                fallback_y,
                False,
                roi_size,
                bbox_xywh,
                bbox_xyxy,
                bbox_score,
                representative_point_mode,
                model_variant,
            )
            append_invisible_track_frame(
                track,
                frame_id=frame_id,
                forced_spot=forced_spot,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                visibility_attribute_name="tomp_visibility",
                extra_attributes={
                    "tomp_filter_rejected": True,
                    "tomp_filter_rejected_reason": rejection_reason,
                },
            )
            continue

        last_valid_xy = append_motion_accepted_spot(
            track,
            frame_id=frame_id,
            spot=candidate_spot,
            context=motion_context,
        )
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)

    commit_motion_filter_context(
        state,
        motion_context,
        extra_state={
            "last_valid_bbox_aspect_ratio": last_valid_bbox_aspect_ratio,
            "last_valid_bbox_center_xy": last_valid_bbox_center_xy,
        },
    )
    return True


# --- Directional geometry helpers -------------------------------------------------------------

def estimate_pattern_center_from_detections(
    file_map: Dict[int, LeedFile],
    frame_ids: Optional[List[int]] = None,
    min_spots_per_frame: int = 3,
    max_spots_per_frame: int = 200,
    max_frames: int = 10,
    fallback_shape: Optional[Tuple[int, int]] = None,
) -> Optional[Dict[str, float]]:
    """
    Aggregate detections across frames and estimate the diffraction pattern center
    using the existing RANSAC-based finder. Falls back to stored pattern_center
    or image center when estimation fails.
    """
    frame_ids = frame_ids or sorted(file_map.keys())
    frame_ids = frame_ids[:max_frames] if max_frames > 0 else frame_ids
    centers: List[Dict[str, float]] = []
    for fid in frame_ids:
        lf = file_map.get(fid)
        if not lf or not lf.spots or len(lf.spots) < min_spots_per_frame:
            continue
        # Limit per-frame sample to keep RANSAC fast
        spots = lf.spots
        if len(spots) > max_spots_per_frame:
            import random

            spots = random.sample(spots, max_spots_per_frame)
        spot_positions = [{"center_x": s.center_x, "center_y": s.center_y} for s in spots]
        center = find_pattern_center(spot_positions) if spot_positions else None
        if center:
            centers.append(center)

    if centers:
        cx = float(np.median([c["center_x"] for c in centers]))
        cy = float(np.median([c["center_y"] for c in centers]))
        ex = float(np.median([c.get("error_x", 0.0) for c in centers]))
        ey = float(np.median([c.get("error_y", 0.0) for c in centers]))
        return {"center_x": cx, "center_y": cy, "error_x": ex, "error_y": ey}

    # Fallback to any precomputed pattern_center stored on files
    for fid in frame_ids:
        lf = file_map.get(fid)
        if lf and lf.pattern_center:
            return lf.pattern_center

    # Last resort: geometric image center if shape is known
    if fallback_shape:
        h, w = fallback_shape
        return {"center_x": w / 2.0, "center_y": h / 2.0, "error_x": 0.0, "error_y": 0.0}
    return None


def radial_unit_and_radius(center_xy: Tuple[float, float], point_xy: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """
    Compute the radial unit vector and radius from center to point.
    Returns (unit_vector, radius). Unit vector is zero when radius is ~0.
    """
    vec = np.array(point_xy, dtype=np.float32) - np.array(center_xy, dtype=np.float32)
    radius = float(np.linalg.norm(vec))
    if radius > 1e-6:
        return vec / radius, radius
    return np.zeros(2, dtype=np.float32), 0.0


def angular_offset_deg(radial_unit: np.ndarray, displacement: np.ndarray) -> float:
    """
    Angle in degrees between the expected radial direction and an observed displacement.
    Returns 0 if either vector is near zero.
    """
    disp = np.array(displacement, dtype=np.float32)
    ru = np.array(radial_unit, dtype=np.float32)
    disp_norm = float(np.linalg.norm(disp))
    ru_norm = float(np.linalg.norm(ru))
    if disp_norm < 1e-9 or ru_norm < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(ru, disp) / (ru_norm * disp_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def ring_bin_index(radius: float, bin_size_px: float) -> int:
    """Assign a radius to a discrete ring/bin index."""
    if bin_size_px <= 0:
        return 0
    return int(np.floor(max(0.0, radius) / float(bin_size_px)))


def median_step_magnitude(points: List[Tuple[float, float]]) -> float:
    """Median magnitude of successive steps in a polyline of points."""
    if len(points) < 2:
        return 0.0
    arr = np.array(points, dtype=np.float32)
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def track_spots_hungarian(
    file_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> List[Track]:
    """
    Link per-frame detections (SpotROI) into tracks using Hungarian assignment.

    params: {
        "max_distance_px": float,
        "max_gap": int,           # allowed missing frames
        "min_track_length": int,  # discard shorter tracks
    }
    """
    max_dist = float(params.get("max_distance_px", 20.0))
    max_gap = int(params.get("max_gap", 2))
    min_len = int(params.get("min_track_length", 2))

    next_track_id = 1
    active_tracks: Dict[int, dict] = {}  # track_id -> {"last_frame": fid, "last_xy": (x,y), "gap": 0, "spots": {fid: spot}}
    finished: List[Track] = []

    total_frames = len(sorted_frame_ids)
    for idx, fid in enumerate(sorted_frame_ids):
        if cancel_event and cancel_event.is_set():
            return []
        if progress_callback:
            progress_callback(int(100 * idx / max(1, total_frames)))

        lf = file_map.get(fid)
        if not lf:
            continue
        detections = lf.spots or []
        det_points = np.array([[s.center_x, s.center_y] for s in detections], dtype=np.float32)

        # Build assignment between active tracks and current detections
        track_ids = list(active_tracks.keys())
        if len(track_ids) and len(det_points):
            cost = np.zeros((len(track_ids), len(det_points)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                last_xy = np.array(active_tracks[tid]["last_xy"])
                dists = np.linalg.norm(det_points - last_xy[None, :], axis=1)
                cost[i] = dists
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_tracks = set()
            assigned_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= max_dist:
                    tid = track_ids[r]
                    spot = detections[c]
                    active_tracks[tid]["last_xy"] = (spot.center_x, spot.center_y)
                    active_tracks[tid]["last_frame"] = fid
                    active_tracks[tid]["gap"] = 0
                    active_tracks[tid]["spots"][fid] = spot
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)
            # Increase gap for unassigned tracks
            for tid in track_ids:
                if tid in assigned_tracks:
                    continue
                active_tracks[tid]["gap"] += 1
        else:
            assigned_dets = set()

        # Start new tracks for unmatched detections
        for j, spot in enumerate(detections):
            if j in assigned_dets:
                continue
            tid = next_track_id
            next_track_id += 1
            active_tracks[tid] = {
                "last_frame": fid,
                "last_xy": (spot.center_x, spot.center_y),
                "gap": 0,
                "spots": {fid: spot},
            }

        # Retire tracks that exceeded gap
        to_remove = []
        for tid, data in active_tracks.items():
            if data["gap"] > max_gap:
                if len(data["spots"]) >= min_len:
                    finished.append(build_track(track_id=tid, spots_by_image_id=data["spots"]))
                to_remove.append(tid)
        for tid in to_remove:
            active_tracks.pop(tid, None)

    # Flush remaining active tracks
    for tid, data in active_tracks.items():
        if len(data["spots"]) >= min_len:
            finished.append(build_track(track_id=tid, spots_by_image_id=data["spots"]))

    return finished


def track_spots_hungarian_directional(
    file_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> List[Track]:
    """
    Directionally gated Hungarian linker that prefers radial motion and penalizes lateral/ring jumps.

    params merges defaults from default_directional_hybrid_params; key overrides:
        center_mode: "auto" or "manual"
        manual_center: (cx, cy) when center_mode == "manual"
        direction_mode: "inward" | "outward" | "auto"
        lateral_angle_tolerance_deg: float
        ring_bin_px: float
        base_max_distance_px: float
        dynamic_max_distance: bool
        dynamic_margin_px: float
        use_ring_penalty: bool
        max_gap: int            # legacy gap counter
        max_misses: int         # tolerated consecutive misses
        min_track_length: int
    """
    cfg = default_directional_hybrid_params(params or {})

    print(
        "[DirectionalLink] start | frames=%d | center_mode=%s | dir=%s | dynamic_max=%s"
        % (
            len(sorted_frame_ids),
            cfg.get("center_mode", "auto"),
            cfg.get("direction_mode", "auto"),
            cfg.get("dynamic_max_distance", True),
        )
    )

    base_max_dist = float(cfg.get("base_max_distance_px", 20.0))
    dynamic_max = bool(cfg.get("dynamic_max_distance", True))
    dynamic_margin = float(cfg.get("dynamic_margin_px", 5.0))
    max_gap = int(cfg.get("max_gap", 2))
    max_misses = int(cfg.get("max_misses", 3))
    min_len = int(cfg.get("min_track_length", 2))
    lateral_tol = float(cfg.get("lateral_angle_tolerance_deg", 20.0))
    ring_bin_px = float(cfg.get("ring_bin_px", 25.0))
    use_ring_penalty = bool(cfg.get("use_ring_penalty", True))
    direction_mode = str(cfg.get("direction_mode", "auto")).lower()

    # Determine center
    fallback_shape = None
    for fid in sorted_frame_ids:
        lf = file_map.get(fid)
        if lf and lf.filepath:
            img = load_tiff(lf.filepath)
            if img is not None and img.size:
                h, w = img.shape[:2]
                fallback_shape = (h, w)
                break

    if str(cfg.get("center_mode", "auto")).lower() == "manual" and cfg.get("manual_center"):
        cx, cy = cfg["manual_center"]
        center = {"center_x": float(cx), "center_y": float(cy), "error_x": 0.0, "error_y": 0.0}
    else:
        center = estimate_pattern_center_from_detections(file_map, sorted_frame_ids, fallback_shape=fallback_shape)
        if not center and fallback_shape:
            h, w = fallback_shape
            center = {"center_x": w / 2.0, "center_y": h / 2.0, "error_x": 0.0, "error_y": 0.0}
    if not center:
        raise ValueError("Could not determine pattern center for directional linking.")
    center_xy = (float(center["center_x"]), float(center["center_y"]))
    print(
        "[DirectionalLink] center set to (%.1f, %.1f) mode=%s"
        % (center_xy[0], center_xy[1], cfg.get("center_mode", "auto"))
    )

    # Helpers for expected direction sign
    def _expected_sign() -> Optional[int]:
        if direction_mode == "inward":
            return -1
        if direction_mode == "outward":
            return 1
        return None  # auto

    expected_sign = _expected_sign()
    flip_candidate: Optional[int] = None
    flip_counter = 0
    min_tracks_for_flip = int(cfg.get("min_tracks_for_flip", 5))
    flip_hold_frames = int(cfg.get("flip_hold_frames", 2))
    min_radial_speed_for_flip = float(cfg.get("min_radial_speed_for_flip", 1.0))

    next_track_id = 1
    active_tracks: Dict[int, dict] = {}  # track_id -> {"last_frame": fid, "last_xy": (x,y), "gap": 0, "misses": 0, "spots": {fid: spot}, "history": [(x,y)]}
    finished: List[Track] = []

    total_frames = len(sorted_frame_ids)
    for idx, fid in enumerate(sorted_frame_ids):
        if cancel_event and cancel_event.is_set():
            return []
        if progress_callback:
            progress_callback(int(100 * idx / max(1, total_frames)))
        if total_frames > 0 and (idx % max(1, total_frames // 10) == 0):
            print(f"[DirectionalLink] progress {idx}/{total_frames} (fid={fid})")

        lf = file_map.get(fid)
        detections = lf.spots if lf else []
        det_points = np.array([[s.center_x, s.center_y] for s in detections], dtype=np.float32) if detections else np.zeros((0, 2), dtype=np.float32)
        det_rings = []
        det_radials = []
        for pt in det_points:
            ru, r = radial_unit_and_radius(center_xy, (float(pt[0]), float(pt[1])))
            det_radials.append((ru, r))
            det_rings.append(ring_bin_index(r, ring_bin_px))

        track_ids = list(active_tracks.keys())
        assigned_tracks = set()
        assigned_dets = set()

        if len(track_ids) and len(det_points):
            cost = np.full((len(track_ids), len(det_points)), 1e9, dtype=np.float32)
            for i, tid in enumerate(track_ids):
                track_data = active_tracks[tid]
                last_xy = np.array(track_data["last_xy"], dtype=np.float32)
                ru_last, r_last = radial_unit_and_radius(center_xy, track_data["last_xy"])
                # Dynamic max distance per track
                max_allowed = base_max_dist
                if dynamic_max:
                    step = median_step_magnitude(track_data.get("history", []))
                    if step > 0:
                        max_allowed = step + dynamic_margin
                for j, det_xy in enumerate(det_points):
                    delta = det_xy - last_xy
                    dist = float(np.linalg.norm(delta))
                    if dist > max_allowed:
                        continue
                    radial_comp = float(np.dot(delta, ru_last))
                    lateral = float(np.sqrt(max(0.0, dist * dist - radial_comp * radial_comp)))
                    angle = angular_offset_deg(ru_last, delta)
                    lateral_penalty = 0.0
                    if angle > lateral_tol:
                        lateral_penalty = (angle - lateral_tol) * 0.1 * (dist + 1.0)

                    dir_penalty = 0.0
                    if expected_sign is not None and radial_comp * expected_sign < 0:
                        dir_penalty = abs(radial_comp) * 2.0

                    ring_penalty = 0.0
                    if use_ring_penalty:
                        det_ring = det_rings[j]
                        last_ring = ring_bin_index(r_last, ring_bin_px)
                        ring_penalty = abs(det_ring - last_ring) * max(1.0, ring_bin_px * 0.1)

                    cost[i, j] = abs(radial_comp) + lateral + lateral_penalty + dir_penalty + ring_penalty

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e9:
                    continue
                tid = track_ids[r]
                spot = detections[c]
                active_tracks[tid]["last_xy"] = (spot.center_x, spot.center_y)
                active_tracks[tid]["last_frame"] = fid
                active_tracks[tid]["gap"] = 0
                active_tracks[tid]["misses"] = 0
                active_tracks[tid]["spots"][fid] = spot
                active_tracks[tid].setdefault("history", []).append((spot.center_x, spot.center_y))
                assigned_tracks.add(tid)
                assigned_dets.add(c)

            # Increase gap for unassigned tracks
            for tid in track_ids:
                if tid in assigned_tracks:
                    continue
                active_tracks[tid]["gap"] += 1
                active_tracks[tid]["misses"] += 1
        else:
            assigned_dets = set()
            for tid in track_ids:
                active_tracks[tid]["gap"] += 1
                active_tracks[tid]["misses"] += 1

        # Start new tracks for unmatched detections
        for j, spot in enumerate(detections):
            if j in assigned_dets:
                continue
            tid = next_track_id
            next_track_id += 1
            active_tracks[tid] = {
                "last_frame": fid,
                "last_xy": (spot.center_x, spot.center_y),
                "gap": 0,
                "misses": 0,
                "spots": {fid: spot},
                "history": [(spot.center_x, spot.center_y)],
            }

        # Retire tracks that exceeded gap
        to_remove = []
        for tid, data in active_tracks.items():
            if data["gap"] > max_gap or data.get("misses", 0) > max_misses:
                if len(data["spots"]) >= min_len:
                    finished.append(build_track(track_id=tid, spots_by_image_id=data["spots"]))
                to_remove.append(tid)
        for tid in to_remove:
            active_tracks.pop(tid, None)

        # --- Global direction flip detection ---
        if direction_mode == "auto" and active_tracks:
            radial_signs = []
            for tid, data in active_tracks.items():
                hist = data.get("history", [])
                if len(hist) < 2:
                    continue
                p_prev = np.array(hist[-2], dtype=np.float32)
                p_last = np.array(hist[-1], dtype=np.float32)
                ru_last, _ = radial_unit_and_radius(center_xy, (float(p_prev[0]), float(p_prev[1])))
                disp = p_last - p_prev
                radial_comp = float(np.dot(disp, ru_last))
                if abs(radial_comp) < min_radial_speed_for_flip:
                    continue
                radial_signs.append(np.sign(radial_comp))

            if len(radial_signs) >= min_tracks_for_flip:
                median_sign = int(np.sign(np.median(radial_signs)))
                if median_sign == 0:
                    flip_candidate = None
                    flip_counter = 0
                elif expected_sign is None:
                    expected_sign = median_sign
                    print(f"[DirectionalLink] auto direction initialized to {expected_sign:+d}")
                elif median_sign != expected_sign:
                    if flip_candidate == median_sign:
                        flip_counter += 1
                    else:
                        flip_candidate = median_sign
                        flip_counter = 1
                    if flip_counter >= flip_hold_frames:
                        expected_sign = flip_candidate
                        print(f"[DirectionalLink] direction flipped to {expected_sign:+d} after {flip_counter} checks")
                        flip_candidate = None
                        flip_counter = 0
                else:
                    flip_candidate = None
                    flip_counter = 0

    # Flush remaining active tracks
    for tid, data in active_tracks.items():
        if len(data["spots"]) >= min_len:
            finished.append(build_track(track_id=tid, spots_by_image_id=data["spots"]))

    return finished

def track_spots_optical_flow(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Tracks spots from a reference frame using the sparse Lucas-Kanade optical flow.

    This method identifies a set of feature points (spots) on a single
    reference frame and attempts to follow them through the subsequent frames
    in the sequence. It includes image preprocessing to enhance spot visibility
    for the tracking algorithm.

    Args:
        all_files_map: A dictionary mapping frame IDs to their LeedFile objects.
        sorted_frame_ids: A chronologically sorted list of all frame IDs to process.
        reference_frame_id: The ID of the frame containing the initial spots to track.
        params: A dictionary of parameters for the LK algorithm (e.g., 'lk_win_size')
                and for creating new ROIs ('roi_size').
        progress_callback: An optional function to report progress as an int percentage.

    Returns:
        A list of Track objects for the successfully tracked spots.
        
    Raises:
        ValueError: If the reference frame contains no spots or cannot be loaded.
    """
    
    subtract_background = bool(params.get("subtract_background"))
    background_image = None
    if subtract_background:
        background_image = _load_background_image(params.get("background_filepath"))
        if background_image is None:
            subtract_background = False

    # --- 1. Initialization ---
    # Get the reference file and the initial spots to track.
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")

    # Convert SpotROI objects into the numpy format required by OpenCV: (N, 1, 2)
    initial_spots = ref_file.spots
    points_to_track = np.array([[s.center_x, s.center_y] for s in initial_spots], dtype=np.float32).reshape(-1, 1, 2)

    # Setup parameters for the Lucas-Kanade optical flow algorithm.
    tracks_map = {
        spot.id: build_track_with_initial_spot(
            track_id=spot.id,
            frame_id=reference_frame_id,
            spot=spot,
        )
        for spot in initial_spots
    }
    
    # Setup parameters for the Lucas-Kanade optical flow algorithm.
    win_size = params.get('lk_win_size', 31)
    max_level = params.get('lk_max_level', 3)
    max_iterations = params.get('lk_max_iterations', 10)
    epsilon = params.get('lk_epsilon', 0.03)
    min_eig = params.get('lk_min_eigen_threshold', 1e-5)

    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon),
        minEigThreshold=min_eig,
    )
    use_centroid_refinement = params.get("use_centroid_refinement", False)
    centroid_patch_multiplier = params.get("centroid_patch_multiplier", 1.5)
    centroid_rel_threshold = params.get("centroid_rel_threshold", 0.4)
    centroid_max_expand = params.get("centroid_max_expand", 1)
    
    # Load and prepare the first frame.
    prev_gray = _load_and_prep_image(ref_file.filepath, background_image, subtract_background)
    if prev_gray is None: raise ValueError("Could not load reference image.")
        
    ref_frame_index = sorted_frame_ids.index(reference_frame_id)
    total_steps = len(sorted_frame_ids) - 1
    
    # --- 2. Forward Pass ---
    # This loop tracks the points from the reference frame to the end of the sequence.
    p0 = points_to_track
    good_track_ids = list(tracks_map.keys())  # Keep track of which tracks are still active.

    for i in range(ref_frame_index + 1, len(sorted_frame_ids)):
        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_percent = int(((i - ref_frame_index) / total_steps) * 100)
            progress_callback(progress_percent)

        # Stop if all points have been lost.
        if len(p0) == 0: break
        
        frame_id = sorted_frame_ids[i]
        current_gray = _load_and_prep_image(
            all_files_map[frame_id].filepath,
            background_image,
            subtract_background,
        )
        if current_gray is None: continue
        
        # Calculate optical flow. p1 contains new positions, status indicates success.
        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
        
        # Filter out points that were lost during tracking (status == 0).
        good_new = p1[status == 1] if p1 is not None else np.array([])
        surviving_track_ids = [good_track_ids[j] for j, s in enumerate(status) if s == 1]

        # Update the tracks with the new, successfully tracked points.
        for idx, (track_id, (x, y)) in enumerate(zip(surviving_track_ids, good_new)):
            roi_size = params.get('roi_size', 32); roi_half = roi_size//2
            if use_centroid_refinement and current_gray.size:
                patch_r = max(1.0, (roi_size / 2) * float(centroid_patch_multiplier))
                refined = refine_center_by_centroid(
                    current_gray,
                    np.array([x, y], dtype=np.float32),
                    patch_r,
                    float(centroid_rel_threshold),
                    int(max(0, centroid_max_expand)),
                )
                x, y = refined.tolist()
                good_new[idx] = refined
            new_rect = QRect(int(x - roi_half), int(y - roi_half), roi_size, roi_size)
            new_spot = build_spot_roi(
                spot_id=track_id,
                roi_rect=new_rect,
                center_x=x,
                center_y=y,
                source="optical_flow",
                intensity=0.0,
            )
            tracks_map[track_id].spots_by_image_id[frame_id] = new_spot
            
        # Prepare for the next iteration.
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        good_track_ids = surviving_track_ids
    
    print("Tracking finished.")
    if progress_callback: progress_callback(100)
    
    # Return the final list of tracks.
    return list(tracks_map.values())


def _build_direction_filter_context_for_legacy_tracking(
    params: dict,
    *,
    all_files_map: Dict[int, LeedFile],
    subset_frame_ids: List[int],
    anchor_filepath: str | None,
):
    return build_direction_filter_context(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=anchor_filepath,
        estimate_pattern_center_fn=estimate_pattern_center_from_detections,
    )


def _legacy_track_spots_tapir_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline PyTorch TAPIR.

    This backend treats each reference spot center as a TAPIR query point on the
    reference frame and converts the returned point trajectories into Track / SpotROI
    objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline TAPIR requires at least two frames starting from the reference.")

    tapir_models_dir = Path(params.get("tapir_models_dir", DEFAULT_TAPIR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_tapir_checkpoints(tapir_models_dir)
        selected_checkpoint = select_default_tapir_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No TAPIR checkpoint selected. Place a checkpoint in tracking_models/tapir "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    tapir_settings = TapirRuntimeSettings(
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25)))
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tapnext_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapir_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapir_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = build_track_with_initial_spot(
                track_id=reference_spot.id,
                frame_id=reference_frame_id,
                spot=reference_spot,
            )
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_tapir_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_tapnext_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapir_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=tapir_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_tapir_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_locotrack_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline PyTorch LocoTrack.

    This backend mirrors TAPIR integration: each reference spot center becomes
    one query point, and returned trajectories are converted to Track / SpotROI
    objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline LocoTrack requires at least two frames starting from the reference.")

    locotrack_models_dir = Path(params.get("locotrack_models_dir", DEFAULT_LOCOTRACK_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_locotrack_checkpoints(locotrack_models_dir)
        selected_checkpoint = select_default_locotrack_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No LocoTrack checkpoint selected. Place a checkpoint in tracking_models/locotrack "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    locotrack_settings = LocotrackRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("locotrack_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_size=str(
            params.get(
                "model_size",
                _infer_locotrack_model_size(checkpoint_path, fallback="small"),
            )
        ),
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_pipspp_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_locotrack_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=locotrack_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_locotrack_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_pipspp_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_locotrack_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=locotrack_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_locotrack_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_cotracker3_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline CoTracker3.

    This backend mirrors TAPIR / LocoTrack integration: each reference spot center
    becomes one query point, and returned trajectories are converted to Track / SpotROI
    objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline CoTracker3 requires at least two frames starting from the reference.")

    cotracker_models_dir = Path(params.get("cotracker3_models_dir", DEFAULT_COTRACKER_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_cotracker_checkpoints(cotracker_models_dir)
        selected_checkpoint = select_default_cotracker_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No CoTracker3 checkpoint selected. Place a checkpoint in tracking_models/cotracker3 "
            "and pass checkpoint_path in params."
        )
    if "online" in Path(checkpoint_path).name.lower():
        raise ValueError(
            "Selected CoTracker3 checkpoint looks online/sliding-window. "
            "Offline CoTracker3 expects scaled_offline.pth or baseline_offline.pth."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    cotracker_settings = CotrackerRuntimeSettings(
        repo_path=(
            params.get("repo_path")
            or params.get("cotracker3_repo_path")
            or params.get("cotracker_repo_path")
        ),
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        window_len=int(params.get("window_len", 60)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_cotracker3_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=cotracker_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_cotracker3_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_cotracker3_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=cotracker_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_cotracker3_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_pipspp_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline PIPs++.

    This backend mirrors the other point-tracker integrations: each reference
    spot center becomes one query point, and returned trajectories are converted
    to Track / SpotROI objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline PIPs++ requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("pipspp_repo_path")
    pipspp_models_dir = Path(params.get("pipspp_models_dir", DEFAULT_PIPSPP_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_pipspp_checkpoints(
            pipspp_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_pipspp_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No PIPs++ checkpoint selected. Add a checkpoint directory such as "
            "`reference_model/` under tracking_models/pipspp or the local repo and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    pipspp_settings = PipsppRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", (256, 384)),
        iters=int(params.get("iters", 4)),
        beautify=bool(params.get("beautify", True)),
        use_inbounds_visibility=bool(params.get("use_inbounds_visibility", True)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_pipspp_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=pipspp_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_pipspp_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_pipspp_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=pipspp_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_pipspp_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_mftiq_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline MFTIQ.

    This backend mirrors the other point-tracker integrations: each reference
    spot center becomes one query point, and returned trajectories are converted
    to Track / SpotROI objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline MFTIQ requires at least two frames starting from the reference.")

    repo_path = params.get("repo_path") or params.get("mftiq_repo_path")
    mftiq_models_dir = Path(params.get("mftiq_models_dir", DEFAULT_MFTIQ_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_mftiq_checkpoints(
            mftiq_models_dir,
            repo_path=repo_path,
        )
        selected_checkpoint = select_default_mftiq_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No MFTIQ checkpoint selected. Place `UOM_bs4_200k.pth` in tracking_models/mftiq "
            "or the local repo/checkpoints and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    mftiq_settings = MftiqRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        config_path=params.get("config_path"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        inference_resolution=params.get("inference_resolution"),
        occlusion_threshold=float(params.get("occlusion_threshold", 0.5)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_mftiq_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_mftiq_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=mftiq_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_mftiq_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                result.occlusion_scores[query_idx],
                result.sigma_scores[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_mftiq_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_mftiq_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=mftiq_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_mftiq_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    result.occlusion_scores[query_idx],
                    result.sigma_scores[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_sam2_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline SAM 2 mask propagation.

    Each reference spot becomes one prompted object. The returned masks are
    reduced to one representative point per frame and converted to Track / SpotROI.
    """
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
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_mask_components = int(params.get("max_mask_components", 0) or 0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_sam2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_sam2_sequence_to_track(
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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
                ok = _append_sam2_sequence_to_track(
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


def _legacy_track_spots_tapnext_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using sequential offline TAPNext.

    This backend mirrors the other point-tracker integrations: each reference
    spot center becomes one query point, and returned trajectories are converted
    to Track / SpotROI objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline TAPNext requires at least two frames starting from the reference.")

    tapnext_models_dir = Path(params.get("tapnext_models_dir", DEFAULT_TAPNEXT_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_tapnext_checkpoints(tapnext_models_dir)
        selected_checkpoint = select_default_tapnext_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No TAPNext checkpoint selected. Place a checkpoint in tracking_models/tapnext "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    tapnext_settings = TapnextRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("tapnext_repo_path"),
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", 256),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_certainty_gating=bool(params.get("use_certainty_gating", False)),
        certainty_threshold=float(params.get("certainty_threshold", 0.5)),
        autocast=bool(params.get("autocast", False)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapnext_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapnext_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_tapnext_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                query_visibility_scores=result.visibility_scores[query_idx],
                query_certainty_scores=result.certainty_scores[query_idx],
                query_combined_scores=result.combined_scores[query_idx],
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapnext_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=tapnext_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_tapnext_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    query_visibility_scores=result.visibility_scores[query_idx],
                    query_certainty_scores=result.certainty_scores[query_idx],
                    query_combined_scores=result.combined_scores[query_idx],
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                )
                if not ok:
                    return None

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_trackonr_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline Track-On-R.

    This backend mirrors the other point-tracker integrations: each reference
    spot center becomes one query point, and returned trajectories are converted
    to Track / SpotROI objects compatible with the rest of Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Offline Track-On-R requires at least two frames starting from the reference.")

    trackonr_models_dir = Path(params.get("trackonr_models_dir", DEFAULT_TRACKONR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_trackonr_checkpoints(trackonr_models_dir)
        selected_checkpoint = select_default_trackonr_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No Track-On-R checkpoint selected. Place a checkpoint in tracking_models/trackonr "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    trackonr_settings = TrackonrRuntimeSettings(
        repo_path=(
            params.get("repo_path")
            or params.get("trackonr_repo_path")
            or params.get("track_on_repo_path")
        ),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", (384, 512)),
        support_grid_size=int(params.get("support_grid_size", 20)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.8)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_trackonr_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_trackonr_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=trackonr_settings,
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_trackonr_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_trackonr_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_trackonr_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=trackonr_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_trackonr_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_track_spots_tapir_online(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using causal / online PyTorch TAPIR.

    This backend keeps the TAPIR runtime state frame-by-frame and converts the
    resulting trajectories into Track / SpotROI objects compatible with Trackleed.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("BootsTAPIR Online requires at least two frames starting from the reference.")

    tapir_models_dir = Path(params.get("tapir_models_dir", DEFAULT_TAPIR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_online_tapir_checkpoints(tapir_models_dir)
        selected_checkpoint = select_default_online_tapir_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No causal TAPIR checkpoint selected. Place causal_bootstapir_checkpoint.pt "
            "in tracking_models/tapir and pass checkpoint_path in params."
        )
    checkpoint_name = Path(checkpoint_path).name.lower()
    if "causal" not in checkpoint_name and "online" not in checkpoint_name:
        raise ValueError(
            "Selected TAPIR checkpoint does not look causal/online. "
            "BootsTAPIR Online expects causal_bootstapir_checkpoint.pt or similar."
        )

    query_points = np.array(
        [[0.0, float(spot.center_y), float(spot.center_x)] for spot in ref_file.spots],
        dtype=np.float32,
    )

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    tapir_settings = TapirOnlineRuntimeSettings(
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tapir_online_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapir_online(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapir_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        converted_tracks: List[Track] = []
        for query_idx, reference_spot in enumerate(ref_file.spots):
            track = build_track_with_initial_spot(
                track_id=reference_spot.id,
                frame_id=reference_frame_id,
                spot=reference_spot,
            )
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_tapir_sequence_to_track(
                track,
                state,
                reference_spot,
                subset_frame_ids,
                frames,
                frame_shapes,
                result.tracks_xy[query_idx],
                result.visible_mask[query_idx],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=1,
                min_global_write_idx=None,
                global_start_idx=0,
                source="tapir_online",
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_tapir_online_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
                all_files_map,
                window_frame_ids,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            if loaded_window is None:
                return None
            window_frames, window_shapes = loaded_window

            query_rows: List[List[float]] = []
            query_specs: List[Tuple[SpotROI, int]] = []
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])

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
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapir_online(
                frames=window_frames,
                query_points_tyx=np.asarray(query_rows, dtype=np.float32),
                settings=tapir_settings,
            )

            write_after_global_idx = covered_until
            for query_idx, (reference_spot, seed_local_idx) in enumerate(query_specs):
                if cancel_event and cancel_event.is_set():
                    return None

                track = track_map[reference_spot.id]
                ok = _append_tapir_sequence_to_track(
                    track,
                    track_states[reference_spot.id],
                    reference_spot,
                    window_frame_ids,
                    window_frames,
                    window_shapes,
                    result.tracks_xy[query_idx],
                    result.visible_mask[query_idx],
                    roi_size=roi_size,
                    keep_invisible_as_gaps=keep_invisible_as_gaps,
                    recenter_mode=recenter_mode,
                    centroid_patch_multiplier=centroid_patch_multiplier,
                    centroid_rel_threshold=centroid_rel_threshold,
                    centroid_max_expand=centroid_max_expand,
                    centroid_max_shift_ratio=centroid_max_shift_ratio,
                    max_step_distance_px=max_step_distance_px,
                    max_reacquisition_gap=max_reacquisition_gap,
                    terminate_after_invisible_frames=terminate_after_invisible_frames,
                    direction_center_xy=direction_center_xy,
                    direction_mode=direction_mode,
                    lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                    max_radial_jump_px=max_radial_jump_px,
                    min_directional_motion_px=min_directional_motion_px,
                    cancel_event=cancel_event,
                    start_local_idx=seed_local_idx + 1,
                    min_global_write_idx=write_after_global_idx,
                    global_start_idx=start_idx,
                    source="tapir_online",
                )
                if not ok:
                    return None

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def _legacy_retrack_single_spot_tapir_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline TAPIR.

    The track is trimmed after `start_frame_id`, then TAPIR is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single TAPIR query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="TAPIR re-tracking",
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

    tapir_models_dir = Path(params.get("tapir_models_dir", DEFAULT_TAPIR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_tapir_checkpoints(tapir_models_dir)
        selected_checkpoint = select_default_tapir_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No TAPIR checkpoint selected. Place a checkpoint in tracking_models/tapir "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    tapir_settings = TapirRuntimeSettings(
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapir_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapir_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_tapir_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapir_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=tapir_settings,
            )

            ok = _append_tapir_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_tapir_online(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using causal / online TAPIR.

    The track is trimmed after `start_frame_id`, then TAPIR is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="BootsTAPIR Online re-tracking",
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

    tapir_models_dir = Path(params.get("tapir_models_dir", DEFAULT_TAPIR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_online_tapir_checkpoints(tapir_models_dir)
        selected_checkpoint = select_default_online_tapir_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No causal TAPIR checkpoint selected. Place causal_bootstapir_checkpoint.pt "
            "in tracking_models/tapir and pass checkpoint_path in params."
        )
    checkpoint_name = Path(checkpoint_path).name.lower()
    if "causal" not in checkpoint_name and "online" not in checkpoint_name:
        raise ValueError(
            "Selected TAPIR checkpoint does not look causal/online. "
            "BootsTAPIR Online expects causal_bootstapir_checkpoint.pt or similar."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    tapir_settings = TapirOnlineRuntimeSettings(
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tapir_online_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapir_online(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapir_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_tapir_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            cancel_event=cancel_event,
            start_local_idx=1,
            min_global_write_idx=None,
            global_start_idx=0,
            source="tapir_online",
        )
        if not ok:
            return None
    else:
        if progress_callback:
            progress_callback(10)

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_tapir_online_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapir_online(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=tapir_settings,
            )

            ok = _append_tapir_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
                source="tapir_online",
            )
            if not ok:
                return None

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_locotrack_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline LocoTrack.

    The track is trimmed after `start_frame_id`, then LocoTrack is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="LocoTrack re-tracking",
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

    locotrack_models_dir = Path(params.get("locotrack_models_dir", DEFAULT_LOCOTRACK_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_locotrack_checkpoints(locotrack_models_dir)
        selected_checkpoint = select_default_locotrack_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No LocoTrack checkpoint selected. Place a checkpoint in tracking_models/locotrack "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    locotrack_settings = LocotrackRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("locotrack_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_size=str(
            params.get(
                "model_size",
                _infer_locotrack_model_size(checkpoint_path, fallback="small"),
            )
        ),
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        query_chunk_size=int(params.get("query_chunk_size", 64)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_uncertainty_gating=bool(params.get("use_uncertainty_gating", True)),
        expected_dist_threshold=float(params.get("expected_dist_threshold", 0.5)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_locotrack_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=locotrack_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_locotrack_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_locotrack_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=locotrack_settings,
            )

            ok = _append_locotrack_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_cotracker3_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline CoTracker3.

    The track is trimmed after `start_frame_id`, then CoTracker3 is run from that
    point to the end of the sequence using the spot center on the start frame as
    the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="CoTracker3 re-tracking",
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

    cotracker_models_dir = Path(params.get("cotracker3_models_dir", DEFAULT_COTRACKER_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_cotracker_checkpoints(cotracker_models_dir)
        selected_checkpoint = select_default_cotracker_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No CoTracker3 checkpoint selected. Place a checkpoint in tracking_models/cotracker3 "
            "and pass checkpoint_path in params."
        )
    if "online" in Path(checkpoint_path).name.lower():
        raise ValueError(
            "Selected CoTracker3 checkpoint looks online/sliding-window. "
            "Offline CoTracker3 expects scaled_offline.pth or baseline_offline.pth."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    cotracker_settings = CotrackerRuntimeSettings(
        repo_path=(
            params.get("repo_path")
            or params.get("cotracker3_repo_path")
            or params.get("cotracker_repo_path")
        ),
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=int(params.get("inference_resolution", 256)),
        window_len=int(params.get("window_len", 60)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = int(params.get("window_overlap", 8))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_cotracker3_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=cotracker_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_cotracker3_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = min(max(0, window_overlap), max(0, window_size - 1))
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_cotracker3_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=cotracker_settings,
            )

            ok = _append_cotracker3_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_tapnext_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline TAPNext.

    The track is trimmed after `start_frame_id`, then TAPNext is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="TAPNext re-tracking",
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

    tapnext_models_dir = Path(params.get("tapnext_models_dir", DEFAULT_TAPNEXT_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_tapnext_checkpoints(tapnext_models_dir)
        selected_checkpoint = select_default_tapnext_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No TAPNext checkpoint selected. Place a checkpoint in tracking_models/tapnext "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    tapnext_settings = TapnextRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("tapnext_repo_path"),
        checkpoint_path=checkpoint_path,
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", 256),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.5)),
        use_certainty_gating=bool(params.get("use_certainty_gating", False)),
        certainty_threshold=float(params.get("certainty_threshold", 0.5)),
        autocast=bool(params.get("autocast", False)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tapnext_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_tapnext_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=tapnext_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_tapnext_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
            direction_center_xy=direction_center_xy,
            direction_mode=direction_mode,
            lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
            max_radial_jump_px=max_radial_jump_px,
            min_directional_motion_px=min_directional_motion_px,
            query_visibility_scores=result.visibility_scores[0],
            query_certainty_scores=result.certainty_scores[0],
            query_combined_scores=result.combined_scores[0],
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_tapnext_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_tapnext_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=tapnext_settings,
            )

            ok = _append_tapnext_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
                direction_center_xy=direction_center_xy,
                direction_mode=direction_mode,
                lateral_angle_tolerance_deg=lateral_angle_tolerance_deg,
                max_radial_jump_px=max_radial_jump_px,
                min_directional_motion_px=min_directional_motion_px,
                query_visibility_scores=result.visibility_scores[0],
                query_certainty_scores=result.certainty_scores[0],
                query_combined_scores=result.combined_scores[0],
                cancel_event=cancel_event,
                start_local_idx=seed_local_idx + 1,
                min_global_write_idx=covered_until,
                global_start_idx=start_idx,
            )
            if not ok:
                return None

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_trackonr_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline Track-On-R.

    The track is trimmed after `start_frame_id`, then Track-On-R is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="Track-On-R re-tracking",
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

    trackonr_models_dir = Path(params.get("trackonr_models_dir", DEFAULT_TRACKONR_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_trackonr_checkpoints(trackonr_models_dir)
        selected_checkpoint = select_default_trackonr_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No Track-On-R checkpoint selected. Place a checkpoint in tracking_models/trackonr "
            "and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    trackonr_settings = TrackonrRuntimeSettings(
        repo_path=(
            params.get("repo_path")
            or params.get("trackonr_repo_path")
            or params.get("track_on_repo_path")
        ),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", (384, 512)),
        support_grid_size=int(params.get("support_grid_size", 20)),
        use_visibility_gating=bool(params.get("use_visibility_gating", True)),
        visibility_threshold=float(params.get("visibility_threshold", 0.8)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_trackonr_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_trackonr_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=trackonr_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_trackonr_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_trackonr_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_trackonr_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=trackonr_settings,
            )

            ok = _append_trackonr_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_pipspp_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline PIPs++.

    The track is trimmed after `start_frame_id`, then PIPs++ is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="PIPs++ re-tracking",
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

    pipspp_models_dir = Path(params.get("pipspp_models_dir", DEFAULT_PIPSPP_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_pipspp_checkpoints(
            pipspp_models_dir,
            repo_path=params.get("repo_path") or params.get("pipspp_repo_path"),
        )
        selected_checkpoint = select_default_pipspp_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No PIPs++ checkpoint selected. Place a checkpoint in tracking_models/pipspp "
            "or in repo/reference_model and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    pipspp_settings = PipsppRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("pipspp_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        device=params.get("device", "auto"),
        inference_resolution=params.get("inference_resolution", (256, 384)),
        iters=int(params.get("iters", 4)),
        beautify=bool(params.get("beautify", True)),
        use_inbounds_visibility=bool(params.get("use_inbounds_visibility", True)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_pipspp_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_pipspp_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=pipspp_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_pipspp_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_pipspp_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_pipspp_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=pipspp_settings,
            )

            ok = _append_pipspp_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_retrack_single_spot_mftiq_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline MFTIQ.

    The track is trimmed after `start_frame_id`, then MFTIQ is run from that
    point to the end of the sequence using the spot center on the start frame
    as the single query point.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="MFTIQ re-tracking",
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

    mftiq_models_dir = Path(params.get("mftiq_models_dir", DEFAULT_MFTIQ_MODELS_DIR))
    checkpoint_path = params.get("checkpoint_path")
    if not checkpoint_path:
        detected_checkpoints = discover_mftiq_checkpoints(
            mftiq_models_dir,
            repo_path=params.get("repo_path") or params.get("mftiq_repo_path"),
        )
        selected_checkpoint = select_default_mftiq_checkpoint(detected_checkpoints)
        checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
    if not checkpoint_path:
        raise ValueError(
            "No MFTIQ checkpoint selected. Place `UOM_bs4_200k.pth` in tracking_models/mftiq "
            "or in repo/checkpoints and pass checkpoint_path in params."
        )

    query_points = np.array(
        [[0.0, float(start_spot.center_y), float(start_spot.center_x)]],
        dtype=np.float32,
    )

    mftiq_settings = MftiqRuntimeSettings(
        repo_path=params.get("repo_path") or params.get("mftiq_repo_path"),
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        config_path=params.get("config_path"),
        device=params.get("device", "auto"),
        backend_mode=params.get("backend_mode", "subprocess"),
        inference_resolution=params.get("inference_resolution"),
        occlusion_threshold=float(params.get("occlusion_threshold", 0.5)),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )

    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    recenter_mode = str(
        params.get(
            "recenter_mode",
            "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
        )
    ).strip().lower()
    centroid_patch_multiplier = float(
        params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
    )
    centroid_rel_threshold = float(params.get("centroid_rel_threshold", 0.35))
    centroid_max_expand = int(max(0, params.get("centroid_max_expand", 1)))
    centroid_max_shift_ratio = float(params.get("centroid_max_shift_ratio", 0.45))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_mftiq_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0))
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0))
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0))
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px

    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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
        result = run_mftiq_offline(
            frames=frames,
            query_points_tyx=query_points,
            settings=mftiq_settings,
        )

        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(90)

        ok = _append_mftiq_sequence_to_track(
            track_to_update,
            state,
            start_spot,
            subset_frame_ids,
            frames,
            frame_shapes,
            result.tracks_xy[0],
            result.visible_mask[0],
            result.occlusion_scores[0],
            result.sigma_scores[0],
            roi_size=roi_size,
            keep_invisible_as_gaps=keep_invisible_as_gaps,
            recenter_mode=recenter_mode,
            centroid_patch_multiplier=centroid_patch_multiplier,
            centroid_rel_threshold=centroid_rel_threshold,
            centroid_max_expand=centroid_max_expand,
            centroid_max_shift_ratio=centroid_max_shift_ratio,
            max_step_distance_px=max_step_distance_px,
            max_reacquisition_gap=max_reacquisition_gap,
            terminate_after_invisible_frames=terminate_after_invisible_frames,
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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_mftiq_window_overlap(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        covered_until = 0

        for window_idx, start_idx in enumerate(window_starts):
            if cancel_event and cancel_event.is_set():
                return None

            end_idx = min(len(subset_frame_ids), start_idx + window_size)
            window_frame_ids = subset_frame_ids[start_idx:end_idx]
            if len(window_frame_ids) < 2:
                continue

            loaded_window = _load_frames_for_tapir(
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
            query_seed_ids = set(window_frame_ids[: max(1, overlap + 1)])
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
                progress_callback(int(base))

            result = run_mftiq_offline(
                frames=window_frames,
                query_points_tyx=np.asarray(
                    [[float(seed_local_idx), float(seed_spot.center_y), float(seed_spot.center_x)]],
                    dtype=np.float32,
                ),
                settings=mftiq_settings,
            )

            ok = _append_mftiq_sequence_to_track(
                track_to_update,
                state,
                start_spot,
                window_frame_ids,
                window_frames,
                window_shapes,
                result.tracks_xy[0],
                result.visible_mask[0],
                result.occlusion_scores[0],
                result.sigma_scores[0],
                roi_size=roi_size,
                keep_invisible_as_gaps=keep_invisible_as_gaps,
                recenter_mode=recenter_mode,
                centroid_patch_multiplier=centroid_patch_multiplier,
                centroid_rel_threshold=centroid_rel_threshold,
                centroid_max_expand=centroid_max_expand,
                centroid_max_shift_ratio=centroid_max_shift_ratio,
                max_step_distance_px=max_step_distance_px,
                max_reacquisition_gap=max_reacquisition_gap,
                terminate_after_invisible_frames=terminate_after_invisible_frames,
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _legacy_track_spots_cutie_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline Cutie mask propagation.

    Each reference spot becomes one seeded Cutie object. The returned masks are
    reduced to one representative point per frame and converted to Track / SpotROI.
    """
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
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    roi_size = int(params.get("roi_size", 32))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_cutie_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [_spot_polygon_seed_mask(spot, frames[0].shape) for spot in ref_file.spots],
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_cutie_sequence_to_track(
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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
                    _spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
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
                ok = _append_cutie_sequence_to_track(
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


def _legacy_track_spots_xmem2_offline(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots from a reference frame using offline XMem2 / XMem++ mask propagation.

    Each reference spot becomes one seeded XMem2 object. The returned masks are
    reduced to one representative point per frame and converted to Track / SpotROI.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")
    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")
    missing_polygon_seed_ids = _collect_missing_polygon_seed_spot_ids(ref_file.spots)
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
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    roi_size = int(params.get("roi_size", 32))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(ref_file.filepath),
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_xmem2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy = direction_filter.direction_center_xy

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [_spot_polygon_seed_mask(spot, frames[0].shape) for spot in ref_file.spots],
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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_xmem2_sequence_to_track(
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
            reference_spot.id: {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "last_valid_mask_area": float(reference_spot.roi_rect.width() * reference_spot.roi_rect.height()),
                "last_valid_bbox_aspect_ratio": (
                    float(reference_spot.roi_rect.width()) / float(reference_spot.roi_rect.height())
                    if reference_spot.roi_rect.height() > 0
                    else None
                ),
                "last_valid_bbox_center_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            for reference_spot in ref_file.spots
        }
        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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
                    _spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
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
                ok = _append_xmem2_sequence_to_track(
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


def _legacy_retrack_single_spot_sam2_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline SAM 2.

    The track is trimmed after `start_frame_id`, then SAM 2 is rerun from that
    point to the end of the sequence using the spot center and ROI on the start
    frame as the seed prompt.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="SAM 2 re-tracking",
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

    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_mask_components = int(params.get("max_mask_components", 0) or 0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_sam2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

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

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "last_valid_mask_area": float(start_mask_area),
        "last_valid_bbox_aspect_ratio": start_bbox_aspect_ratio,
        "last_valid_bbox_center_xy": np.asarray(start_bbox_center_xy, dtype=np.float32),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
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

        ok = _append_sam2_sequence_to_track(
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

        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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

            ok = _append_sam2_sequence_to_track(
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


def _legacy_retrack_single_spot_cutie_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline Cutie.

    The track is trimmed after `start_frame_id`, then Cutie is rerun from that
    point to the end of the sequence using the spot center and ROI on the start
    frame as the seed prompt.
    """
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="Cutie re-tracking",
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

    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_cutie_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

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

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "last_valid_mask_area": float(start_mask_area),
        "last_valid_bbox_aspect_ratio": start_bbox_aspect_ratio,
        "last_valid_bbox_center_xy": np.asarray(start_bbox_center_xy, dtype=np.float32),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [_spot_polygon_seed_mask(start_spot, frames[0].shape)],
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

        ok = _append_cutie_sequence_to_track(
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

        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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
                    [_spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)],
                    dtype=bool,
                ),
            )

            ok = _append_cutie_sequence_to_track(
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


def _legacy_collect_xmem2_permanent_reference_spots(
    track: Track,
    upto_frame_id: int,
    sorted_frame_ids: List[int],
) -> List[Tuple[int, SpotROI]]:
    """Collect the earliest, current, and explicitly marked XMem2 reference spots."""
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


def _legacy_retrack_single_spot_xmem2_offline(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Re-track a single existing track from a chosen frame using offline XMem2 / XMem++.

    The track is trimmed after `start_frame_id`, then XMem2 is rerun from that
    point to the end of the sequence using the spot center and ROI on the start
    frame as the seed prompt.
    """
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
    reference_entries = _legacy_collect_xmem2_permanent_reference_spots(
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

    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    max_mask_area_ratio_change = float(params.get("max_mask_area_ratio_change", 0.0) or 0.0)
    max_bbox_aspect_ratio_change = float(params.get("max_bbox_aspect_ratio_change", 0.0) or 0.0)
    max_bbox_center_jump_px = float(params.get("max_bbox_center_jump_px", 0.0) or 0.0)
    start_file = all_files_map.get(start_frame_id)
    direction_filter = _build_direction_filter_context_for_legacy_tracking(
        params,
        all_files_map=all_files_map,
        subset_frame_ids=subset_frame_ids,
        anchor_filepath=str(start_file.filepath) if start_file is not None else None,
    )
    direction_mode = direction_filter.direction_mode
    lateral_angle_tolerance_deg = direction_filter.lateral_angle_tolerance_deg
    max_radial_jump_px = direction_filter.max_radial_jump_px
    min_directional_motion_px = direction_filter.min_directional_motion_px
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_xmem2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    force_full_sequence = len(reference_entries) > 1
    start_local_idx = int(start_frame_index - subset_start_index + 1)

    direction_center_xy = direction_filter.direction_center_xy

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

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

    state = {
        "last_valid_xy": np.array(
            [float(start_spot.center_x), float(start_spot.center_y)],
            dtype=np.float32,
        ),
        "last_valid_mask_area": float(start_mask_area),
        "last_valid_bbox_aspect_ratio": start_bbox_aspect_ratio,
        "last_valid_bbox_center_xy": np.asarray(start_bbox_center_xy, dtype=np.float32),
        "invisible_streak": 0,
        "terminated": False,
        "expected_direction_sign": None,
    }

    if force_full_sequence or window_size <= 1 or window_size >= len(subset_frame_ids):
        loaded = _load_frames_for_tapir(
            all_files_map,
            subset_frame_ids,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if loaded is None:
            return None
        frames, frame_shapes = loaded
        query_initial_masks = np.asarray(
            [_spot_polygon_seed_mask(reference_spot, frames[0].shape) for _, reference_spot in reference_entries],
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

        ok = _append_xmem2_sequence_to_track(
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

        window_plans, _ = build_tracking_window_plans(
            subset_frame_ids,
            window_size,
            window_overlap,
            require_reseed_overlap=True,
        )
        covered_until = 0

        for window in window_plans:
            if cancel_event and cancel_event.is_set():
                return None

            window_idx = window.index
            start_idx = window.start_idx
            end_idx = window.end_idx
            window_frame_ids = window.frame_ids

            loaded_window = _load_frames_for_tapir(
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
                    [_spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)],
                    dtype=bool,
                ),
            )

            ok = _append_xmem2_sequence_to_track(
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


def _legacy_retrack_single_spot_optical_flow(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Track]:
    """
    Performs optical flow tracking for a single, existing track starting from a specific frame.
    It removes all subsequent points in the track and recalculates them.

    Args:
        track_to_update: The Track object to be modified.
        start_frame_id: The ID of the frame from which to start re-tracking.
        all_files_map: A map of all file IDs to their LeedFile objects.
        sorted_frame_ids: A sorted list of all frame IDs.
        params: A dictionary of parameters (lk_win_size, lk_max_level, etc.).
        progress_callback: A function to report progress.

    Returns:
        The modified Track object with the re-tracked points.
    """
    # --- 1. Initialization ---
    retrack_start = prepare_retrack_start_context(
        track_to_update,
        start_frame_id=start_frame_id,
        sorted_frame_ids=sorted_frame_ids,
        operation_label="re-tracking",
    )
    start_spot = retrack_start.start_spot
    p0 = np.array([[[start_spot.center_x, start_spot.center_y]]], dtype=np.float32)
    start_frame_index = retrack_start.start_frame_index
    prev_gray = _load_and_prep_image(all_files_map[start_frame_id].filepath)
    if prev_gray is None:
        raise ValueError("Could not load the starting image for re-tracking.")

    # --- 2. LK Parameters ---
    win_size = params.get('lk_win_size', 31)
    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=params.get('lk_max_level', 3),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            params.get('lk_max_iterations', 10),
            params.get('lk_epsilon', 0.03),
        ),
        minEigThreshold=params.get('lk_min_eigen_threshold', 1e-5),
    )

    # --- 3. Tracking Loop ---
    num_frames_to_process = len(sorted_frame_ids) - (start_frame_index + 1)
    if num_frames_to_process <= 0:
        return track_to_update # Nothing else to process

    for i in range(start_frame_index + 1, len(sorted_frame_ids)):
        if cancel_event and cancel_event.is_set():
            return None
        if len(p0) == 0: break # The spot disappeared

        current_frame_id = sorted_frame_ids[i]
        current_gray = _load_and_prep_image(all_files_map[current_frame_id].filepath)
        if current_gray is None: continue

        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)

        if status is not None and status[0][0] == 1:
            x, y = p1[0][0]
            roi_size = params.get('roi_size', 32)
            roi_half = roi_size // 2
            new_rect = QRect(int(x - roi_half), int(y - roi_half), roi_size, roi_size)
            new_spot = build_spot_roi(
                spot_id=start_spot.id,
                roi_rect=new_rect,
                center_x=x,
                center_y=y,
                source="re-tracked",
                intensity=0.0,
            )
            track_to_update.spots_by_image_id[current_frame_id] = new_spot

            prev_gray = current_gray.copy()
            p0 = p1
        else:
            # Lost the point, stop re-tracking
            break
        
        if progress_callback:
            progress = int(((i - start_frame_index) / num_frames_to_process) * 100)
            progress_callback(progress)

    if progress_callback:
        progress_callback(100)
        
    return track_to_update

def optical_flow_with_skip_anchor(
    frames: List[np.ndarray],
    points0: List[Tuple[float, float]],
    lk_params: Optional[Dict] = None,
    err_threshold: float = 30.0,
    fb_threshold: float = 8.0,
    use_forward_backward: bool = True,
    max_skip_frames_per_point: int = 0,
    oob_behavior: str = "skip",
    use_ncc_check: bool = False,
    ncc_threshold: float = 0.65,
    ncc_patch_radius: int = 4,
    use_centroid_refinement: bool = False,
    centroid_patch_multiplier: float = 1.5,
    centroid_rel_threshold: float = 0.4,
    centroid_max_expand: int = 1,
    roi_size: int = 32,
    cancel_event: Optional[Event] = None,
) -> Optional[List[List[Tuple[int, float, float]]]]:
    """
    Skip-anchor tracker: when a frame fails to produce a match we do not append
    a point and we keep the anchor on the last successful frame. The next LK
    solve always starts from that anchor frame instead of the previous attempt.

    Args:
        frames: list of grayscale images (uint8) with identical shape.
        points0: list of (x, y) coordinates defined on frame 0.
        err_threshold: maximum acceptable LK residual (smaller is stricter).
        fb_threshold: closure distance for the forward-backward check in pixels.
        max_skip_frames_per_point: how many consecutive misses are tolerated.
        oob_behavior: either 'skip' (ignore) or 'mark_lost' when leaving frame.
        use_ncc_check: enable an extra NCC gate around the tracked point.
        ncc_threshold: minimum NCC value to treat the match as valid.
        ncc_patch_radius: NCC patch radius (e.g., 4 → 9x9 window).
        use_centroid_refinement: recenter ROI via centroid after a success.
        centroid_patch_multiplier: ROI radius multiplier for centroid cutout.
        centroid_rel_threshold: relative intensity threshold for centroid mask.
        centroid_max_expand: how many times the centroid patch can expand.
        roi_size: base ROI size used for centroid operations.

    Returns:
        tracks: list of polylines; each entry is a list of (frame_idx, x, y)
                recorded only for successful frames.
    """
    if frames is None or len(frames) < 2:
        raise ValueError("At least two frames are required for skip-anchor tracking.")

    H, W = frames[0].shape[:2]
    base_lk = dict(
        winSize=(41, 41),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
        minEigThreshold=1e-5,
        flags=0
    )
    if lk_params:
        base_lk.update(lk_params)

    n = len(points0)
    max_skip_frames_per_point = max(0, int(max_skip_frames_per_point or 0))
    oob_behavior = (oob_behavior or "skip").lower()
    if oob_behavior not in {"skip", "mark_lost"}:
        oob_behavior = "skip"
    centroid_patch_multiplier = float(centroid_patch_multiplier)
    centroid_rel_threshold = float(centroid_rel_threshold)
    centroid_max_expand = int(max(0, centroid_max_expand))
    roi_radius = max(1.0, roi_size / 2)
    tracks: List[List[Tuple[int, float, float]]] = [[] for _ in range(n)]

    # Per-point anchors (frame index and position). Update them only after success.
    anchor_idx = np.zeros(n, dtype=int)
    anchor_pos = np.array(points0, dtype=np.float32)  # (N,2)
    alive = np.ones(n, dtype=bool)
    skip_counters = np.zeros(n, dtype=int)

    # Store the initial detections (frame 0)
    for i, (x, y) in enumerate(points0):
        if 0 <= x < W and 0 <= y < H:
            tracks[i].append((0, float(x), float(y)))
        else:
            alive[i] = False  # Initial position is outside the image

    # Main loop over target frames
    for t in range(1, len(frames)):
        if cancel_event and cancel_event.is_set():
            return None
        img_t = frames[t]

        # Consider only living tracks whose anchor frame is older than t
        idxs = [i for i in range(n) if alive[i] and anchor_idx[i] < t]
        if not idxs:
            continue

        # Group by shared anchor; OpenCV runs LK on an image pair
        from collections import defaultdict
        groups = defaultdict(list)
        for i in idxs:
            groups[anchor_idx[i]].append(i)

        for a_idx, group in groups.items():
            if cancel_event and cancel_event.is_set():
                return None
            img_anchor = frames[a_idx]
            p0 = anchor_pos[group].reshape(-1, 1, 2).astype(np.float32)

            # LK always starts from the anchor frame, never the previous attempt
            p1, st, err = cv2.calcOpticalFlowPyrLK(img_anchor, img_t, p0, None, **base_lk)

            # Optional forward-backward verification
            fb_ok = np.ones(len(group), dtype=bool)
            if use_forward_backward:
                p0_fb, st_fb, _ = cv2.calcOpticalFlowPyrLK(
                    img_t, img_anchor, p1, None,
                    winSize=base_lk["winSize"],
                    maxLevel=base_lk["maxLevel"],
                    criteria=base_lk["criteria"],
                    minEigThreshold=base_lk["minEigThreshold"]
                )
                for k in range(len(group)):
                    if st[k, 0] != 1 or st_fb is None or st_fb[k, 0] != 1:
                        fb_ok[k] = False
                    else:
                        fb_dist = float(np.linalg.norm(p0_fb[k, 0] - p0[k, 0]))
                        if fb_dist > fb_threshold:
                            fb_ok[k] = False

            # Record only successful matches; failures simply keep the previous anchor.
            for k, gi in enumerate(group):
                success = True
                out_of_bounds = False

                if st[k, 0] != 1:
                    success = False
                elif err is not None and float(err[k, 0]) > float(err_threshold):
                    success = False
                elif use_forward_backward and not fb_ok[k]:
                    success = False

                if success:
                    x, y = float(p1[k, 0, 0]), float(p1[k, 0, 1])
                    if not (0 <= x < W and 0 <= y < H):
                        success = False
                        out_of_bounds = True

                if success and use_ncc_check:
                    ncc_val = _ncc_around(
                        img_anchor,
                        img_t,
                        p0[k, 0],
                        p1[k, 0],
                        max(1, int(ncc_patch_radius)),
                    )
                    if ncc_val < float(ncc_threshold):
                        success = False

                if success and use_centroid_refinement:
                    patch_r = max(1.0, roi_radius * centroid_patch_multiplier)
                    refined = refine_center_by_centroid(
                        img_t,
                        np.array([x, y], dtype=np.float32),
                        patch_r,
                        centroid_rel_threshold,
                        centroid_max_expand,
                    )
                    x, y = refined.tolist()

                if success:
                    tracks[gi].append((t, x, y))
                    # Only now do we move the anchor to frame t
                    anchor_idx[gi] = t
                    anchor_pos[gi] = np.array([x, y], dtype=np.float32)
                    skip_counters[gi] = 0
                else:
                    if out_of_bounds and oob_behavior == "mark_lost":
                        alive[gi] = False
                        continue

                    if max_skip_frames_per_point > 0:
                        skip_counters[gi] += 1
                        if skip_counters[gi] > max_skip_frames_per_point:
                            alive[gi] = False

    return tracks


def track_spots_optical_flow_skip_anchor(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Wrapper that prepares image stacks and converts the skip-anchor LK output into Track objects.
    """
    ref_file = all_files_map.get(reference_frame_id)
    if not ref_file or not ref_file.spots:
        raise ValueError("Reference frame must contain at least one defined spot.")

    if reference_frame_id not in sorted_frame_ids:
        raise ValueError("Reference frame is missing from the provided frame list.")

    ref_index = sorted_frame_ids.index(reference_frame_id)
    subset_frame_ids = sorted_frame_ids[ref_index:]
    if len(subset_frame_ids) < 2:
        raise ValueError("Optical Flow (skip, anchor) requires at least two frames starting from the reference.")

    subtract_background = bool(params.get("subtract_background"))
    background_image = None
    if subtract_background:
        background_image = _load_background_image(params.get("background_filepath"))
        if background_image is None:
            subtract_background = False

    frames: List[np.ndarray] = []
    for idx, frame_id in enumerate(subset_frame_ids):
        if cancel_event and cancel_event.is_set():
            return None
        prepared = _load_and_prep_image(
            all_files_map[frame_id].filepath,
            background_image,
            subtract_background,
        )
        if prepared is None:
            raise ValueError(f"Could not load image for frame {frame_id}.")
        frames.append(prepared)
        if progress_callback:
            progress = int(((idx + 1) / len(subset_frame_ids)) * 40)
            progress_callback(progress)

    points0 = [(float(spot.center_x), float(spot.center_y)) for spot in ref_file.spots]
    win_size = params.get("lk_win_size", 41)
    lk_params = {
        "winSize": (win_size, win_size),
        "maxLevel": params.get("lk_max_level", 4),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            params.get("lk_max_iterations", 50),
            params.get("lk_epsilon", 0.01),
        ),
        "minEigThreshold": params.get("lk_min_eigen_threshold", 1e-5),
    }

    raw_tracks = optical_flow_with_skip_anchor(
        frames=frames,
        points0=points0,
        lk_params=lk_params,
        err_threshold=params.get("lk_err_threshold", 30.0),
        fb_threshold=params.get("lk_fb_threshold", 8.0),
        use_forward_backward=params.get("use_forward_backward", True),
        max_skip_frames_per_point=params.get("max_skip_frames", 0),
        oob_behavior=params.get("oob_behavior", "skip"),
        use_ncc_check=params.get("use_ncc_check", False),
        ncc_threshold=params.get("ncc_threshold", 0.65),
        ncc_patch_radius=params.get("ncc_patch_radius", 4),
        use_centroid_refinement=params.get("use_centroid_refinement", False),
        centroid_patch_multiplier=params.get("centroid_patch_multiplier", 1.5),
        centroid_rel_threshold=params.get("centroid_rel_threshold", 0.4),
        centroid_max_expand=params.get("centroid_max_expand", 1),
        roi_size=params.get("roi_size", 32),
        cancel_event=cancel_event,
    )
    if raw_tracks is None:
        return None

    roi_size = params.get("roi_size", 32)
    roi_half = roi_size // 2
    converted_tracks: List[Track] = []

    for spot, positions in zip(ref_file.spots, raw_tracks):
        if cancel_event and cancel_event.is_set():
            return None
        if not positions:
            continue
        track = build_track(track_id=spot.id)
        for rel_frame_idx, x, y in positions:
            frame_id = subset_frame_ids[rel_frame_idx]
            if rel_frame_idx == 0:
                track.spots_by_image_id[frame_id] = spot
                continue

            rect = QRect(int(x - roi_half), int(y - roi_half), roi_size, roi_size)
            track.spots_by_image_id[frame_id] = build_spot_roi(
                spot_id=spot.id,
                roi_rect=rect,
                center_x=x,
                center_y=y,
                source="optical_flow_skip_anchor",
                intensity=0.0,
            )
        converted_tracks.append(track)

    if progress_callback:
        progress_callback(100)
    return converted_tracks
