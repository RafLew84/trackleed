from __future__ import annotations

# stal/analysis/tracking.py

from pathlib import Path
from typing import Any, List, Dict, Callable, Tuple, Optional
from threading import Event

import cv2
import numpy as np
from PyQt6.QtCore import QRect
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2gray
from skimage.morphology import white_tophat, disk
from skimage.util import img_as_ubyte

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
from trackleed.analysis.tomp_backend import (
    DEFAULT_TAMOS_MODELS_DIR,
    DEFAULT_TOMP_MODELS_DIR,
    TompRuntimeSettings,
    discover_tamos_checkpoints,
    discover_tomp_checkpoints,
    run_tomp_offline,
    select_default_tamos_checkpoint,
    select_default_tomp_checkpoint,
)
from trackleed.analysis.polygon_mask import (
    normalize_polygon_points,
    polygon_points_are_valid,
    polygon_points_to_mask,
)


def _load_and_prep_image(filepath, background_image: Optional[np.ndarray] = None, subtract_background: bool = False):
    """
    Loads a TIFF image, normalizes it, and returns an 8-bit grayscale array suitable for LK tracking.
    """
    image_raw = load_tiff(filepath)
    if image_raw is None or image_raw.size == 0:
        return None

    if image_raw.ndim == 3:
        gray_float = rgb2gray(image_raw)
    else:
        image_float = image_raw.astype(np.float32)
        min_val, max_val = np.min(image_float), np.max(image_float)
        if max_val > min_val:
            gray_float = (image_float - min_val) / (max_val - min_val)
        else:
            gray_float = np.zeros_like(image_float)

    working_image = gray_float.copy()
    if subtract_background and background_image is not None and background_image.shape == working_image.shape:
        working_image = working_image - background_image
        np.clip(working_image, 0.0, None, out=working_image)

    blurred = gaussian_filter(working_image, sigma=5)
    processed_image = working_image - blurred
    footprint = disk(5)
    final_for_tracking = white_tophat(processed_image, footprint=footprint)
    min_val, max_val = np.min(final_for_tracking), np.max(final_for_tracking)
    if max_val > min_val:
        normalized_final = (final_for_tracking - min_val) / (max_val - min_val)
    else:
        normalized_final = np.zeros_like(final_for_tracking)
    return img_as_ubyte(normalized_final)


def _load_background_image(filepath: Optional[str | Path]) -> Optional[np.ndarray]:
    if not filepath:
        return None
    path = Path(filepath)
    bg_raw = load_tiff(path)
    if bg_raw is None or bg_raw.size == 0:
        return None
    if bg_raw.ndim == 3:
        return rgb2gray(bg_raw)
    image_float = bg_raw.astype(np.float32)
    min_val, max_val = np.min(image_float), np.max(image_float)
    if max_val > min_val:
        return (image_float - min_val) / (max_val - min_val)
    return np.zeros_like(image_float)


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


def _bbox_center_and_aspect_ratio_from_xyxy(
    bbox_xyxy: np.ndarray | tuple[float, float, float, float] | None,
) -> tuple[np.ndarray | None, float | None]:
    """Return bbox center and width/height aspect ratio for a valid xyxy box."""
    if bbox_xyxy is None:
        return None, None
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
        return None, None
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    width = float(x1 - x0)
    height = float(y1 - y0)
    if width <= 0.0 or height <= 0.0:
        return None, None
    center_xy = np.array(
        [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
        dtype=np.float32,
    )
    return center_xy, float(width / height)


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
    if normalized in {"tamos", "tamos50", "tamos_resnet50"}:
        return "tamos_resnet50"
    raise ValueError(
        "Unsupported ToMP / TaMOs model variant. "
        "Expected `tomp50`/`tomp` or `tamos_resnet50`/`tamos`."
    )


def _tomp_tracker_source_name(model_variant: str | None) -> str:
    if _normalize_tomp_model_variant_key(model_variant) == "tamos_resnet50":
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
    x = float(raw_x)
    y = float(raw_y)
    if visible:
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source=source,
    )
    setattr(spot, "tapir_raw_x", float(raw_x))
    setattr(spot, "tapir_raw_y", float(raw_y))
    setattr(spot, "tapir_refined_x", float(x))
    setattr(spot, "tapir_refined_y", float(y))
    setattr(spot, "tapir_visibility", bool(visible))
    return spot


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
    x = float(raw_x)
    y = float(raw_y)
    if visible:
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="locotrack_offline",
    )
    setattr(spot, "locotrack_raw_x", float(raw_x))
    setattr(spot, "locotrack_raw_y", float(raw_y))
    setattr(spot, "locotrack_refined_x", float(x))
    setattr(spot, "locotrack_refined_y", float(y))
    setattr(spot, "locotrack_visibility", bool(visible))
    return spot


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
    x = float(raw_x)
    y = float(raw_y)
    if visible:
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="cotracker3_offline",
    )
    setattr(spot, "cotracker3_raw_x", float(raw_x))
    setattr(spot, "cotracker3_raw_y", float(raw_y))
    setattr(spot, "cotracker3_refined_x", float(x))
    setattr(spot, "cotracker3_refined_y", float(y))
    setattr(spot, "cotracker3_visibility", bool(visible))
    return spot


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
    x = float(raw_x)
    y = float(raw_y)
    if visible:
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="tapnext_offline",
    )
    setattr(spot, "tapnext_raw_x", float(raw_x))
    setattr(spot, "tapnext_raw_y", float(raw_y))
    setattr(spot, "tapnext_refined_x", float(x))
    setattr(spot, "tapnext_refined_y", float(y))
    setattr(spot, "tapnext_visibility", bool(visible))
    if visibility_score is not None:
        setattr(spot, "tapnext_visibility_score", float(visibility_score))
    if certainty_score is not None:
        setattr(spot, "tapnext_certainty_score", float(certainty_score))
    if combined_score is not None:
        setattr(spot, "tapnext_combined_score", float(combined_score))
    return spot


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
    x = float(raw_x)
    y = float(raw_y)
    if visible:
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="trackonr_offline",
    )
    setattr(spot, "trackonr_raw_x", float(raw_x))
    setattr(spot, "trackonr_raw_y", float(raw_y))
    setattr(spot, "trackonr_refined_x", float(x))
    setattr(spot, "trackonr_refined_y", float(y))
    setattr(spot, "trackonr_visibility", bool(visible))
    return spot


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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="pipspp_offline",
    )
    setattr(spot, "pipspp_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "pipspp_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "pipspp_refined_x", float(x))
    setattr(spot, "pipspp_refined_y", float(y))
    setattr(spot, "pipspp_visibility", bool(visible))
    return spot


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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="mftiq_offline",
    )
    setattr(spot, "mftiq_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "mftiq_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "mftiq_refined_x", float(x))
    setattr(spot, "mftiq_refined_y", float(y))
    setattr(spot, "mftiq_visibility", bool(visible))
    setattr(spot, "mftiq_occlusion_score", float(occlusion_score))
    setattr(spot, "mftiq_sigma_score", float(sigma_score))
    return spot


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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="sam2_offline",
    )
    setattr(spot, "sam2_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "sam2_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "sam2_refined_x", float(x))
    setattr(spot, "sam2_refined_y", float(y))
    setattr(spot, "sam2_visibility", bool(visible))
    setattr(spot, "sam2_mask_area", float(mask_area))
    setattr(spot, "sam2_mask_score", float(mask_score))
    setattr(spot, "sam2_mask_component_count", int(mask_component_count))
    if mask_bbox_xyxy is not None:
        bbox = np.asarray(mask_bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            setattr(spot, "sam2_mask_bbox_xyxy", tuple(float(v) for v in bbox[:4].tolist()))
            _, aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(bbox[:4])
            setattr(
                spot,
                "sam2_mask_bbox_aspect_ratio",
                float(aspect_ratio) if aspect_ratio is not None else None,
            )
        else:
            setattr(spot, "sam2_mask_bbox_xyxy", None)
            setattr(spot, "sam2_mask_bbox_aspect_ratio", None)
    else:
        setattr(spot, "sam2_mask_bbox_xyxy", None)
        setattr(spot, "sam2_mask_bbox_aspect_ratio", None)
    setattr(spot, "sam2_repr_mode", str(representative_point_mode))
    setattr(spot, "sam2_prompt_mode", str(prompt_mode))
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="cutie_offline",
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
    setattr(spot, "cutie_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "cutie_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "cutie_refined_x", float(x))
    setattr(spot, "cutie_refined_y", float(y))
    setattr(spot, "cutie_visibility", bool(visible))
    setattr(spot, "cutie_mask_area", float(mask_area))
    setattr(spot, "cutie_mask_score", float(mask_score))
    if mask_bbox_xyxy is not None:
        bbox = np.asarray(mask_bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            setattr(spot, "cutie_mask_bbox_xyxy", tuple(float(v) for v in bbox[:4].tolist()))
            _, aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(bbox[:4])
            setattr(
                spot,
                "cutie_mask_bbox_aspect_ratio",
                float(aspect_ratio) if aspect_ratio is not None else None,
            )
        else:
            setattr(spot, "cutie_mask_bbox_xyxy", None)
            setattr(spot, "cutie_mask_bbox_aspect_ratio", None)
    else:
        setattr(spot, "cutie_mask_bbox_xyxy", None)
        setattr(spot, "cutie_mask_bbox_aspect_ratio", None)
    setattr(spot, "cutie_repr_mode", str(representative_point_mode))
    setattr(spot, "cutie_initial_mask_mode", str(initial_mask_mode))
    setattr(spot, "polygon_seed_used", bool(reference_spot.has_polygon_seed))
    setattr(spot, "polygon_vertex_count", int(len(spot.polygon_seed_points)))
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source="xmem2_offline",
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
    setattr(spot, "xmem2_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "xmem2_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "xmem2_refined_x", float(x))
    setattr(spot, "xmem2_refined_y", float(y))
    setattr(spot, "xmem2_visibility", bool(visible))
    setattr(spot, "xmem2_mask_area", float(mask_area))
    setattr(spot, "xmem2_mask_score", float(mask_score))
    if mask_bbox_xyxy is not None:
        bbox = np.asarray(mask_bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size >= 4 and np.all(np.isfinite(bbox[:4])):
            setattr(spot, "xmem2_mask_bbox_xyxy", tuple(float(v) for v in bbox[:4].tolist()))
            _, aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(bbox[:4])
            setattr(
                spot,
                "xmem2_mask_bbox_aspect_ratio",
                float(aspect_ratio) if aspect_ratio is not None else None,
            )
        else:
            setattr(spot, "xmem2_mask_bbox_xyxy", None)
            setattr(spot, "xmem2_mask_bbox_aspect_ratio", None)
    else:
        setattr(spot, "xmem2_mask_bbox_xyxy", None)
        setattr(spot, "xmem2_mask_bbox_aspect_ratio", None)
    setattr(spot, "xmem2_repr_mode", str(representative_point_mode))
    setattr(spot, "xmem2_initial_mask_mode", str(initial_mask_mode))
    setattr(spot, "polygon_seed_used", bool(reference_spot.has_polygon_seed))
    setattr(spot, "polygon_vertex_count", int(len(spot.polygon_seed_points)))
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
    spot = SpotROI(
        id=reference_spot.id,
        roi_rect=rect,
        center_x=x,
        center_y=y,
        intensity=1.0 if visible else 0.0,
        source=_tomp_tracker_source_name(model_variant),
    )
    setattr(spot, "tomp_raw_x", float(raw_x) if np.isfinite(raw_x) else float("nan"))
    setattr(spot, "tomp_raw_y", float(raw_y) if np.isfinite(raw_y) else float("nan"))
    setattr(spot, "tomp_refined_x", float(x))
    setattr(spot, "tomp_refined_y", float(y))
    setattr(spot, "tomp_visibility", bool(visible))
    setattr(spot, "tomp_bbox_score", float(bbox_score) if np.isfinite(bbox_score) else float("nan"))
    setattr(spot, "tomp_model_variant", str(model_variant))
    setattr(spot, "tomp_repr_mode", str(representative_point_mode))

    if bbox_xywh is not None:
        bbox_xywh_arr = np.asarray(bbox_xywh, dtype=np.float32).reshape(-1)
        if bbox_xywh_arr.size >= 4 and np.all(np.isfinite(bbox_xywh_arr[:4])):
            setattr(spot, "tomp_bbox_xywh", tuple(float(v) for v in bbox_xywh_arr[:4].tolist()))
        else:
            setattr(spot, "tomp_bbox_xywh", None)
    else:
        setattr(spot, "tomp_bbox_xywh", None)

    if bbox_xyxy is not None:
        bbox_xyxy_arr = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox_xyxy_arr.size >= 4 and np.all(np.isfinite(bbox_xyxy_arr[:4])):
            setattr(spot, "tomp_bbox_xyxy", tuple(float(v) for v in bbox_xyxy_arr[:4].tolist()))
            _, aspect_ratio = _bbox_center_and_aspect_ratio_from_xyxy(bbox_xyxy_arr[:4])
            setattr(
                spot,
                "tomp_bbox_aspect_ratio",
                float(aspect_ratio) if aspect_ratio is not None else None,
            )
        else:
            setattr(spot, "tomp_bbox_xyxy", None)
            setattr(spot, "tomp_bbox_aspect_ratio", None)
    else:
        setattr(spot, "tomp_bbox_xyxy", None)
        setattr(spot, "tomp_bbox_aspect_ratio", None)
    return spot


def _compute_point_tracker_window_starts(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> List[int]:
    """Return monotonically increasing window starts covering the whole sequence."""
    if total_frames <= 1:
        return [0]
    if window_size <= 1 or window_size >= total_frames:
        return [0]

    overlap = min(max(0, int(window_overlap)), max(0, window_size - 1))
    step = max(1, int(window_size) - overlap)
    starts = list(range(0, max(1, total_frames - 1), step))
    last_start = max(0, total_frames - window_size)
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def _compute_tapir_window_starts(total_frames: int, window_size: int, window_overlap: int) -> List[int]:
    """Backward-compatible alias for the generic point-tracker window helper."""
    return _compute_point_tracker_window_starts(total_frames, window_size, window_overlap)


def _resolve_tapnext_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    TAPNext windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked point
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_trackonr_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    Track-On-R windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked point
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_pipspp_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    PIPs++ windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked point
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_mftiq_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    MFTIQ windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked point
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_sam2_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    SAM 2 windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked seed
    object to continue propagation from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_cutie_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    Cutie windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked seed
    object to continue propagation from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_xmem2_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    XMem2 windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked seed
    object to continue propagation from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_tomp_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    ToMP / TaMOs windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked bbox
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


def _resolve_tapir_online_window_overlap(
    total_frames: int,
    window_size: int,
    window_overlap: int,
) -> int:
    """
    Online / causal TAPIR windows are restarted independently and reseeded from overlap frames.

    That means multi-window tracking needs at least one shared frame between
    neighboring windows; otherwise the next window has no already-tracked point
    on its first frame to seed from.
    """
    overlap = min(max(0, int(window_overlap)), max(0, int(window_size) - 1))
    if total_frames > 1 and 1 < int(window_size) < int(total_frames):
        overlap = max(1, overlap)
    return overlap


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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "tapir_filter_rejected", True)
                    setattr(candidate_spot, "tapir_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "tapir_filter_rejected", True)
                    setattr(candidate_spot, "tapir_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "tapir_filter_rejected", True)
                            setattr(candidate_spot, "tapir_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "tapir_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "locotrack_filter_rejected", True)
                    setattr(candidate_spot, "locotrack_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "locotrack_filter_rejected", True)
                    setattr(candidate_spot, "locotrack_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "locotrack_filter_rejected", True)
                            setattr(candidate_spot, "locotrack_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "locotrack_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "cotracker3_filter_rejected", True)
                    setattr(candidate_spot, "cotracker3_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "cotracker3_filter_rejected", True)
                    setattr(candidate_spot, "cotracker3_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "cotracker3_filter_rejected", True)
                            setattr(candidate_spot, "cotracker3_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "cotracker3_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "tapnext_filter_rejected", True)
                    setattr(candidate_spot, "tapnext_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "tapnext_filter_rejected", True)
                    setattr(candidate_spot, "tapnext_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "tapnext_filter_rejected", True)
                            setattr(candidate_spot, "tapnext_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "tapnext_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "trackonr_filter_rejected", True)
                    setattr(candidate_spot, "trackonr_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "trackonr_filter_rejected", True)
                    setattr(candidate_spot, "trackonr_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "trackonr_filter_rejected", True)
                            setattr(candidate_spot, "trackonr_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "trackonr_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "pipspp_filter_rejected", True)
                    setattr(candidate_spot, "pipspp_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "pipspp_filter_rejected", True)
                    setattr(candidate_spot, "pipspp_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "pipspp_filter_rejected", True)
                            setattr(candidate_spot, "pipspp_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "pipspp_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    setattr(candidate_spot, "mftiq_filter_rejected", True)
                    setattr(candidate_spot, "mftiq_filter_rejected_reason", "max_step_distance")

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    setattr(candidate_spot, "mftiq_filter_rejected", True)
                    setattr(candidate_spot, "mftiq_filter_rejected_reason", "max_radial_jump")

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            setattr(candidate_spot, "mftiq_filter_rejected", True)
                            setattr(candidate_spot, "mftiq_filter_rejected_reason", "direction_gating")

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "mftiq_visibility", False)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    last_valid_mask_area = state["last_valid_mask_area"]
    last_valid_bbox_aspect_ratio = state["last_valid_bbox_aspect_ratio"]
    last_valid_bbox_center_xy = state["last_valid_bbox_center_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_mask_area_ratio_change = float(max(0.0, max_mask_area_ratio_change))
    max_bbox_aspect_ratio_change = float(max(0.0, max_bbox_aspect_ratio_change))
    max_mask_components = int(max(0, max_mask_components))
    max_bbox_center_jump_px = float(max(0.0, max_bbox_center_jump_px))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
                return True

            if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                accepted = False
                rejection_reason = "invalid_mask"
            elif not (0.0 <= raw_x < float(image_w) and 0.0 <= raw_y < float(image_h)):
                accepted = False
                rejection_reason = "out_of_bounds"
            elif max_mask_components > 0 and mask_component_count > max_mask_components:
                accepted = False
                rejection_reason = "mask_fragmentation"

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

            if (
                accepted
                and max_mask_area_ratio_change > 1.0
                and last_valid_mask_area is not None
                and np.isfinite(last_valid_mask_area)
                and last_valid_mask_area > 0.0
                and np.isfinite(mask_area)
                and mask_area > 0.0
            ):
                area_ratio = max(float(mask_area), float(last_valid_mask_area)) / min(
                    float(mask_area),
                    float(last_valid_mask_area),
                )
                if area_ratio > max_mask_area_ratio_change:
                    accepted = False
                    rejection_reason = "mask_area_drift"
                    setattr(candidate_spot, "sam2_filter_rejected", True)
                    setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_aspect_ratio_change > 1.0
                and last_valid_bbox_aspect_ratio is not None
                and np.isfinite(last_valid_bbox_aspect_ratio)
                and last_valid_bbox_aspect_ratio > 0.0
                and bbox_aspect_ratio is not None
                and np.isfinite(bbox_aspect_ratio)
                and bbox_aspect_ratio > 0.0
            ):
                aspect_ratio_change = max(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                ) / min(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                )
                if aspect_ratio_change > max_bbox_aspect_ratio_change:
                    accepted = False
                    rejection_reason = "bbox_aspect_ratio_drift"
                    setattr(candidate_spot, "sam2_filter_rejected", True)
                    setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_center_jump_px > 0.0
                and last_valid_bbox_center_xy is not None
                and bbox_center_xy is not None
                and float(np.linalg.norm(bbox_center_xy - last_valid_bbox_center_xy)) > max_bbox_center_jump_px
            ):
                accepted = False
                rejection_reason = "bbox_jump"
                setattr(candidate_spot, "sam2_filter_rejected", True)
                setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    rejection_reason = "max_step_distance"
                    setattr(candidate_spot, "sam2_filter_rejected", True)
                    setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    rejection_reason = "max_radial_jump"
                    setattr(candidate_spot, "sam2_filter_rejected", True)
                    setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            rejection_reason = "direction_gating"
                            setattr(candidate_spot, "sam2_filter_rejected", True)
                            setattr(candidate_spot, "sam2_filter_rejected_reason", rejection_reason)

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "sam2_visibility", False)
                setattr(forced_spot, "sam2_filter_rejected", True)
                setattr(forced_spot, "sam2_filter_rejected_reason", rejection_reason)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        if np.isfinite(mask_area) and mask_area > 0.0:
            last_valid_mask_area = float(mask_area)
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["last_valid_mask_area"] = last_valid_mask_area
    state["last_valid_bbox_aspect_ratio"] = last_valid_bbox_aspect_ratio
    state["last_valid_bbox_center_xy"] = last_valid_bbox_center_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    last_valid_mask_area = state["last_valid_mask_area"]
    last_valid_bbox_aspect_ratio = state["last_valid_bbox_aspect_ratio"]
    last_valid_bbox_center_xy = state["last_valid_bbox_center_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_mask_area_ratio_change = float(max(0.0, max_mask_area_ratio_change))
    max_bbox_aspect_ratio_change = float(max(0.0, max_bbox_aspect_ratio_change))
    max_bbox_center_jump_px = float(max(0.0, max_bbox_center_jump_px))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if (
                accepted
                and max_mask_area_ratio_change > 1.0
                and last_valid_mask_area is not None
                and np.isfinite(last_valid_mask_area)
                and last_valid_mask_area > 0.0
                and np.isfinite(mask_area)
                and mask_area > 0.0
            ):
                area_ratio = max(float(mask_area), float(last_valid_mask_area)) / min(
                    float(mask_area),
                    float(last_valid_mask_area),
                )
                if area_ratio > max_mask_area_ratio_change:
                    accepted = False
                    rejection_reason = "mask_area_drift"
                    setattr(candidate_spot, "cutie_filter_rejected", True)
                    setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_aspect_ratio_change > 1.0
                and last_valid_bbox_aspect_ratio is not None
                and np.isfinite(last_valid_bbox_aspect_ratio)
                and last_valid_bbox_aspect_ratio > 0.0
                and bbox_aspect_ratio is not None
                and np.isfinite(bbox_aspect_ratio)
                and bbox_aspect_ratio > 0.0
            ):
                aspect_ratio_change = max(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                ) / min(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                )
                if aspect_ratio_change > max_bbox_aspect_ratio_change:
                    accepted = False
                    rejection_reason = "bbox_aspect_ratio_drift"
                    setattr(candidate_spot, "cutie_filter_rejected", True)
                    setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_center_jump_px > 0.0
                and last_valid_bbox_center_xy is not None
                and bbox_center_xy is not None
                and float(np.linalg.norm(bbox_center_xy - last_valid_bbox_center_xy)) > max_bbox_center_jump_px
            ):
                accepted = False
                rejection_reason = "bbox_jump"
                setattr(candidate_spot, "cutie_filter_rejected", True)
                setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    rejection_reason = "max_step_distance"
                    setattr(candidate_spot, "cutie_filter_rejected", True)
                    setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    rejection_reason = "max_radial_jump"
                    setattr(candidate_spot, "cutie_filter_rejected", True)
                    setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            rejection_reason = "direction_gating"
                            setattr(candidate_spot, "cutie_filter_rejected", True)
                            setattr(candidate_spot, "cutie_filter_rejected_reason", rejection_reason)

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
                return True
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
                forced_spot.intensity = 0.0
                setattr(forced_spot, "cutie_visibility", False)
                setattr(forced_spot, "cutie_filter_rejected", True)
                setattr(forced_spot, "cutie_filter_rejected_reason", rejection_reason)
                track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        if np.isfinite(mask_area) and mask_area > 0.0:
            last_valid_mask_area = float(mask_area)
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["last_valid_mask_area"] = last_valid_mask_area
    state["last_valid_bbox_aspect_ratio"] = last_valid_bbox_aspect_ratio
    state["last_valid_bbox_center_xy"] = last_valid_bbox_center_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    last_valid_mask_area = state["last_valid_mask_area"]
    last_valid_bbox_aspect_ratio = state["last_valid_bbox_aspect_ratio"]
    last_valid_bbox_center_xy = state["last_valid_bbox_center_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_mask_area_ratio_change = float(max(0.0, max_mask_area_ratio_change))
    max_bbox_aspect_ratio_change = float(max(0.0, max_bbox_aspect_ratio_change))
    max_bbox_center_jump_px = float(max(0.0, max_bbox_center_jump_px))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if (
                accepted
                and max_mask_area_ratio_change > 1.0
                and last_valid_mask_area is not None
                and np.isfinite(last_valid_mask_area)
                and last_valid_mask_area > 0.0
                and np.isfinite(mask_area)
                and mask_area > 0.0
            ):
                area_ratio = max(float(mask_area), float(last_valid_mask_area)) / min(
                    float(mask_area),
                    float(last_valid_mask_area),
                )
                if area_ratio > max_mask_area_ratio_change:
                    accepted = False
                    rejection_reason = "mask_area_drift"
                    setattr(candidate_spot, "xmem2_filter_rejected", True)
                    setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_aspect_ratio_change > 1.0
                and last_valid_bbox_aspect_ratio is not None
                and np.isfinite(last_valid_bbox_aspect_ratio)
                and last_valid_bbox_aspect_ratio > 0.0
                and bbox_aspect_ratio is not None
                and np.isfinite(bbox_aspect_ratio)
                and bbox_aspect_ratio > 0.0
            ):
                aspect_ratio_change = max(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                ) / min(
                    float(bbox_aspect_ratio),
                    float(last_valid_bbox_aspect_ratio),
                )
                if aspect_ratio_change > max_bbox_aspect_ratio_change:
                    accepted = False
                    rejection_reason = "bbox_aspect_ratio_drift"
                    setattr(candidate_spot, "xmem2_filter_rejected", True)
                    setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and max_bbox_center_jump_px > 0.0
                and last_valid_bbox_center_xy is not None
                and bbox_center_xy is not None
                and float(np.linalg.norm(bbox_center_xy - last_valid_bbox_center_xy)) > max_bbox_center_jump_px
            ):
                accepted = False
                rejection_reason = "bbox_jump"
                setattr(candidate_spot, "xmem2_filter_rejected", True)
                setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    rejection_reason = "max_step_distance"
                    setattr(candidate_spot, "xmem2_filter_rejected", True)
                    setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    rejection_reason = "max_radial_jump"
                    setattr(candidate_spot, "xmem2_filter_rejected", True)
                    setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            rejection_reason = "direction_gating"
                            setattr(candidate_spot, "xmem2_filter_rejected", True)
                            setattr(candidate_spot, "xmem2_filter_rejected_reason", rejection_reason)

        if keep_invisible_as_gaps:
            if not accepted:
                invisible_streak += 1
                if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                    state["terminated"] = True
                    return True
                continue

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
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
            forced_spot.intensity = 0.0
            setattr(forced_spot, "xmem2_visibility", False)
            setattr(forced_spot, "xmem2_filter_rejected", True)
            setattr(forced_spot, "xmem2_filter_rejected_reason", rejection_reason)
            track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        if np.isfinite(mask_area) and mask_area > 0.0:
            last_valid_mask_area = float(mask_area)
        if bbox_aspect_ratio is not None and np.isfinite(bbox_aspect_ratio) and bbox_aspect_ratio > 0.0:
            last_valid_bbox_aspect_ratio = float(bbox_aspect_ratio)
        if bbox_center_xy is not None:
            last_valid_bbox_center_xy = np.asarray(bbox_center_xy, dtype=np.float32)
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["last_valid_mask_area"] = last_valid_mask_area
    state["last_valid_bbox_aspect_ratio"] = last_valid_bbox_aspect_ratio
    state["last_valid_bbox_center_xy"] = last_valid_bbox_center_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
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
    last_valid_xy = state["last_valid_xy"]
    invisible_streak = int(state["invisible_streak"])
    terminated = bool(state["terminated"])
    max_step_distance_px = float(max(0.0, max_step_distance_px))
    max_reacquisition_gap = int(max(0, max_reacquisition_gap))
    terminate_after_invisible_frames = int(max(0, terminate_after_invisible_frames))
    max_radial_jump_px = float(max(0.0, max_radial_jump_px))
    lateral_angle_tolerance_deg = float(max(0.0, lateral_angle_tolerance_deg))
    min_directional_motion_px = float(max(0.0, min_directional_motion_px))
    direction_mode = str(direction_mode or "none").strip().lower()

    if terminated:
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

        candidate_spot: SpotROI | None = None
        rejection_reason = "not_visible"
        accepted = visible
        if accepted:
            if max_reacquisition_gap > 0 and invisible_streak > max_reacquisition_gap:
                state["terminated"] = True
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

            if max_step_distance_px > 0.0 and last_valid_xy is not None:
                candidate_xy = np.array(
                    [float(candidate_spot.center_x), float(candidate_spot.center_y)],
                    dtype=np.float32,
                )
                if float(np.linalg.norm(candidate_xy - last_valid_xy)) > max_step_distance_px:
                    accepted = False
                    rejection_reason = "max_step_distance"
                    setattr(candidate_spot, "tomp_filter_rejected", True)
                    setattr(candidate_spot, "tomp_filter_rejected_reason", rejection_reason)

            if (
                accepted
                and direction_center_xy is not None
                and last_valid_xy is not None
                and direction_mode != "none"
            ):
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
                    accepted = False
                    rejection_reason = "max_radial_jump"
                    setattr(candidate_spot, "tomp_filter_rejected", True)
                    setattr(candidate_spot, "tomp_filter_rejected_reason", rejection_reason)

                if accepted and step_norm >= min_directional_motion_px:
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

                    if expected_sign is not None:
                        expected_vec = radial_unit * float(expected_sign)
                        angle_deg = angular_offset_deg(expected_vec, delta)
                        if lateral_angle_tolerance_deg > 0.0 and angle_deg > lateral_angle_tolerance_deg:
                            accepted = False
                            rejection_reason = "direction_gating"
                            setattr(candidate_spot, "tomp_filter_rejected", True)
                            setattr(candidate_spot, "tomp_filter_rejected_reason", rejection_reason)

        if not accepted:
            invisible_streak += 1
            if terminate_after_invisible_frames > 0 and invisible_streak > terminate_after_invisible_frames:
                state["terminated"] = True
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
            forced_spot.intensity = 0.0
            setattr(forced_spot, "tomp_visibility", False)
            setattr(forced_spot, "tomp_filter_rejected", True)
            setattr(forced_spot, "tomp_filter_rejected_reason", rejection_reason)
            track.spots_by_image_id[frame_id] = forced_spot
            continue

        track.spots_by_image_id[frame_id] = candidate_spot
        last_valid_xy = np.array(
            [float(candidate_spot.center_x), float(candidate_spot.center_y)],
            dtype=np.float32,
        )
        invisible_streak = 0

    state["last_valid_xy"] = last_valid_xy
    state["invisible_streak"] = invisible_streak
    state["terminated"] = terminated
    return True


def _ncc_around(img_a: np.ndarray, img_b: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray, radius: int) -> float:
    x_a, y_a = int(round(pt_a[0])), int(round(pt_a[1]))
    x_b, y_b = int(round(pt_b[0])), int(round(pt_b[1]))
    if (
        x_a - radius < 0
        or y_a - radius < 0
        or x_a + radius >= img_a.shape[1]
        or y_a + radius >= img_a.shape[0]
    ):
        return -1.0
    if (
        x_b - radius < 0
        or y_b - radius < 0
        or x_b + radius >= img_b.shape[1]
        or y_b + radius >= img_b.shape[0]
    ):
        return -1.0
    patch_a = img_a[y_a - radius : y_a + radius + 1, x_a - radius : x_a + radius + 1].astype(np.float32)
    patch_b = img_b[y_b - radius : y_b + radius + 1, x_b - radius : x_b + radius + 1].astype(np.float32)
    patch_a -= patch_a.mean()
    patch_b -= patch_b.mean()
    denom = (np.linalg.norm(patch_a) * np.linalg.norm(patch_b)) + 1e-6
    if denom == 0:
        return -1.0
    return float((patch_a * patch_b).sum() / denom)


def refine_center_by_centroid(
    image: np.ndarray,
    approx_xy: np.ndarray,
    patch_radius: float,
    rel_threshold: float = 0.4,
    max_expand: int = 1,
) -> np.ndarray:
    H, W = image.shape[:2]
    cx, cy = map(int, np.round(approx_xy))
    r = max(1, int(patch_radius))
    for _ in range(max_expand + 1):
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        patch = image[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return approx_xy
        m = patch.max()
        thr = max(1.0, rel_threshold * m)
        mask = (patch >= thr).astype(np.float32)
        wsum = mask.sum()
        if wsum < 10:
            _, mask = cv2.threshold(patch.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            wsum = mask.sum()
            if wsum < 10:
                r = int(r * 1.5)
                continue
        ys, xs = np.nonzero(mask)
        vals = patch[ys, xs]
        w = vals / (vals.sum() + 1e-6)
        cx_loc = (xs * w).sum()
        cy_loc = (ys * w).sum()
        new_x, new_y = x0 + cx_loc, y0 + cy_loc
        edge_margin = 2
        if (
            abs(cx_loc - (x1 - x0) / 2) > (r - edge_margin)
            or abs(cy_loc - (y1 - y0) / 2) > (r - edge_margin)
        ):
            r = int(r * 1.5)
            continue
        return np.array([new_x, new_y], dtype=np.float32)
    return approx_xy


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
                    finished.append(Track(track_id=tid, spots_by_image_id=data["spots"]))
                to_remove.append(tid)
        for tid in to_remove:
            active_tracks.pop(tid, None)

    # Flush remaining active tracks
    for tid, data in active_tracks.items():
        if len(data["spots"]) >= min_len:
            finished.append(Track(track_id=tid, spots_by_image_id=data["spots"]))

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
                    finished.append(Track(track_id=tid, spots_by_image_id=data["spots"]))
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
            finished.append(Track(track_id=tid, spots_by_image_id=data["spots"]))

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
    tracks_map = {spot.id: Track(track_id=spot.id, spots_by_image_id={reference_frame_id: spot}) for spot in initial_spots}
    
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
            new_spot = SpotROI(id=track_id, roi_rect=new_rect, center_x=x, center_y=y, intensity=0, source='optical_flow')
            tracks_map[track_id].spots_by_image_id[frame_id] = new_spot
            
        # Prepare for the next iteration.
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        good_track_ids = surviving_track_ids
    
    print("Tracking finished.")
    if progress_callback: progress_callback(100)
    
    # Return the final list of tracks.
    return list(tracks_map.values())


def track_spots_tapir_offline(
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
    missing_polygon_seed_ids = _collect_missing_polygon_seed_spot_ids(ref_file.spots)
    if missing_polygon_seed_ids:
        raise ValueError(
            "Offline Cutie requires a polygon seed for every reference spot. "
            f"Missing polygon for spot IDs: {', '.join(str(v) for v in missing_polygon_seed_ids)}."
        )

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
            track = Track(track_id=reference_spot.id, spots_by_image_id={reference_frame_id: reference_spot})
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


def track_spots_locotrack_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_cotracker3_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_pipspp_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_mftiq_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_sam2_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_sam2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_sam2_window_overlap(
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
            query_box_rows: List[List[float]] = []
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


def track_spots_tapnext_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_trackonr_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_tapir_online(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
            track = Track(track_id=reference_spot.id, spots_by_image_id={reference_frame_id: reference_spot})
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


def retrack_single_spot_tapir_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start TAPIR re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_rect = start_spot.roi_rect
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_tapir_online(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start BootsTAPIR Online re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    if not polygon_points_are_valid(start_spot.polygon_seed_points):
        raise ValueError(
            f"Cutie re-tracking requires a polygon seed on the start spot {start_spot.id}."
        )
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_locotrack_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start LocoTrack re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_cotracker3_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start CoTracker3 re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_tapnext_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start TAPNext re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_trackonr_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start Track-On-R re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_pipspp_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start PIPs++ re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def retrack_single_spot_mftiq_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start MFTIQ re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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


def track_spots_cutie_offline(
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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_cutie_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_cutie_window_overlap(
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
            query_mask_rows: List[np.ndarray] = []
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
                query_mask_rows.append(
                    _spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
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

            covered_until = max(covered_until, end_idx - 1)

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_xmem2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_xmem2_window_overlap(
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
            query_mask_rows: List[np.ndarray] = []
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
                query_mask_rows.append(
                    _spot_polygon_seed_mask(seed_spot, window_frames[seed_local_idx].shape)
                )
                query_specs.append((reference_spot, seed_local_idx))

            if not query_rows:
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
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

            covered_until = max(covered_until, end_idx - 1)

        converted_tracks = list(track_map.values())

    if progress_callback:
        progress_callback(100)
    return converted_tracks


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
    checkpoint_path = params.get("checkpoint_path")
    if model_variant_key == "tamos_resnet50":
        tamos_models_dir = Path(params.get("tamos_models_dir", DEFAULT_TAMOS_MODELS_DIR))
        if not checkpoint_path:
            detected_checkpoints = discover_tamos_checkpoints(
                tamos_models_dir,
                repo_path=repo_path,
            )
            selected_checkpoint = select_default_tamos_checkpoint(detected_checkpoints)
            checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
        if not checkpoint_path:
            raise ValueError(
                "No TaMOs checkpoint selected. Place `tamos_resnet50.pth.tar` in "
                "tracking_models/tamos or pytracking/networks and pass checkpoint_path in params."
            )
    else:
        tomp_models_dir = Path(params.get("tomp_models_dir", DEFAULT_TOMP_MODELS_DIR))
        if not checkpoint_path:
            detected_checkpoints = discover_tomp_checkpoints(
                tomp_models_dir,
                repo_path=repo_path,
            )
            selected_checkpoint = select_default_tomp_checkpoint(detected_checkpoints)
            checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
        if not checkpoint_path:
            raise ValueError(
                "No ToMP checkpoint selected. Place `tomp50.pth.tar` in "
                "tracking_models/tomp or pytracking/networks and pass checkpoint_path in params."
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

    tomp_settings = TompRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_variant=model_variant_key,
        device=params.get("device", "auto"),
        representative_point_mode=params.get("representative_point_mode", "bbox_center"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 0.0)),
        use_presence_score_gating=bool(params.get("use_presence_score_gating", False)),
        presence_threshold=float(params.get("presence_threshold", 0.0) or 0.0),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tomp_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            ref_image = load_tiff(ref_file.filepath)
            fallback_shape = None
            if ref_image is not None and np.asarray(ref_image).size:
                fallback_shape = (int(ref_image.shape[0]), int(ref_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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
            state = {
                "last_valid_xy": np.array(
                    [float(reference_spot.center_x), float(reference_spot.center_y)],
                    dtype=np.float32,
                ),
                "invisible_streak": 0,
                "terminated": False,
                "expected_direction_sign": None,
            }
            ok = _append_tomp_sequence_to_track(
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
        overlap = _resolve_tomp_window_overlap(
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
            query_box_rows: List[List[float]] = []
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
                covered_until = max(covered_until, end_idx - 1)
                continue

            if progress_callback:
                base = 10.0 + (window_idx / max(1, len(window_starts))) * 75.0
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
                ok = _append_tomp_sequence_to_track(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start ToMP / TaMOs re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

    start_variant = (
        params.get("model_variant")
        or getattr(start_spot, "tomp_model_variant", None)
        or ("tamos_resnet50" if str(getattr(start_spot, "source", "")).strip().lower() == "tamos_offline" else "tomp50")
    )
    model_variant_key = _normalize_tomp_model_variant_key(start_variant)
    repo_path = (
        params.get("repo_path")
        or params.get("tomp_repo_path")
        or params.get("pytracking_repo_path")
    )
    checkpoint_path = params.get("checkpoint_path")
    if model_variant_key == "tamos_resnet50":
        tamos_models_dir = Path(params.get("tamos_models_dir", DEFAULT_TAMOS_MODELS_DIR))
        if not checkpoint_path:
            detected_checkpoints = discover_tamos_checkpoints(
                tamos_models_dir,
                repo_path=repo_path,
            )
            selected_checkpoint = select_default_tamos_checkpoint(detected_checkpoints)
            checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
        if not checkpoint_path:
            raise ValueError(
                "No TaMOs checkpoint selected. Place `tamos_resnet50.pth.tar` in "
                "tracking_models/tamos or pytracking/networks and pass checkpoint_path in params."
            )
    else:
        tomp_models_dir = Path(params.get("tomp_models_dir", DEFAULT_TOMP_MODELS_DIR))
        if not checkpoint_path:
            detected_checkpoints = discover_tomp_checkpoints(
                tomp_models_dir,
                repo_path=repo_path,
            )
            selected_checkpoint = select_default_tomp_checkpoint(detected_checkpoints)
            checkpoint_path = str(selected_checkpoint) if selected_checkpoint is not None else None
        if not checkpoint_path:
            raise ValueError(
                "No ToMP checkpoint selected. Place `tomp50.pth.tar` in "
                "tracking_models/tomp or pytracking/networks and pass checkpoint_path in params."
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

    if progress_callback:
        progress_callback(5)
    if cancel_event and cancel_event.is_set():
        return None

    tomp_settings = TompRuntimeSettings(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        python_executable=params.get("python_executable"),
        backend_mode=params.get("backend_mode", "subprocess"),
        model_variant=model_variant_key,
        device=params.get("device", "auto"),
        representative_point_mode=params.get("representative_point_mode", "bbox_center"),
        inference_resolution=params.get("inference_resolution"),
        initial_bbox_padding_px=float(params.get("initial_bbox_padding_px", 0.0)),
        use_presence_score_gating=bool(params.get("use_presence_score_gating", False)),
        presence_threshold=float(params.get("presence_threshold", 0.0) or 0.0),
        subprocess_timeout_sec=float(params.get("subprocess_timeout_sec", 1800.0)),
    )
    roi_size = int(params.get("roi_size", 32))
    keep_invisible_as_gaps = bool(params.get("keep_invisible_points_as_gaps", True))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_tomp_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    max_step_distance_px = float(params.get("max_step_distance_px", 0.0) or 0.0)
    max_reacquisition_gap = int(params.get("max_reacquisition_gap", 0) or 0)
    terminate_after_invisible_frames = int(params.get("terminate_after_invisible_frames", 0) or 0)
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            start_image = load_tiff(start_file.filepath) if start_file is not None else None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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

        ok = _append_tomp_sequence_to_track(
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
        overlap = _resolve_tomp_window_overlap(
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

            ok = _append_tomp_sequence_to_track(
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


def retrack_single_spot_sam2_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start SAM 2 re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_sam2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_sam2_window_overlap(
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

            covered_until = max(covered_until, end_idx - 1)

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
    """
    Re-track a single existing track from a chosen frame using offline Cutie.

    The track is trimmed after `start_frame_id`, then Cutie is rerun from that
    point to the end of the sequence using the spot center and ROI on the start
    frame as the seed prompt.
    """
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start Cutie re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    start_frame_index = sorted_frame_ids.index(start_frame_id)
    subset_frame_ids = sorted_frame_ids[start_frame_index:]
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_cutie_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            if start_file is not None:
                start_image = load_tiff(start_file.filepath)
            else:
                start_image = None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_cutie_window_overlap(
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def _collect_xmem2_permanent_reference_spots(
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


def retrack_single_spot_xmem2_offline(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(
            f"Cannot start XMem2 re-tracking. Track {track_to_update.track_id} "
            f"has no point on frame {start_frame_id}."
        )
    if start_frame_id not in sorted_frame_ids:
        raise ValueError("Start frame is missing from the provided frame list.")

    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]

    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    if not polygon_points_are_valid(start_spot.polygon_seed_points):
        raise ValueError(
            f"XMem2 re-tracking requires a polygon seed on the start spot {start_spot.id}."
        )
    start_frame_index = sorted_frame_ids.index(start_frame_id)
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
    if len(subset_frame_ids) < 2:
        if progress_callback:
            progress_callback(100)
        return track_to_update

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
    direction_mode = str(params.get("direction_mode", "none")).strip().lower()
    center_mode = str(params.get("center_mode", "auto")).strip().lower()
    manual_center = params.get("manual_center")
    lateral_angle_tolerance_deg = float(params.get("lateral_angle_tolerance_deg", 20.0))
    max_radial_jump_px = float(params.get("max_radial_jump_px", 0.0))
    min_directional_motion_px = float(params.get("min_directional_motion_px", 1.0))
    roi_size = int(params.get("roi_size", 32))
    window_size = int(params.get("window_size", 64))
    window_overlap = _resolve_xmem2_window_overlap(
        len(subset_frame_ids),
        window_size,
        int(params.get("window_overlap", 8)),
    )
    force_full_sequence = len(reference_entries) > 1
    start_local_idx = int(start_frame_index - subset_start_index + 1)

    direction_center_xy: Tuple[float, float] | None = None
    if direction_mode != "none":
        if center_mode == "manual" and manual_center is not None:
            try:
                direction_center_xy = (float(manual_center[0]), float(manual_center[1]))
            except Exception:
                direction_center_xy = None
        if direction_center_xy is None:
            start_file = all_files_map.get(start_frame_id)
            start_image = load_tiff(start_file.filepath) if start_file is not None else None
            fallback_shape = None
            if start_image is not None and np.asarray(start_image).size:
                fallback_shape = (int(start_image.shape[0]), int(start_image.shape[1]))
            estimated_center = estimate_pattern_center_from_detections(
                all_files_map,
                frame_ids=subset_frame_ids,
                fallback_shape=fallback_shape,
            )
            if estimated_center is not None:
                direction_center_xy = (
                    float(estimated_center["center_x"]),
                    float(estimated_center["center_y"]),
                )

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

        window_starts = _compute_point_tracker_window_starts(
            len(subset_frame_ids),
            window_size,
            window_overlap,
        )
        overlap = _resolve_xmem2_window_overlap(
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

            covered_until = max(covered_until, end_idx - 1)

    if progress_callback:
        progress_callback(100)
    return track_to_update


def retrack_single_spot_optical_flow(
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
    if start_frame_id not in track_to_update.spots_by_image_id:
        raise ValueError(f"Cannot start re-tracking. Track {track_to_update.track_id} has no point on frame {start_frame_id}.")

    # Trim the track by discarding every point that comes after the start frame
    frames_to_remove = [fid for fid in track_to_update.spots_by_image_id if fid > start_frame_id]
    for fid in frames_to_remove:
        del track_to_update.spots_by_image_id[fid]
    
    start_spot = track_to_update.spots_by_image_id[start_frame_id]
    p0 = np.array([[[start_spot.center_x, start_spot.center_y]]], dtype=np.float32)

    start_frame_index = sorted_frame_ids.index(start_frame_id)
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
            new_spot = SpotROI(
                id=start_spot.id, # Preserve the original identifier for consistency
                roi_rect=new_rect,
                center_x=x, center_y=y, intensity=0, source='re-tracked'
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
        track = Track(track_id=spot.id, spots_by_image_id={})
        for rel_frame_idx, x, y in positions:
            frame_id = subset_frame_ids[rel_frame_idx]
            if rel_frame_idx == 0:
                track.spots_by_image_id[frame_id] = spot
                continue

            rect = QRect(int(x - roi_half), int(y - roi_half), roi_size, roi_size)
            track.spots_by_image_id[frame_id] = SpotROI(
                id=spot.id,
                roi_rect=rect,
                center_x=x,
                center_y=y,
                intensity=0,
                source="optical_flow_skip_anchor",
            )
        converted_tracks.append(track)

    if progress_callback:
        progress_callback(100)
    return converted_tracks
