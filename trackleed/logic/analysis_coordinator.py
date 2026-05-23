# stal/logic/analysis_coordinator.py

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from threading import Event

from skimage.color import rgb2gray
from skimage.util import img_as_float

from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.peak_analyzer import (
    fit_spot_to_gaussian,
    find_pattern_center,
    fit_circle_center,
    fit_ellipse_center,
)

def run_full_analysis(
    tracks_to_analyze: List[Track],
    analysis_config: Dict[int, Dict],
    all_files_map: Dict[int, LeedFile],
    background_file: Optional[LeedFile],
    progress_callback: Callable[[int], None],
    center_mode: str = "per_frame",
    manual_ellipse: Optional[Dict[str, float]] = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Coordinates the entire quantitative analysis pipeline within a single main loop.
    """
    print(f"[Analysis] Starting run_full_analysis | center_mode={center_mode} | tracks={len(tracks_to_analyze)}")

    def is_cancelled() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    cancel_notice_emitted = False

    def cancelled(stage: str) -> Optional[List[Track]]:
        nonlocal cancel_notice_emitted
        if not cancel_notice_emitted:
            print(f"[Analysis] Cancel detected at: {stage}. Stopping analysis.")
            cancel_notice_emitted = True
        return None

    if is_cancelled():
        return cancelled("initialization")

    for track in tracks_to_analyze:
        if is_cancelled():
            return cancelled(f"track init (track_id={track.track_id})")
        track.analysis_results = {}
    bg_image_data = None
    if background_file:
        if is_cancelled():
            return cancelled("before background load")
        temp_bg_data = load_tiff(background_file.filepath)
        if temp_bg_data is not None:
            if temp_bg_data.ndim == 3:
                bg_image_data = rgb2gray(temp_bg_data)
            else:
                bg_image_data = img_as_float(temp_bg_data)
        print("[Analysis] Background loaded for subtraction.")

    center_mode = (center_mode or "per_frame").lower()
    manual_ellipse_mode = center_mode == "manual_ellipse"
    ellipse_fit_mode = center_mode == "ellipse_fit"
    circle_bbox_fallback_mode = center_mode == "circle_fit_bbox_fallback"
    ellipse_info = _normalize_manual_ellipse(manual_ellipse) if manual_ellipse_mode else None
    if manual_ellipse_mode and ellipse_info is None:
        raise ValueError("center_mode='manual_ellipse' requires a valid manual_ellipse definition.")

    per_frame_mode = center_mode in {
        "per_frame",
        "circle_fit",
        "circle_fit_bbox_fallback",
        "manual_ellipse",
        "ellipse_fit",
    }
    temporal_circle_mode = center_mode == "circle_fit_temporal"
    all_frame_ids = sorted(list({fid for t in tracks_to_analyze for fid in t.spots_by_image_id}))
    if not all_frame_ids: return tracks_to_analyze
    _reset_per_frame_analysis_state(all_files_map, all_frame_ids)

    per_frame_centers: Dict[int, Dict] = {}
    centers_for_median: List[Dict[str, float]] = []
    total_frames = len(all_frame_ids)
    log_every = max(1, total_frames // 10)

    for i, frame_id in enumerate(all_frame_ids):
        if is_cancelled():
            return cancelled(f"pass1 frame loop start (frame_id={frame_id})")
        full_image = load_tiff(all_files_map[frame_id].filepath)
        if is_cancelled():
            return cancelled(f"after image load (frame_id={frame_id})")
        if full_image is None or full_image.size == 0:
            continue
        
        spot_positions_on_this_frame = []

        for track in tracks_to_analyze:
            if is_cancelled():
                return cancelled(f"pass1 track loop (frame_id={frame_id})")
            if frame_id not in track.spots_by_image_id:
                continue
            if not hasattr(track, 'analysis_results'):
                track.analysis_results = {}
            
            spot = track.spots_by_image_id[frame_id]
            rect = spot.roi_rect
            roi_data = full_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()].copy()

            track_config = analysis_config.get(track.track_id, {})
            should_subtract_bg = track_config.get('subtract_bg', False)
            final_roi_to_fit = img_as_float(roi_data)

            if should_subtract_bg and bg_image_data is not None:
                bg_roi_data = bg_image_data[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
                if bg_roi_data.shape == final_roi_to_fit.shape:
                    final_roi_to_fit = final_roi_to_fit - bg_roi_data
                    final_roi_to_fit[final_roi_to_fit < 0] = 0
            
            fit_results = fit_spot_to_gaussian(final_roi_to_fit)
            if is_cancelled():
                return cancelled(f"after gaussian fit (frame_id={frame_id}, track_id={track.track_id})")
            if not _is_fit_valid(fit_results, final_roi_to_fit.shape):
                fit_results = None # Uznaj dopasowanie za nieudane
            track.analysis_results[frame_id] = fit_results
            
            if fit_results:
                global_x = rect.x() + fit_results['center_x']
                global_y = rect.y() + fit_results['center_y']
                fit_results['global_center_x'] = global_x
                fit_results['global_center_y'] = global_y
                spot_positions_on_this_frame.append({'center_x': global_x, 'center_y': global_y})
            elif circle_bbox_fallback_mode:
                spot_positions_on_this_frame.append(_bbox_center_position(rect))
        
        center_info = _estimate_center_for_frame(spot_positions_on_this_frame, center_mode)
        if manual_ellipse_mode:
            center_info = {
                "center_x": ellipse_info["center_x"],
                "center_y": ellipse_info["center_y"],
                "error_x": 0.0,
                "error_y": 0.0,
            }
        if center_info:
            per_frame_centers[frame_id] = center_info
            centers_for_median.append(center_info)
            if per_frame_mode:
                all_files_map[frame_id].pattern_center = center_info

        if per_frame_mode and center_info:
            ellipse_payload = center_info if ellipse_fit_mode else (ellipse_info if manual_ellipse_mode else None)
            _apply_center_to_frame(
                tracks_to_analyze,
                all_files_map,
                frame_id,
                center_info,
                manual_ellipse=ellipse_payload,
            )

        if per_frame_mode:
            _update_average_radius(all_files_map, tracks_to_analyze, frame_id)
            progress_callback(int((i + 1) * 100 / len(all_frame_ids)))
        elif center_mode in {"global_median", "circle_fit_temporal"}:
            progress_callback(int((i + 1) * 50 / len(all_frame_ids)))

        if i == 0 or (i + 1) % log_every == 0 or i == total_frames - 1:
            print(f"[Analysis] Pass1 progress: {i + 1}/{total_frames} frames (frame_id={frame_id})")

    if center_mode == "global_median":
        if centers_for_median:
            center_info = {
                "center_x": float(np.median([c["center_x"] for c in centers_for_median])),
                "center_y": float(np.median([c["center_y"] for c in centers_for_median])),
                "error_x": float(np.median([c.get("error_x", 0.0) for c in centers_for_median])),
                "error_y": float(np.median([c.get("error_y", 0.0) for c in centers_for_median])),
            }
        else:
            center_info = None

        for i, frame_id in enumerate(all_frame_ids):
            if is_cancelled():
                return cancelled(f"pass2 global_median loop (frame_id={frame_id})")
            if center_info:
                all_files_map[frame_id].pattern_center = center_info
                _apply_center_to_frame(tracks_to_analyze, all_files_map, frame_id, center_info)
            _update_average_radius(all_files_map, tracks_to_analyze, frame_id)
            progress_callback(50 + int((i + 1) * 50 / len(all_frame_ids)))
            if i == 0 or (i + 1) % log_every == 0 or i == total_frames - 1:
                print(f"[Analysis] Pass2 (global median) progress: {i + 1}/{total_frames} frames (frame_id={frame_id})")
    elif temporal_circle_mode:
        smoothed_centers = _smooth_centers_temporally(all_frame_ids, per_frame_centers)
        for i, frame_id in enumerate(all_frame_ids):
            if is_cancelled():
                return cancelled(f"pass2 temporal smoothing loop (frame_id={frame_id})")
            center_info = smoothed_centers.get(frame_id)
            if center_info:
                all_files_map[frame_id].pattern_center = center_info
                _apply_center_to_frame(tracks_to_analyze, all_files_map, frame_id, center_info)
            _update_average_radius(all_files_map, tracks_to_analyze, frame_id)
            progress_callback(50 + int((i + 1) * 50 / len(all_frame_ids)))
            if i == 0 or (i + 1) % log_every == 0 or i == total_frames - 1:
                print(f"[Analysis] Pass2 (temporal center) progress: {i + 1}/{total_frames} frames (frame_id={frame_id})")

    print("[Analysis] Analysis completed successfully.")
    return tracks_to_analyze


def _reset_per_frame_analysis_state(
    all_files_map: Dict[int, LeedFile],
    frame_ids: List[int],
) -> None:
    """Clear frame-level analysis fields before a new analysis run."""
    for frame_id in frame_ids:
        leed_file = all_files_map.get(frame_id)
        if leed_file is None:
            continue
        leed_file.pattern_center = None
        leed_file.post_analysis_results = {}


def _bbox_center_position(rect) -> Dict[str, float]:
    return {
        "center_x": float(rect.x()) + float(rect.width()) / 2.0,
        "center_y": float(rect.y()) + float(rect.height()) / 2.0,
    }


def _estimate_center_for_frame(spot_positions_on_this_frame: List[Dict[str, float]], center_mode: str) -> Optional[Dict[str, float]]:
    """Estimate per-frame center using the selected center mode."""
    if not spot_positions_on_this_frame:
        return None
    if center_mode == "manual_ellipse":
        return None
    if center_mode == "ellipse_fit":
        return fit_ellipse_center(spot_positions_on_this_frame)
    if center_mode == "circle_fit_bbox_fallback":
        return fit_circle_center(spot_positions_on_this_frame)
    if center_mode in {"circle_fit", "circle_fit_temporal"}:
        return fit_circle_center(spot_positions_on_this_frame)
    return find_pattern_center(spot_positions_on_this_frame)


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs and keep endpoints clamped to nearest valid value."""
    if values.size == 0:
        return values.copy()
    mask = np.isfinite(values)
    if not np.any(mask):
        return values.copy()
    if np.sum(mask) == 1:
        return np.full_like(values, values[mask][0], dtype=np.float64)

    idx = np.arange(values.size, dtype=np.float64)
    out = values.astype(np.float64, copy=True)
    out[~mask] = np.interp(idx[~mask], idx[mask], out[mask])
    return out


def _rolling_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Small-window median smoother to reject local outliers."""
    if values.size == 0:
        return values.copy()
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2

    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def _ema(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Exponential moving average for temporal center stabilization."""
    if values.size == 0:
        return values.copy()
    a = float(min(max(alpha, 0.01), 1.0))
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = a * values[i] + (1.0 - a) * out[i - 1]
    return out


def _nan_median_or_default(values: np.ndarray, default: float) -> float:
    """Return median of finite values or fallback default."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def _smooth_centers_temporally(all_frame_ids: List[int], raw_centers: Dict[int, Dict]) -> Dict[int, Dict[str, float]]:
    """
    Stabilize per-frame circle-fit centers over time.

    Pipeline:
    1) interpolate missing center estimates,
    2) rolling median filter,
    3) EMA smoothing,
    4) clamp frame-to-frame jumps to robustly reject residual outliers.
    """
    if not all_frame_ids or not raw_centers:
        return {}

    n_frames = len(all_frame_ids)
    center_x = np.full(n_frames, np.nan, dtype=np.float64)
    center_y = np.full(n_frames, np.nan, dtype=np.float64)
    error_x = np.full(n_frames, np.nan, dtype=np.float64)
    error_y = np.full(n_frames, np.nan, dtype=np.float64)
    circle_radius = np.full(n_frames, np.nan, dtype=np.float64)

    fid_to_idx = {fid: i for i, fid in enumerate(all_frame_ids)}
    for fid, info in raw_centers.items():
        idx = fid_to_idx.get(fid)
        if idx is None:
            continue
        center_x[idx] = float(info.get("center_x", np.nan))
        center_y[idx] = float(info.get("center_y", np.nan))
        error_x[idx] = float(info.get("error_x", np.nan))
        error_y[idx] = float(info.get("error_y", np.nan))
        circle_radius[idx] = float(info.get("circle_radius", np.nan))

    valid = np.isfinite(center_x) & np.isfinite(center_y)
    if not np.any(valid):
        return {}

    cx = _interpolate_nans(center_x)
    cy = _interpolate_nans(center_y)
    radius = _interpolate_nans(circle_radius)

    # Keep center drift smooth but responsive.
    cx = _ema(_rolling_median(cx, window=5), alpha=0.25)
    cy = _ema(_rolling_median(cy, window=5), alpha=0.25)
    radius = _ema(_rolling_median(radius, window=5), alpha=0.25)

    # Robust step clamp based on raw (interpolated) trajectory statistics.
    step = np.sqrt(np.diff(_interpolate_nans(center_x)) ** 2 + np.diff(_interpolate_nans(center_y)) ** 2)
    step_med = _nan_median_or_default(step, default=2.0)
    step_mad = _nan_median_or_default(np.abs(step - step_med), default=0.5)
    max_step = float(min(25.0, max(1.5, step_med + 4.0 * step_mad + 2.0)))

    for i in range(1, n_frames):
        dx = cx[i] - cx[i - 1]
        dy = cy[i] - cy[i - 1]
        dist = float(np.hypot(dx, dy))
        if dist > max_step and dist > 1e-9:
            scale = max_step / dist
            cx[i] = cx[i - 1] + dx * scale
            cy[i] = cy[i - 1] + dy * scale

    default_err_x = _nan_median_or_default(error_x, default=1.0)
    default_err_y = _nan_median_or_default(error_y, default=1.0)

    smoothed: Dict[int, Dict[str, float]] = {}
    for i, fid in enumerate(all_frame_ids):
        ex = float(error_x[i]) if np.isfinite(error_x[i]) else default_err_x
        ey = float(error_y[i]) if np.isfinite(error_y[i]) else default_err_y

        # If raw center exists, include raw-vs-smoothed discrepancy in uncertainty.
        if np.isfinite(center_x[i]) and np.isfinite(center_y[i]):
            ex = max(ex, abs(float(center_x[i] - cx[i])))
            ey = max(ey, abs(float(center_y[i] - cy[i])))

        smoothed[fid] = {
            "center_x": float(cx[i]),
            "center_y": float(cy[i]),
            "error_x": float(ex),
            "error_y": float(ey),
        }
        if np.isfinite(radius[i]) and radius[i] > 0:
            smoothed[fid]["circle_radius"] = float(radius[i])

    return smoothed


def _apply_center_to_frame(
    tracks_to_analyze: List[Track],
    all_files_map: Dict[int, LeedFile],
    frame_id: int,
    center_info: Dict[str, float],
    manual_ellipse: Optional[Dict[str, float]] = None,
) -> None:
    for track in tracks_to_analyze:
        results = track.analysis_results.get(frame_id)
        if not results:
            continue

        if manual_ellipse is not None:
            radius, radius_error = _compute_ellipse_corrected_radius(results, center_info, manual_ellipse)
            results['radius'] = radius
            results['radius_error'] = radius_error
            continue

        dx = results['global_center_x'] - center_info['center_x']
        dy = results['global_center_y'] - center_info['center_y']
        results['radius'] = np.sqrt(dx**2 + dy**2)

        err_x_spot = results.get('center_x_error', 0)
        err_y_spot = results.get('center_y_error', 0)
        err_x_center = center_info.get('error_x', 0)
        err_y_center = center_info.get('error_y', 0)

        var_dx = err_x_spot**2 + err_x_center**2
        var_dy = err_y_spot**2 + err_y_center**2

        if results['radius'] > 1e-6:
            results['radius_error'] = (1/results['radius']) * np.sqrt(dx**2 * var_dx + dy**2 * var_dy)


def _normalize_manual_ellipse(manual_ellipse: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Normalize and validate manual ellipse payload to internal analysis fields."""
    if not manual_ellipse:
        return None
    try:
        cx = float(manual_ellipse.get("cx"))
        cy = float(manual_ellipse.get("cy"))
        rx = float(manual_ellipse.get("rx"))
        ry = float(manual_ellipse.get("ry"))
        angle_deg = float(manual_ellipse.get("angle_deg", 0.0))
    except Exception:
        return None

    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(rx) or not np.isfinite(ry):
        return None
    if rx <= 0 or ry <= 0:
        return None

    return {
        "center_x": cx,
        "center_y": cy,
        "radius_x": rx,
        "radius_y": ry,
        "angle_deg": angle_deg,
        "radius_scale_px": float(np.sqrt(rx * ry)),
    }


def _compute_ellipse_corrected_radius(
    fit_results: Dict[str, float],
    center_info: Dict[str, float],
    manual_ellipse: Dict[str, float],
) -> Tuple[float, float]:
    """
    Compute radius after affine correction implied by manual ellipse.

    The point is rotated into ellipse-aligned coordinates and normalized by
    semi-axes. The normalized radius is then scaled by sqrt(rx*ry) so values
    stay in pixel-like magnitude.
    """
    cx = float(center_info.get("center_x", manual_ellipse["center_x"]))
    cy = float(center_info.get("center_y", manual_ellipse["center_y"]))
    rx = float(manual_ellipse["radius_x"])
    ry = float(manual_ellipse["radius_y"])
    angle_deg = float(manual_ellipse.get("angle_deg", 0.0))
    scale_px = float(manual_ellipse.get("radius_scale_px", np.sqrt(rx * ry)))

    dx = float(fit_results["global_center_x"]) - cx
    dy = float(fit_results["global_center_y"]) - cy

    theta = np.deg2rad(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    inv_rx = 1.0 / max(rx, 1e-9)
    inv_ry = 1.0 / max(ry, 1e-9)
    u = xr * inv_rx
    v = yr * inv_ry
    q = float(np.sqrt(u * u + v * v))
    radius = float(scale_px * q)

    err_x_spot = float(fit_results.get("center_x_error", 0.0))
    err_y_spot = float(fit_results.get("center_y_error", 0.0))
    err_x_center = float(center_info.get("error_x", 0.0))
    err_y_center = float(center_info.get("error_y", 0.0))
    var_dx = err_x_spot**2 + err_x_center**2
    var_dy = err_y_spot**2 + err_y_center**2

    if q <= 1e-9:
        return radius, 0.0

    dqx_ddx = (u * (cos_t * inv_rx) + v * (-sin_t * inv_ry)) / q
    dqy_ddy = (u * (sin_t * inv_rx) + v * (cos_t * inv_ry)) / q
    dr_ddx = scale_px * dqx_ddx
    dr_ddy = scale_px * dqy_ddy
    radius_error = float(np.sqrt((dr_ddx**2) * var_dx + (dr_ddy**2) * var_dy))
    return radius, radius_error


def _update_average_radius(
    all_files_map: Dict[int, LeedFile],
    tracks_to_analyze: List[Track],
    frame_id: int,
) -> None:
    leed_file = all_files_map.get(frame_id)
    if leed_file is None:
        return

    radii_on_this_frame = []
    for track in tracks_to_analyze:
        results = getattr(track, "analysis_results", {}).get(frame_id)
        if not results or "radius" not in results:
            continue
        try:
            radius = float(results["radius"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(radius):
            radii_on_this_frame.append(radius)

    if len(radii_on_this_frame) > 1:
        avg_radius = np.mean(radii_on_this_frame)
        avg_radius_error = np.std(radii_on_this_frame) / np.sqrt(len(radii_on_this_frame))
        if not hasattr(leed_file, 'post_analysis_results'):
            leed_file.post_analysis_results = {}
        leed_file.post_analysis_results['avg_radius'] = avg_radius
        leed_file.post_analysis_results['avg_radius_error'] = avg_radius_error
        return

    post_analysis_results = getattr(leed_file, 'post_analysis_results', None)
    if post_analysis_results:
        post_analysis_results.pop('avg_radius', None)
        post_analysis_results.pop('avg_radius_error', None)

def _is_fit_valid(fit_results: Optional[Dict], roi_shape: Tuple) -> bool:
    """Checks if the Gaussian fit results are physically plausible."""
    if not fit_results:
        return False

    if roi_shape is None or len(roi_shape) < 2:
        return False
    height, width = roi_shape[:2]
    
    # 1. Validate goodness of fit (R-squared threshold)
    if fit_results.get('r_squared', 0) < 0.85: # Threshold can be tuned
        return False
        
    # 2. Reject very flat peaks (near-zero amplitude)
    if fit_results.get('amplitude', 0) < 0.05: # Threshold for normalized [0,1] data
        return False

    # 3. Ensure the peak center stays inside the ROI (with a small margin)
    if not (-2 < fit_results.get('center_x', 0) < width + 2):
        return False
    if not (-2 < fit_results.get('center_y', 0) < height + 2):
        return False
        
    # 4. Verify that sigmas remain within reasonable bounds
    if not (0.5 < fit_results.get('sigma_x', 0) < width):
        return False
    if not (0.5 < fit_results.get('sigma_y', 0) < height):
        return False

    return True # All checks passed

# def run_full_analysis(
#     tracks_to_analyze: List[Track],
#     analysis_config: Dict[int, Dict],
#     all_files_map: Dict[int, LeedFile],
#     background_file: Optional[LeedFile],
#     progress_callback: Callable[[int], None]
# ) -> List[Track]:
#     """
#     Coordinates the entire quantitative analysis pipeline.

#     This function iterates through all specified frames and tracks, performs a
#     2D Gaussian fit for each spot (with optional background subtraction),
#     calculates the pattern center for each frame, and populates the Track
#     objects with detailed analysis results. It is designed to be run in a
#     background thread and reports progress via a callback.

#     Args:
#         tracks_to_analyze: The list of Track objects to be processed.
#         analysis_config: A dictionary mapping track IDs to their specific
#                          configuration (e.g., {'subtract_bg': True}).
#         all_files_map: A dictionary mapping frame IDs to LeedFile objects,
#                        used for loading image data and storing results.
#         background_file: An optional LeedFile object for the background image.
#         progress_callback: A function (e.g., a signal's emit method) to
#                            report progress as an integer percentage.

#     Returns:
#         The list of Track objects, with their `analysis_results` attribute
#         populated with the fitting results.
#     """
#     # --- 1. Pre-load and Process Background Image ---
#     # Load the background image once at the beginning for efficiency.
#     bg_image_data = None
#     if background_file:
#         temp_bg_data = load_tiff(background_file.filepath)
#         if temp_bg_data is not None:
#             # Convert the background to grayscale float [0,1] if it's an RGB image.
#             if temp_bg_data.ndim == 3:
#                 bg_image_data = rgb2gray(temp_bg_data)
#             else: # If it's already grayscale, just convert to float.
#                 bg_image_data = temp_bg_data 

#     # --- 2. Main Analysis Loop ---
#     # Get a sorted list of all unique frame IDs that need to be processed.
#     all_frame_ids = sorted(list({fid for t in tracks_to_analyze for fid in t.spots_by_image_id}))
#     if not all_frame_ids: return tracks_to_analyze

#     # The main loop is frame-centric for efficiency.
#     for i, frame_id in enumerate(all_frame_ids):
#         # Load the full image for the current frame once.
#         full_image = load_tiff(all_files_map[frame_id].filepath)
#         if full_image is None or full_image.size == 0: continue
        
#         # This list will hold all spot positions for the current frame.
#         spot_positions_on_this_frame = []

#         # The inner loop processes each track that exists on the current frame.
#         for track in tracks_to_analyze:
#             if frame_id not in track.spots_by_image_id: continue

#             # Ensure the results dictionary exists for this track.
#             if not hasattr(track, 'analysis_results'): track.analysis_results = {}
            
#             spot = track.spots_by_image_id[frame_id]
#             rect = spot.roi_rect
#             # Extract the Region of Interest (ROI) from the full image.
#             roi_data = full_image[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()].copy()

#             # Get the analysis settings for this specific track from the UI.
#             track_config = analysis_config.get(track.track_id, {})
#             should_subtract_bg = track_config.get('subtract_bg', False)

#             # --- Prepare ROI for Fitting ---
#             # By default, we fit the normalized raw data.
#             final_roi_to_fit = img_as_float(roi_data)

#             # If background subtraction is enabled, perform the operation.
#             if should_subtract_bg and bg_image_data is not None:
#                 bg_roi_data = bg_image_data[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
                
#                 if bg_roi_data.shape == final_roi_to_fit.shape:
#                     # Both arrays are now on the same [0,1] float scale.
#                     final_roi_to_fit = final_roi_to_fit - bg_roi_data
#                     final_roi_to_fit[final_roi_to_fit < 0] = 0
            
#             # --- Perform Fit and Store Results ---
#             fit_results = fit_spot_to_gaussian(final_roi_to_fit)
#             # Store the entire results dictionary in the track object.
#             track.analysis_results[frame_id] = fit_results
            
#             if fit_results:
#                 # Calculate global coordinates by adding the ROI's top-left position.
#                 global_x = rect.x() + fit_results['center_x']
#                 global_y = rect.y() + fit_results['center_y']
#                 # Add global coordinates to the results dict for easy access.
#                 fit_results['global_center_x'] = global_x
#                 fit_results['global_center_y'] = global_y
#                 # Collect all spot positions on this frame to find the pattern center.
#                 spot_positions_on_this_frame.append({'center_x': global_x, 'center_y': global_y})

#         if spot_positions_on_this_frame:
#             center_info = find_pattern_center(spot_positions_on_this_frame)
#             if center_info and frame_id in all_files_map:
#                 all_files_map[frame_id].pattern_center = center_info
        
#         progress_callback(int((i + 1) * 100 / len(all_frame_ids)))

#     return tracks_to_analyze
