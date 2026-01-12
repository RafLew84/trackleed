# stal/logic/analysis_coordinator.py

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple

from skimage.color import rgb2gray
from skimage.util import img_as_float

from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian, find_pattern_center

def run_full_analysis(
    tracks_to_analyze: List[Track],
    analysis_config: Dict[int, Dict],
    all_files_map: Dict[int, LeedFile],
    background_file: Optional[LeedFile],
    progress_callback: Callable[[int], None],
    center_mode: str = "per_frame",
) -> List[Track]:
    """
    Coordinates the entire quantitative analysis pipeline within a single main loop.
    """
    for track in tracks_to_analyze:
        track.analysis_results = {}
    bg_image_data = None
    if background_file:
        temp_bg_data = load_tiff(background_file.filepath)
        if temp_bg_data is not None:
            if temp_bg_data.ndim == 3:
                bg_image_data = rgb2gray(temp_bg_data)
            else:
                bg_image_data = img_as_float(temp_bg_data)

    center_mode = (center_mode or "per_frame").lower()
    all_frame_ids = sorted(list({fid for t in tracks_to_analyze for fid in t.spots_by_image_id}))
    if not all_frame_ids: return tracks_to_analyze

    per_frame_centers: Dict[int, Dict] = {}
    centers_for_median: List[Dict[str, float]] = []

    for i, frame_id in enumerate(all_frame_ids):
        full_image = load_tiff(all_files_map[frame_id].filepath)
        if full_image is None or full_image.size == 0: continue
        
        spot_positions_on_this_frame = []

        for track in tracks_to_analyze:
            if frame_id not in track.spots_by_image_id: continue
            if not hasattr(track, 'analysis_results'): track.analysis_results = {}
            
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
            if not _is_fit_valid(fit_results, final_roi_to_fit.shape):
                fit_results = None # Uznaj dopasowanie za nieudane
            track.analysis_results[frame_id] = fit_results
            
            if fit_results:
                global_x = rect.x() + fit_results['center_x']
                global_y = rect.y() + fit_results['center_y']
                fit_results['global_center_x'] = global_x
                fit_results['global_center_y'] = global_y
                spot_positions_on_this_frame.append({'center_x': global_x, 'center_y': global_y})
        
        center_info = None
        if spot_positions_on_this_frame:
            center_info = find_pattern_center(spot_positions_on_this_frame)
            if center_info:
                per_frame_centers[frame_id] = center_info
                centers_for_median.append(center_info)
                if center_mode == "per_frame":
                    all_files_map[frame_id].pattern_center = center_info

        if center_mode == "per_frame" and center_info:
            _apply_center_to_frame(tracks_to_analyze, all_files_map, frame_id, center_info)

        if center_mode == "per_frame":
            _update_average_radius(all_files_map, tracks_to_analyze, frame_id)
            progress_callback(int((i + 1) * 100 / len(all_frame_ids)))
        elif center_mode == "global_median":
            progress_callback(int((i + 1) * 50 / len(all_frame_ids)))

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
            if center_info:
                all_files_map[frame_id].pattern_center = center_info
                _apply_center_to_frame(tracks_to_analyze, all_files_map, frame_id, center_info)
            _update_average_radius(all_files_map, tracks_to_analyze, frame_id)
            progress_callback(50 + int((i + 1) * 50 / len(all_frame_ids)))

    return tracks_to_analyze


def _apply_center_to_frame(
    tracks_to_analyze: List[Track],
    all_files_map: Dict[int, LeedFile],
    frame_id: int,
    center_info: Dict[str, float],
) -> None:
    for track in tracks_to_analyze:
        results = track.analysis_results.get(frame_id)
        if not results:
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


def _update_average_radius(
    all_files_map: Dict[int, LeedFile],
    tracks_to_analyze: List[Track],
    frame_id: int,
) -> None:
    radii_on_this_frame = [
        t.analysis_results[frame_id]['radius']
        for t in tracks_to_analyze
        if frame_id in t.analysis_results and t.analysis_results[frame_id] and 'radius' in t.analysis_results[frame_id]
    ]

    if len(radii_on_this_frame) > 1:
        avg_radius = np.mean(radii_on_this_frame)
        avg_radius_error = np.std(radii_on_this_frame) / np.sqrt(len(radii_on_this_frame))
        if not hasattr(all_files_map[frame_id], 'post_analysis_results'):
            all_files_map[frame_id].post_analysis_results = {}
        all_files_map[frame_id].post_analysis_results['avg_radius'] = avg_radius
        all_files_map[frame_id].post_analysis_results['avg_radius_error'] = avg_radius_error

def _is_fit_valid(fit_results: Optional[Dict], roi_shape: Tuple) -> bool:
    """Checks if the Gaussian fit results are physically plausible."""
    if not fit_results:
        return False
    
    height, width = roi_shape
    
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
