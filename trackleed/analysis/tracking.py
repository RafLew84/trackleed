# stal/analysis/tracking.py

from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QRect
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.morphology import white_tophat, disk
from skimage.util import img_as_ubyte

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.image_reader import load_tiff


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

# --- HUNGARIAN TRACKER ---
def track_spots_hungarian(
    all_spots_by_frame: Dict[int, List[SpotROI]], 
    max_distance: float,
    progress_callback: Callable[[int], None] | None = None
) -> List[Track]:
    """
    Tracks spots between consecutive frames using the Hungarian algorithm.

    This function is ideal for scenarios where spots are detected independently
    on each frame. It works by solving the assignment problem for each pair of
    frames, finding the optimal pairing of spots that minimizes the total
    distance.

    Args:
        all_spots_by_frame: A dictionary mapping a frame ID (int) to a list
                            of SpotROI objects detected on that frame.
        max_distance: The maximum distance (in pixels) a spot can move
                      between two consecutive frames to be considered part of
                      the same track.
        progress_callback: An optional function that can be called to report
                           tracking progress as an integer percentage (0-100).

    Returns:
        A list of Track objects. Each Track represents a single spot's
        trajectory across multiple frames.
    """
    # active_tracks: Tracks that appeared in the previous frame and can be extended.
    # completed_tracks: Tracks that have terminated (disappeared).
    active_tracks, completed_tracks, next_track_id = [], [], 0

    # Sort frame IDs to process them in chronological order.
    sorted_frame_ids = sorted(all_spots_by_frame.keys())
    if not sorted_frame_ids: return []
    
    # --- Bootstrap the process ---
    # Initialize active tracks with all spots from the very first frame.
    first_frame_id = sorted_frame_ids[0]
    for spot in all_spots_by_frame[first_frame_id]:
        active_tracks.append(Track(track_id=next_track_id, spots_by_image_id={first_frame_id: spot}))
        next_track_id += 1

    # Main loop: iterate through consecutive pairs of frames.
    num_frames_to_process = len(sorted_frame_ids) - 1
    if num_frames_to_process <= 0: return active_tracks

    for i in range(num_frames_to_process):
        prev_frame_id, current_frame_id = sorted_frame_ids[i], sorted_frame_ids[i+1]
        prev_spots = [t.spots_by_image_id[prev_frame_id] for t in active_tracks]
        current_spots = all_spots_by_frame[current_frame_id]
        
        # Only perform assignment if there are spots in both frames.
        if prev_spots and current_spots:
            # --- Core of the Algorithm ---
            # 1. Build a cost matrix: each element (r, c) is the squared Euclidean
            #    distance between previous spot r and current spot c.
            cost_matrix = np.array([[(p.center_x - c.center_x)**2 + (p.center_y - c.center_y)**2 for c in current_spots] for p in prev_spots])
            
            # 2. Solve the assignment problem using the Hungarian algorithm.
            #    This finds the optimal pairing that minimizes the total cost (distance).
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # --- Process Assignments ---
            max_dist_sq = max_distance**2
            unassigned_tracks, assigned_current_spot_indices = list(active_tracks), set()

            # 3. Link successfully matched spots.
            for r, c in zip(row_ind, col_ind):
                # Check if the matched distance is within the allowed maximum.
                if cost_matrix[r, c] < max_dist_sq:
                    track, spot = active_tracks[r], current_spots[c]
                    # Extend the existing track with the new spot.
                    track.spots_by_image_id[current_frame_id] = spot
                    # Mark track and spot as assigned.
                    unassigned_tracks.remove(track)
                    assigned_current_spot_indices.add(c)

            # 4. Handle unassigned tracks and spots.
            # Tracks from the previous frame that found no partner are considered terminated.
            completed_tracks.extend(unassigned_tracks)

            # The new set of active tracks are those that were successfully linked.
            active_tracks = [t for t in active_tracks if t not in unassigned_tracks]

            # Spots in the current frame that were not assigned to any old track start a new track.
            for c, spot in enumerate(current_spots):
                if c not in assigned_current_spot_indices:
                    active_tracks.append(Track(track_id=next_track_id, spots_by_image_id={current_frame_id: spot})); next_track_id += 1
        else:
            # Handle cases where one of the frames is empty.
            # All previously active tracks are terminated.
            completed_tracks.extend(active_tracks)
            # All spots in the current frame (if any) start new tracks.
            active_tracks = [Track(track_id=next_track_id + i, spots_by_image_id={current_frame_id: s}) for i, s in enumerate(current_spots)]
            next_track_id += len(current_spots)

        # Report progress if a callback function is provided.
        if progress_callback:
            progress_percent = int(((i + 1) / num_frames_to_process) * 100)
            progress_callback(progress_percent)

    # After the loop, any remaining active tracks are also considered completed.
    completed_tracks.extend(active_tracks)
    return completed_tracks


# --- OPTICAL FLOW TRACKER ---
def track_spots_optical_flow(
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    reference_frame_id: int,
    params: dict,
    progress_callback: Callable[[int], None] | None = None
) -> List[Track]:
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

def retrack_single_spot_optical_flow(
    track_to_update: Track,
    start_frame_id: int,
    all_files_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None
) -> Track:
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
) -> List[List[Tuple[int, float, float]]]:
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
) -> List[Track]:
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
    )

    roi_size = params.get("roi_size", 32)
    roi_half = roi_size // 2
    converted_tracks: List[Track] = []

    for spot, positions in zip(ref_file.spots, raw_tracks):
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
