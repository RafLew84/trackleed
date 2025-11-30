# stal/analysis/tracking.py

from pathlib import Path
from typing import List, Dict, Callable, Tuple, Optional
from threading import Event

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QRect
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2gray
from skimage.morphology import white_tophat, disk
from skimage.util import img_as_ubyte

from trackleed.core.data_models import LeedFile, SpotROI, Track
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.flow_raft import RAFTWrapper, RAFTConfig
from trackleed.analysis.peak_analyzer import find_pattern_center


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


def track_spots_raft_fallback(
    file_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots using RAFT dense flow with LK fallback for unreliable vectors.
    Expects detections (SpotROI) already present in each LeedFile.
    """
    model_path = params.get("model_path")
    if not model_path:
        raise ValueError("RAFT model path not provided.")
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    raft_cfg = RAFTConfig(
        model_path=Path(model_path),
        device=device,
        resize_short_side=params.get("raft_resize_short_side", 512),
        fwd_bwd_tau=float(params.get("fwd_bwd_tau", 1.5)),
        flow_conf_threshold=float(params.get("flow_conf_threshold", 0.0)),
        max_step_px=params.get("max_step_px"),
        small=bool(params.get("raft_small", True)),
        alternate_corr=bool(params.get("alternate_corr", False)),
        mixed_precision=bool(params.get("mixed_precision", False)),
        dropout=float(params.get("dropout", 0.0)),
    )
    raft = RAFTWrapper(raft_cfg)

    roi_size = params.get("roi_size", 32)
    roi_half = roi_size // 2
    lk_params = dict(
        winSize=(params.get("lk_win_size", 41), params.get("lk_win_size", 41)),
        maxLevel=params.get("lk_max_level", 3),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            params.get("lk_max_iterations", 10),
            params.get("lk_epsilon", 0.03),
        ),
        minEigThreshold=params.get("lk_min_eigen_threshold", 1e-5),
    )

    # Initialize tracks from first frame detections
    if not sorted_frame_ids:
        return []
    first_id = sorted_frame_ids[0]
    first_file = file_map.get(first_id)
    if not first_file or not first_file.spots:
        raise ValueError("First frame must contain detections to start tracking.")
    tracks_map = {s.id: Track(track_id=s.id, spots_by_image_id={first_id: s}) for s in first_file.spots}
    last_positions = {s.id: np.array([s.center_x, s.center_y], dtype=np.float32) for s in first_file.spots}

    total = max(1, len(sorted_frame_ids) - 1)
    for idx in range(len(sorted_frame_ids) - 1):
        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(int(100 * idx / total))

        fid_a = sorted_frame_ids[idx]
        fid_b = sorted_frame_ids[idx + 1]
        file_a = file_map.get(fid_a)
        file_b = file_map.get(fid_b)
        if not file_a or not file_b:
            continue

        img_a = load_tiff(file_a.filepath)
        img_b = load_tiff(file_b.filepath)
        if img_a is None or img_a.size == 0 or img_b is None or img_b.size == 0:
            continue

        try:
            flow, fb_mask = raft.compute_flow(img_a, img_b)
        except Exception as exc:
            print(f"RAFT flow failed on frames {fid_a}->{fid_b}: {exc}")
            continue

        # Prepare LK images for fallback
        prev_gray = _load_and_prep_image(file_a.filepath)
        curr_gray = _load_and_prep_image(file_b.filepath)
        p0_list, track_ids = [], []
        for tid, pos in last_positions.items():
            p0_list.append(pos)
            track_ids.append(tid)
        if p0_list:
            p0 = np.array(p0_list, dtype=np.float32).reshape(-1, 1, 2)
        else:
            break

        # For each track: sample flow
        new_positions = {}
        need_lk = []
        for tid, pos in zip(track_ids, p0_list):
            fv = _bilinear_sample_flow(flow, pos)
            if fv is None:
                need_lk.append((tid, pos))
                continue
            x_new = pos[0] + fv[0]
            y_new = pos[1] + fv[1]
            # check fb mask
            x_i, y_i = int(round(pos[0])), int(round(pos[1]))
            ok_fb = False
            if 0 <= y_i < fb_mask.shape[0] and 0 <= x_i < fb_mask.shape[1]:
                ok_fb = bool(fb_mask[y_i, x_i])
            if not ok_fb:
                need_lk.append((tid, pos))
                continue
            new_positions[tid] = np.array([x_new, y_new], dtype=np.float32)

        # LK fallback where needed
        if need_lk and prev_gray is not None and curr_gray is not None:
            p0_fallback = np.array([p for _, p in need_lk], dtype=np.float32).reshape(-1, 1, 2)
            p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0_fallback, None, **lk_params)
            for (tid, pos), st, newp in zip(need_lk, status, p1):
                if st[0] == 1:
                    new_positions[tid] = newp.reshape(-1)

        # Update tracks
        for tid, new_pos in new_positions.items():
            roi = QRect(int(new_pos[0] - roi_half), int(new_pos[1] - roi_half), roi_size, roi_size)
            spot = SpotROI(id=tid, roi_rect=roi, center_x=float(new_pos[0]), center_y=float(new_pos[1]), intensity=0, source="raft")
            tracks_map[tid].spots_by_image_id[fid_b] = spot
        last_positions = new_positions

    return list(tracks_map.values())


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


def _bilinear_sample_flow(flow: np.ndarray, xy: np.ndarray) -> Optional[np.ndarray]:
    """Sample flow (H,W,2) at floating-point xy."""
    h, w, _ = flow.shape
    x, y = xy
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    f00 = flow[y0, x0]
    f10 = flow[y0, x0 + 1]
    f01 = flow[y0 + 1, x0]
    f11 = flow[y0 + 1, x0 + 1]
    return (f00 * (1 - dx) * (1 - dy) + f10 * dx * (1 - dy) + f01 * (1 - dx) * dy + f11 * dx * dy)


def compute_global_pattern_center(
    files: list[LeedFile],
    per_frame_error_threshold: float = 3.0,
    iqr_multiplier: float = 3.0,
) -> Optional[dict]:
    """
    Estimate a global pattern center using YOLO/spot detections from many frames.
    - For each frame with >=4 spots: run RANSAC center; collect (cx, cy, err).
    - Reject frames with large error; reject outliers by distance from median.
    - Return median center and an uncertainty (mad-based) or None if insufficient data.
    """
    candidates = []
    for lf in files:
        if not lf.spots or len(lf.spots) < 4:
            continue
        spot_positions = [{"center_x": s.center_x, "center_y": s.center_y} for s in lf.spots]
        center_info = find_pattern_center(spot_positions)
        if center_info is None:
            continue
        err = max(center_info.get("error_x", 0), center_info.get("error_y", 0))
        if err > per_frame_error_threshold:
            continue
        candidates.append(center_info)

    if len(candidates) < 1:
        return None

    cx_all = np.array([c["center_x"] for c in candidates], dtype=np.float32)
    cy_all = np.array([c["center_y"] for c in candidates], dtype=np.float32)

    med_cx, med_cy = np.median(cx_all), np.median(cy_all)
    dist = np.sqrt((cx_all - med_cx) ** 2 + (cy_all - med_cy) ** 2)
    q1, q3 = np.percentile(dist, [25, 75])
    iqr = q3 - q1
    keep_mask = dist <= (q3 + iqr_multiplier * iqr if iqr > 0 else dist.max())
    cx_keep = cx_all[keep_mask]
    cy_keep = cy_all[keep_mask]

    if cx_keep.size == 0:
        return None

    final_cx = float(np.median(cx_keep))
    final_cy = float(np.median(cy_keep))
    mad_cx = float(np.median(np.abs(cx_keep - final_cx))) if cx_keep.size > 1 else 0.0
    mad_cy = float(np.median(np.abs(cy_keep - final_cy))) if cy_keep.size > 1 else 0.0

    result = {
        "center_x": final_cx,
        "center_y": final_cy,
        "error_x": mad_cx,
        "error_y": mad_cy,
        "frames_used": int(cx_keep.size),
        "frames_total": int(len(candidates)),
    }
    return result


def track_spots_polar_hungarian(
    file_map: Dict[int, LeedFile],
    sorted_frame_ids: List[int],
    params: dict,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: Optional[Event] = None,
) -> Optional[List[Track]]:
    """
    Track spots using Hungarian matching in polar coordinates with gating.
    """
    if not sorted_frame_ids:
        return []
    # Determine center
    center_source = params.get("center_source", "auto")
    center: Tuple[float, float] | None = None
    if center_source == "file":
        for fid in sorted_frame_ids:
            lf = file_map.get(fid)
            if lf and lf.pattern_center:
                c = _center_from_pattern_dict(lf.pattern_center)
                if c:
                    center = c
                    break
    if center is None:
        center_info = compute_global_pattern_center(list(file_map.values()))
        if center_info:
            center = (center_info["center_x"], center_info["center_y"])
    if center is None:
        # fallback: center of first frame image
        first_file = file_map.get(sorted_frame_ids[0])
        if first_file:
            img = load_tiff(first_file.filepath)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                center = (w / 2.0, h / 2.0)
            else:
                center = (0.0, 0.0)
        else:
            center = (0.0, 0.0)

    max_dr = float(params.get("max_dr", 20.0))
    max_dtheta = float(params.get("max_dtheta", 0.05))
    direction = params.get("direction", "both").lower()
    max_missing = int(params.get("max_missing", 3))
    radial_slack = float(params.get("radial_slack", max_dr * 0.25))
    next_track_id = 1

    # Initialize tracks from first frame detections
    first_id = sorted_frame_ids[0]
    first_file = file_map.get(first_id)
    if not first_file or not first_file.spots:
        return []
    tracks: Dict[int, Track] = {s.id: Track(track_id=s.id, spots_by_image_id={first_id: s}) for s in first_file.spots}
    missing_counts: Dict[int, int] = {s.id: 0 for s in first_file.spots}
    next_track_id = max(next_track_id, max(tracks.keys()) + 1 if tracks else 1)

    total = max(1, len(sorted_frame_ids) - 1)
    for idx in range(len(sorted_frame_ids) - 1):
        if cancel_event and cancel_event.is_set():
            return None
        if progress_callback:
            progress_callback(int(100 * idx / total))

        fid = sorted_frame_ids[idx + 1]
        lf = file_map.get(fid)
        if lf is None:
            continue
        detections = lf.spots if lf.spots else []
        tracks_list = list(tracks.values())
        tracks_list, missing_counts, assigned_dets = match_tracks_polar(
            tracks_list,
            detections,
            center=center,
            max_dr=max_dr,
            max_dtheta=max_dtheta,
            direction=direction,
            max_missing=max_missing,
            radial_slack=radial_slack,
            missing_counts=missing_counts,
            frame_id=fid,
        )
        # Rebuild dict and drop missing
        tracks = {tr.track_id: tr for tr in tracks_list}
        # Start new tracks for unmatched detections
        unmatched = [det for idx, det in enumerate(detections) if idx not in assigned_dets]
        for det in unmatched:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = Track(track_id=tid, spots_by_image_id={fid: det})
            missing_counts[tid] = 0
    return list(tracks.values())


def _to_polar(center: Tuple[float, float], point: Tuple[float, float]) -> Tuple[float, float]:
    cx, cy = center
    x, y = point
    dx, dy = x - cx, y - cy
    r = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    return r, theta


def match_tracks_polar(
    tracks: List[Track],
    detections: List[SpotROI],
    center: Tuple[float, float],
    max_dr: float,
    max_dtheta: float,
    direction: str = "both",
    alpha: float = 1.0,
    beta: float = 10.0,
    max_missing: int = 3,
    radial_slack: float = 0.0,
    missing_counts: Optional[dict] = None,
    frame_id: Optional[int] = None,
) -> Tuple[List[Track], dict, set]:
    """
    Hungarian matching in polar coords with gating.
    direction: "both" | "inward" | "outward"
    missing_counts: dict track_id -> consecutive misses
    """
    if missing_counts is None:
        missing_counts = {}
    # Current positions of tracks: take last known spot
    track_points = []
    for tr in tracks:
        if not tr.spots_by_image_id:
            continue
        last_frame = max(tr.spots_by_image_id.keys())
        last_spot = tr.spots_by_image_id[last_frame]
        track_points.append((tr, last_spot))

    if not track_points or not detections:
        if not track_points:
            return tracks, missing_counts, set()
        # No detections: increase missing counters
        new_tracks = []
        for tr, _ in track_points:
            miss = missing_counts.get(tr.track_id, 0) + 1
            missing_counts[tr.track_id] = miss
            if miss <= max_missing:
                new_tracks.append(tr)
        return new_tracks, missing_counts, set()

    # Build cost matrix with gating
    nT = len(track_points)
    nD = len(detections)
    cost = np.full((nT, nD), fill_value=1e9, dtype=np.float32)
    for i, (tr, spot_t) in enumerate(track_points):
        r_t, th_t = _to_polar(center, (spot_t.center_x, spot_t.center_y))
        for j, det in enumerate(detections):
            r_d, th_d = _to_polar(center, (det.center_x, det.center_y))
            dr = r_d - r_t
            dtheta = abs(th_d - th_t)
            # wrap angle to [-pi, pi]
            if dtheta > np.pi:
                dtheta = 2 * np.pi - dtheta
            # direction gating
            if direction == "inward" and dr > radial_slack:
                continue
            if direction == "outward" and dr < -radial_slack:
                continue
            if abs(dr) > max_dr or dtheta > max_dtheta:
                continue
            cost[i, j] = alpha * abs(dr) + beta * (dtheta * r_t)

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_tracks = set()
    assigned_dets = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e9:
            continue  # gated out
        tr, spot_prev = track_points[r]
        det = detections[c]
        frame_key = frame_id if frame_id is not None else det.id
        tr.spots_by_image_id[frame_key] = det
        assigned_tracks.add(tr.track_id)
        assigned_dets.add(c)
        missing_counts[tr.track_id] = 0

    # Update missing counts and drop tracks that exceed max_missing
    new_tracks = []
    for tr, _ in track_points:
        if tr.track_id not in assigned_tracks:
            miss = missing_counts.get(tr.track_id, 0) + 1
            missing_counts[tr.track_id] = miss
            if miss > max_missing:
                continue  # drop track
        new_tracks.append(tr)

    return new_tracks, missing_counts, assigned_dets


def _center_from_pattern_dict(center_dict: dict) -> Optional[Tuple[float, float]]:
    """Extract (cx, cy) from various center dict formats."""
    if not center_dict:
        return None
    for kx, ky in [("center_x", "center_y"), ("cx", "cy"), ("x", "y")]:
        if kx in center_dict and ky in center_dict:
            try:
                return float(center_dict[kx]), float(center_dict[ky])
            except Exception:
                continue
    return None


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
