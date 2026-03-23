# stal/logic/positions_analysis_coordinator.py

from __future__ import annotations

from threading import Event
from typing import Callable, Dict, List, Optional

import numpy as np
from skimage.color import rgb2gray
from skimage.util import img_as_float

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian
from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff


def run_positions_analysis(
    tracks: List[Track],
    groups_config: List[Dict],
    all_files_map: Dict[int, LeedFile],
    progress_callback: Callable[[int], None],
    cancel_event: Optional[Event] = None,
) -> Optional[Dict]:
    """
    Run positions analysis for configured groups on all frames.

    The function fits a 2D Gaussian inside each tracked ROI, then computes
    relative satellite positions with respect to the group's reference track.
    """

    def is_cancelled() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    track_map = {track.track_id: track for track in tracks}
    frame_ids = sorted(all_files_map.keys())
    groups = _normalize_groups(groups_config, track_map)
    if not groups:
        raise ValueError(
            "No valid groups configured. Each group must have one reference track and at least one satellite track."
        )

    required_track_ids = sorted(
        {
            group["reference_track_id"]
            for group in groups
            if group["reference_track_id"] is not None
        }
        | {tid for group in groups for tid in group["satellite_track_ids"]}
    )
    if not required_track_ids:
        raise ValueError("No tracks selected for positions analysis.")

    fit_points_by_frame: Dict[int, Dict[int, Dict]] = {}
    rows: List[Dict] = []
    failed_fits = 0
    successful_fits = 0

    total_steps = max(1, len(frame_ids) * len(required_track_ids))
    completed_steps = 0

    for frame_id in frame_ids:
        if is_cancelled():
            return None

        leed_file = all_files_map.get(frame_id)
        if leed_file is None:
            continue

        full_image = load_tiff(leed_file.filepath)
        if full_image is None or full_image.size == 0:
            completed_steps += len(required_track_ids)
            progress_callback(int(completed_steps * 100 / total_steps))
            continue
        full_image = _normalize_image_layout(full_image)

        frame_fits: Dict[int, Dict] = {}
        for track_id in required_track_ids:
            if is_cancelled():
                return None

            track = track_map[track_id]
            payload = fit_track_position_on_frame(track, frame_id, full_image)
            frame_fits[track_id] = payload

            if payload["track_present"]:
                if payload["fit_ok"]:
                    successful_fits += 1
                else:
                    failed_fits += 1

            completed_steps += 1
            progress_callback(int(completed_steps * 100 / total_steps))

        fit_points_by_frame[frame_id] = frame_fits

    for frame_id in frame_ids:
        if is_cancelled():
            return None
        leed_file = all_files_map.get(frame_id)
        if leed_file is None:
            continue

        frame_fits = fit_points_by_frame.get(frame_id, {})
        for group in groups:
            ref_track_id = group["reference_track_id"]
            ref_fit = frame_fits.get(ref_track_id)
            if ref_fit is None:
                ref_fit = _empty_fit_payload(frame_id, ref_track_id)

            for sat_track_id in group["satellite_track_ids"]:
                sat_fit = frame_fits.get(sat_track_id)
                if sat_fit is None:
                    sat_fit = _empty_fit_payload(frame_id, sat_track_id)

                dx = np.nan
                dy = np.nan
                r = np.nan
                if ref_fit["fit_ok"] and sat_fit["fit_ok"]:
                    dx = float(sat_fit["global_center_x"] - ref_fit["global_center_x"])
                    dy = float(sat_fit["global_center_y"] - ref_fit["global_center_y"])
                    r = float(np.hypot(dx, dy))

                rows.append(
                    {
                        "frame_id": frame_id,
                        "temperature": leed_file.temperature,
                        "coverage": leed_file.coverage,
                        "energy": leed_file.energy,
                        "group_id": group["group_id"],
                        "group_name": group["name"],
                        "reference_track_id": ref_track_id,
                        "satellite_track_id": sat_track_id,
                        "reference_spot_present": bool(ref_fit["track_present"]),
                        "satellite_spot_present": bool(sat_fit["track_present"]),
                        "reference_fit_ok": bool(ref_fit["fit_ok"]),
                        "satellite_fit_ok": bool(sat_fit["fit_ok"]),
                        "reference_spot_id": ref_fit.get("spot_id"),
                        "satellite_spot_id": sat_fit.get("spot_id"),
                        "x_fit_ref": ref_fit["global_center_x"],
                        "y_fit_ref": ref_fit["global_center_y"],
                        "x_fit_sat": sat_fit["global_center_x"],
                        "y_fit_sat": sat_fit["global_center_y"],
                        "dx": dx,
                        "dy": dy,
                        "r": r,
                        "reference_r_squared": ref_fit.get("r_squared", np.nan),
                        "satellite_r_squared": sat_fit.get("r_squared", np.nan),
                    }
                )

    summary = {
        "groups_configured": len(groups),
        "tracks_in_analysis": len(required_track_ids),
        "frames_processed": len(frame_ids),
        "rows_generated": len(rows),
        "successful_fits": successful_fits,
        "failed_fits": failed_fits,
    }

    progress_callback(100)
    return {
        "groups": groups,
        "rows": rows,
        "fit_points_by_frame": fit_points_by_frame,
        "summary": summary,
    }


def _normalize_groups(groups_config: List[Dict], track_map: Dict[int, Track]) -> List[Dict]:
    normalized: List[Dict] = []
    for group in groups_config:
        try:
            group_id = int(group.get("group_id"))
        except Exception:
            continue

        name = str(group.get("name", f"Group {group_id}"))
        reference_track_id = group.get("reference_track_id")
        try:
            reference_track_id = int(reference_track_id) if reference_track_id is not None else None
        except Exception:
            reference_track_id = None

        raw_satellites = group.get("satellite_track_ids", [])
        satellite_track_ids: List[int] = []
        for track_id in raw_satellites:
            try:
                tid = int(track_id)
            except Exception:
                continue
            if tid not in track_map:
                continue
            if reference_track_id is not None and tid == reference_track_id:
                continue
            if tid not in satellite_track_ids:
                satellite_track_ids.append(tid)

        if reference_track_id not in track_map:
            continue
        if not satellite_track_ids:
            continue

        normalized.append(
            {
                "group_id": group_id,
                "name": name,
                "reference_track_id": reference_track_id,
                "satellite_track_ids": satellite_track_ids,
            }
        )
    return normalized


def fit_track_position_on_frame(track: Track, frame_id: int, full_image: np.ndarray) -> Dict:
    """Fit a single tracked ROI on a single frame and return fit payload."""
    full_image = _normalize_image_layout(full_image)
    if frame_id not in track.spots_by_image_id:
        return _empty_fit_payload(frame_id, track.track_id)

    spot = track.spots_by_image_id[frame_id]
    rect = spot.roi_rect
    x0 = max(0, int(rect.x()))
    y0 = max(0, int(rect.y()))
    x1 = max(x0, min(full_image.shape[1], x0 + int(rect.width())))
    y1 = max(y0, min(full_image.shape[0], y0 + int(rect.height())))
    roi_data = full_image[y0:y1, x0:x1].copy()

    if roi_data.size == 0:
        payload = _empty_fit_payload(frame_id, track.track_id)
        payload["track_present"] = True
        payload["spot_id"] = spot.id
        return payload

    roi_to_fit = prepare_roi_for_fit(roi_data)
    fit_results = fit_spot_to_gaussian(roi_to_fit)
    fit_ok = is_fit_valid(fit_results, roi_to_fit.shape)
    if not fit_ok:
        fit_results = None

    payload = {
        "frame_id": frame_id,
        "track_id": track.track_id,
        "spot_id": spot.id,
        "track_present": True,
        "fit_ok": fit_ok,
        "global_center_x": np.nan,
        "global_center_y": np.nan,
        "center_x_local": np.nan,
        "center_y_local": np.nan,
        "r_squared": np.nan,
        "fit_results": fit_results,
    }

    if fit_results:
        payload["center_x_local"] = float(fit_results["center_x"])
        payload["center_y_local"] = float(fit_results["center_y"])
        payload["global_center_x"] = float(x0 + fit_results["center_x"])
        payload["global_center_y"] = float(y0 + fit_results["center_y"])
        payload["r_squared"] = float(fit_results.get("r_squared", np.nan))

    return payload


def prepare_roi_for_fit(roi_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(roi_data)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim >= 3:
        if arr.shape[-1] >= 3:
            arr = rgb2gray(arr[..., :3])
        else:
            arr = arr[..., 0]
    return img_as_float(arr)


def _normalize_image_layout(image_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_data)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _empty_fit_payload(frame_id: int, track_id: int) -> Dict:
    return {
        "frame_id": frame_id,
        "track_id": track_id,
        "spot_id": None,
        "track_present": False,
        "fit_ok": False,
        "global_center_x": np.nan,
        "global_center_y": np.nan,
        "center_x_local": np.nan,
        "center_y_local": np.nan,
        "r_squared": np.nan,
        "fit_results": None,
    }


def is_fit_valid(fit_results: Optional[Dict], roi_shape: tuple) -> bool:
    if not fit_results:
        return False
    if roi_shape is None or len(roi_shape) < 2:
        return False

    height, width = roi_shape[:2]
    if fit_results.get("r_squared", 0) < 0.85:
        return False
    if fit_results.get("amplitude", 0) < 0.05:
        return False
    if not (-2 < fit_results.get("center_x", 0) < width + 2):
        return False
    if not (-2 < fit_results.get("center_y", 0) < height + 2):
        return False
    if not (0.5 < fit_results.get("sigma_x", 0) < width):
        return False
    if not (0.5 < fit_results.get("sigma_y", 0) < height):
        return False
    return True
