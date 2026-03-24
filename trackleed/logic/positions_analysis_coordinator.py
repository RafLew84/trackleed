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

DEFAULT_VALIDATION_CONFIG: Dict[str, float] = {
    "min_r_squared": 0.85,
    "min_amplitude": 0.05,
    "center_margin_px": 2.0,
    "min_sigma_px": 0.5,
    "max_sigma_factor": 1.0,
    "center_preference_factor": 0.35,
}

QUALITY_FLAG_LABELS: Dict[str, str] = {
    "fit_missing": "fit missing",
    "invalid_roi_shape": "invalid ROI shape",
    "low_r_squared": "low R²",
    "low_amplitude": "low amplitude",
    "center_x_outside_roi": "center X outside ROI",
    "center_y_outside_roi": "center Y outside ROI",
    "sigma_x_out_of_range": "sigma X out of range",
    "sigma_y_out_of_range": "sigma Y out of range",
    "r_squared_near_threshold": "R² near threshold",
    "amplitude_near_threshold": "amplitude near threshold",
    "center_near_roi_edge": "center near ROI edge",
    "center_far_from_roi_center": "center far from ROI center",
    "sigma_near_lower_bound": "sigma near lower bound",
    "sigma_near_upper_bound": "sigma near upper bound",
}


def normalize_validation_config(config: Optional[Dict] = None) -> Dict[str, float]:
    normalized = dict(DEFAULT_VALIDATION_CONFIG)
    if not config:
        return normalized

    for key, default_value in DEFAULT_VALIDATION_CONFIG.items():
        raw_value = config.get(key, default_value)
        try:
            value = float(raw_value)
        except Exception:
            value = float(default_value)
        if not np.isfinite(value):
            value = float(default_value)
        normalized[key] = value

    normalized["min_r_squared"] = float(np.clip(normalized["min_r_squared"], 0.0, 1.0))
    normalized["min_amplitude"] = max(0.0, normalized["min_amplitude"])
    normalized["center_margin_px"] = max(0.0, normalized["center_margin_px"])
    normalized["min_sigma_px"] = max(0.01, normalized["min_sigma_px"])
    normalized["max_sigma_factor"] = max(0.01, normalized["max_sigma_factor"])
    normalized["center_preference_factor"] = max(0.0, normalized["center_preference_factor"])
    return normalized


def run_positions_analysis(
    tracks: List[Track],
    groups_config: List[Dict],
    all_files_map: Dict[int, LeedFile],
    progress_callback: Callable[[int], None],
    validation_config: Optional[Dict] = None,
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
    validation_config = normalize_validation_config(validation_config)
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
    suspicious_fits = 0

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
            payload = fit_track_position_on_frame(
                track,
                frame_id,
                full_image,
                validation_config=validation_config,
            )
            frame_fits[track_id] = payload

            if payload["track_present"]:
                if payload["fit_ok"]:
                    successful_fits += 1
                else:
                    failed_fits += 1
                if payload.get("is_suspicious"):
                    suspicious_fits += 1

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
                        "reference_quality_status": ref_fit.get("quality_status", "failed"),
                        "satellite_quality_status": sat_fit.get("quality_status", "failed"),
                        "reference_quality_flags": list(ref_fit.get("quality_flags", [])),
                        "satellite_quality_flags": list(sat_fit.get("quality_flags", [])),
                        "reference_validation_failures": list(ref_fit.get("validation_failures", [])),
                        "satellite_validation_failures": list(sat_fit.get("validation_failures", [])),
                        "reference_warning_flags": list(ref_fit.get("warning_flags", [])),
                        "satellite_warning_flags": list(sat_fit.get("warning_flags", [])),
                        "reference_suspicious": bool(ref_fit.get("is_suspicious", False)),
                        "satellite_suspicious": bool(sat_fit.get("is_suspicious", False)),
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
                        "suspicious": bool(
                            ref_fit.get("is_suspicious", False)
                            or sat_fit.get("is_suspicious", False)
                        ),
                        "row_quality_status": _combine_row_quality_status(
                            ref_fit.get("quality_status", "failed"),
                            sat_fit.get("quality_status", "failed"),
                        ),
                    }
                )

    suspicious_rows = int(sum(1 for row in rows if row.get("suspicious")))
    summary = {
        "groups_configured": len(groups),
        "tracks_in_analysis": len(required_track_ids),
        "frames_processed": len(frame_ids),
        "rows_generated": len(rows),
        "successful_fits": successful_fits,
        "failed_fits": failed_fits,
        "suspicious_fits": suspicious_fits,
        "suspicious_rows": suspicious_rows,
    }

    progress_callback(100)
    return {
        "groups": groups,
        "rows": rows,
        "fit_points_by_frame": fit_points_by_frame,
        "summary": summary,
        "validation_config": validation_config,
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


def fit_track_position_on_frame(
    track: Track,
    frame_id: int,
    full_image: np.ndarray,
    validation_config: Optional[Dict] = None,
) -> Dict:
    """Fit a single tracked ROI on a single frame and return fit payload."""
    full_image = _normalize_image_layout(full_image)
    validation_config = normalize_validation_config(validation_config)
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
    fit_results = fit_spot_to_gaussian(
        roi_to_fit,
        center_preference_factor=validation_config.get("center_preference_factor", 0.0),
    )
    quality_info = evaluate_fit_quality(fit_results, roi_to_fit.shape, validation_config)
    fit_ok = bool(quality_info["fit_ok"])

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
        "validation_failures": list(quality_info["validation_failures"]),
        "warning_flags": list(quality_info["warning_flags"]),
        "quality_flags": list(quality_info["quality_flags"]),
        "quality_status": quality_info["quality_status"],
        "is_suspicious": bool(quality_info["is_suspicious"]),
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
        "validation_failures": ["fit_missing"],
        "warning_flags": [],
        "quality_flags": ["fit_missing"],
        "quality_status": "failed",
        "is_suspicious": True,
    }


def describe_quality_flags(flags: Optional[List[str]]) -> List[str]:
    descriptions: List[str] = []
    for flag in flags or []:
        descriptions.append(QUALITY_FLAG_LABELS.get(str(flag), str(flag).replace("_", " ")))
    return descriptions


def format_quality_flags(flags: Optional[List[str]], *, separator: str = ", ") -> str:
    return separator.join(describe_quality_flags(flags))


def evaluate_fit_quality(
    fit_results: Optional[Dict],
    roi_shape: tuple,
    validation_config: Optional[Dict] = None,
) -> Dict[str, object]:
    validation_failures: List[str] = []
    warning_flags: List[str] = []

    if not fit_results:
        validation_failures.append("fit_missing")
        return {
            "fit_ok": False,
            "validation_failures": validation_failures,
            "warning_flags": warning_flags,
            "quality_flags": validation_failures + warning_flags,
            "quality_status": "failed",
            "is_suspicious": True,
        }
    if roi_shape is None or len(roi_shape) < 2:
        validation_failures.append("invalid_roi_shape")
        return {
            "fit_ok": False,
            "validation_failures": validation_failures,
            "warning_flags": warning_flags,
            "quality_flags": validation_failures + warning_flags,
            "quality_status": "failed",
            "is_suspicious": True,
        }

    validation_config = normalize_validation_config(validation_config)
    height, width = roi_shape[:2]
    center_margin = validation_config["center_margin_px"]
    min_sigma = validation_config["min_sigma_px"]
    max_sigma_x = max(min_sigma, validation_config["max_sigma_factor"] * width)
    max_sigma_y = max(min_sigma, validation_config["max_sigma_factor"] * height)

    r_squared = _safe_float(fit_results.get("r_squared"))
    amplitude = _safe_float(fit_results.get("amplitude"))
    center_x = _safe_float(fit_results.get("center_x"))
    center_y = _safe_float(fit_results.get("center_y"))
    sigma_x = _safe_float(fit_results.get("sigma_x"))
    sigma_y = _safe_float(fit_results.get("sigma_y"))

    if r_squared is None or r_squared < validation_config["min_r_squared"]:
        validation_failures.append("low_r_squared")
    elif r_squared < min(1.0, validation_config["min_r_squared"] + 0.05):
        warning_flags.append("r_squared_near_threshold")

    if amplitude is None or amplitude < validation_config["min_amplitude"]:
        validation_failures.append("low_amplitude")
    elif amplitude < max(
        validation_config["min_amplitude"] * 1.5,
        validation_config["min_amplitude"] + 0.05,
    ):
        warning_flags.append("amplitude_near_threshold")

    if center_x is None or not (-center_margin < center_x < width + center_margin):
        validation_failures.append("center_x_outside_roi")
    if center_y is None or not (-center_margin < center_y < height + center_margin):
        validation_failures.append("center_y_outside_roi")

    if sigma_x is None or not (min_sigma < sigma_x < max_sigma_x):
        validation_failures.append("sigma_x_out_of_range")
    if sigma_y is None or not (min_sigma < sigma_y < max_sigma_y):
        validation_failures.append("sigma_y_out_of_range")

    if center_x is not None and center_y is not None:
        edge_distance = min(center_x, center_y, width - center_x, height - center_y)
        if edge_distance < max(1.0, 0.10 * min(width, height)):
            warning_flags.append("center_near_roi_edge")
        roi_center_x = (width - 1) / 2.0
        roi_center_y = (height - 1) / 2.0
        center_offset = float(np.hypot(center_x - roi_center_x, center_y - roi_center_y))
        if center_offset > 0.35 * min(width, height):
            warning_flags.append("center_far_from_roi_center")

    if sigma_x is not None and sigma_y is not None:
        if sigma_x < 1.2 * min_sigma or sigma_y < 1.2 * min_sigma:
            warning_flags.append("sigma_near_lower_bound")
        if sigma_x > 0.80 * max_sigma_x or sigma_y > 0.80 * max_sigma_y:
            warning_flags.append("sigma_near_upper_bound")

    validation_failures = _deduplicate_preserve_order(validation_failures)
    warning_flags = _deduplicate_preserve_order(warning_flags)
    fit_ok = len(validation_failures) == 0
    quality_status = "failed" if not fit_ok else ("warning" if warning_flags else "ok")
    return {
        "fit_ok": fit_ok,
        "validation_failures": validation_failures,
        "warning_flags": warning_flags,
        "quality_flags": validation_failures + warning_flags,
        "quality_status": quality_status,
        "is_suspicious": bool((not fit_ok) or warning_flags),
    }


def is_fit_valid(
    fit_results: Optional[Dict],
    roi_shape: tuple,
    validation_config: Optional[Dict] = None,
) -> bool:
    return bool(
        evaluate_fit_quality(
            fit_results,
            roi_shape,
            validation_config,
        )["fit_ok"]
    )


def _safe_float(value) -> float | None:
    try:
        numeric_value = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _deduplicate_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    deduplicated: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _combine_row_quality_status(reference_status: str, satellite_status: str) -> str:
    statuses = {reference_status, satellite_status}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return "ok"
