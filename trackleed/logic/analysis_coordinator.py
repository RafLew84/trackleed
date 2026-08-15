"""Compatibility entry point and legacy Track Analysis geometry helpers."""

from __future__ import annotations

from threading import Event
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from trackleed.analysis.legacy_track_geometry import (
    LegacyCenterMode,
    ManualEllipse,
    ellipse_corrected_radius,
    estimate_legacy_center,
    euclidean_radius,
    manual_ellipse_from_mapping,
    smooth_legacy_centers,
    summarize_radii,
)
from trackleed.core.data_models import LeedFile, Track
from trackleed.logic.workflows.track_analysis import FitSet


def run_full_analysis(
    tracks_to_analyze: List[Track],
    analysis_config: Dict[int, Dict],
    all_files_map: Dict[int, LeedFile],
    background_file: Optional[LeedFile],
    progress_callback: Callable[[int], None],
    center_mode: str = "per_frame",
    manual_ellipse: Optional[Dict[str, float]] = None,
    cancel_event: Optional[Event] = None,
    previous_fit_set: FitSet | None = None,
) -> Optional[List[Track]]:
    """Preserve the historical return type over the atomic compatibility adapter."""
    from trackleed.logic.track_analysis_compatibility import (
        publish_analysis,
        run_full_analysis_publication,
    )

    publication = run_full_analysis_publication(
        tracks_to_analyze,
        analysis_config,
        all_files_map,
        background_file,
        progress_callback,
        center_mode,
        manual_ellipse,
        cancel_event,
        previous_fit_set,
    )
    if publication is None:
        return None
    publish_analysis(publication, tracks_to_analyze, all_files_map)
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
    try:
        mode = LegacyCenterMode(center_mode)
    except ValueError:
        mode = LegacyCenterMode.PER_FRAME
    if mode is LegacyCenterMode.MANUAL_ELLIPSE:
        return None
    return estimate_legacy_center(spot_positions_on_this_frame, mode)


def _smooth_centers_temporally(all_frame_ids: List[int], raw_centers: Dict[int, Dict]) -> Dict[int, Dict[str, float]]:
    """Compatibility wrapper around the neutral scientific implementation."""
    return smooth_legacy_centers(all_frame_ids, raw_centers)


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

        radius, ordinary_error = euclidean_radius(
            float(results['global_center_x']),
            float(results['global_center_y']),
            float(results.get('center_x_error', 0.0)),
            float(results.get('center_y_error', 0.0)),
            center_info,
        )
        results['radius'] = radius
        if ordinary_error is not None:
            results['radius_error'] = ordinary_error


def _normalize_manual_ellipse(manual_ellipse: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Normalize and validate manual ellipse payload to internal analysis fields."""
    ellipse = manual_ellipse_from_mapping(manual_ellipse)
    if ellipse is None:
        return None
    return {
        "center_x": ellipse.center_x,
        "center_y": ellipse.center_y,
        "radius_x": ellipse.radius_x,
        "radius_y": ellipse.radius_y,
        "angle_deg": ellipse.angle_degrees,
        "radius_scale_px": float(np.sqrt(ellipse.radius_x * ellipse.radius_y)),
    }


def _compute_ellipse_corrected_radius(
    fit_results: Dict[str, float],
    center_info: Dict[str, float],
    manual_ellipse: Dict[str, float],
) -> Tuple[float, float]:
    """Compatibility wrapper around the neutral scientific implementation."""
    ellipse = ManualEllipse(
        float(manual_ellipse["center_x"]),
        float(manual_ellipse["center_y"]),
        float(manual_ellipse["radius_x"]),
        float(manual_ellipse["radius_y"]),
        float(manual_ellipse.get("angle_deg", 0.0)),
    )
    resolved_center = {
        "center_x": float(center_info.get("center_x", ellipse.center_x)),
        "center_y": float(center_info.get("center_y", ellipse.center_y)),
        "error_x": float(center_info.get("error_x", 0.0)),
        "error_y": float(center_info.get("error_y", 0.0)),
    }
    return ellipse_corrected_radius(
        float(fit_results["global_center_x"]),
        float(fit_results["global_center_y"]),
        float(fit_results.get("center_x_error", 0.0)),
        float(fit_results.get("center_y_error", 0.0)),
        resolved_center,
        ellipse,
        float(manual_ellipse.get("radius_scale_px", np.sqrt(
            ellipse.radius_x * ellipse.radius_y
        ))),
    )


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

    avg_radius, avg_radius_error = summarize_radii(radii_on_this_frame)
    if avg_radius is not None:
        if not hasattr(leed_file, 'post_analysis_results'):
            leed_file.post_analysis_results = {}
        leed_file.post_analysis_results['avg_radius'] = avg_radius
        leed_file.post_analysis_results['avg_radius_error'] = avg_radius_error
        return

    post_analysis_results = getattr(leed_file, 'post_analysis_results', None)
    if post_analysis_results:
        post_analysis_results.pop('avg_radius', None)
        post_analysis_results.pop('avg_radius_error', None)
