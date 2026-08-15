"""Numerical checks for explicitly declared local Track Analysis series."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from trackleed.analysis.current_analysis_acceptance import sha256_files
from trackleed.core.data_models import LeedFile, Track
from trackleed.io.session import load_session_from_file
from trackleed.logic.workflows.gaussian_fit_recovery import (
    FitRecoveryConfig,
    RecoveryStatus,
)
from trackleed.logic.workflows.track_analysis import (
    FitObservationInput,
    FitRequest,
    FitSet,
    Rect,
    fit,
)
from trackleed.logic.workflows.track_geometry import (
    GeometryMethod,
    GeometryParameters,
    GeometryRequest,
    GeometryResult,
    GeometryStatus,
    derive_geometry,
)

GOLDEN_CENTER_TOLERANCE_PX = 1.0
CENTER_METHODS = (
    GeometryMethod.PER_FRAME,
    GeometryMethod.CIRCLE_FIT,
    GeometryMethod.CIRCLE_FIT_BBOX_FALLBACK,
    GeometryMethod.ELLIPSE_FIT,
    GeometryMethod.CIRCLE_FIT_TEMPORAL,
    GeometryMethod.GLOBAL_MEDIAN,
)
RANSAC_SEEDS = (1, 7, 20260811, 999999)


class ReferenceSeriesError(RuntimeError):
    """A persisted real-series or golden-frame check failed."""


@dataclass(frozen=True)
class ReferenceAcceptanceCriteria:
    minimum_method_available_frames: int
    minimum_temporal_pairs: int
    max_center_error_px: float
    max_temporal_jump_px: float
    max_spot_removal_shift_px: float
    max_ransac_seed_spread_px: float


@dataclass(frozen=True)
class ReferenceSeriesSpec:
    name: str
    session_path: Path
    analysis_manifest_path: Path
    golden_frame_ids: tuple[int, ...]
    criteria: ReferenceAcceptanceCriteria


def analyze_reference_series(spec: ReferenceSeriesSpec) -> dict[str, Any]:
    """Rerun approved frames and evaluate every production center strategy."""
    contents = load_session_from_file(spec.session_path)
    files = {int(item.id): item for item in contents.files if item.id is not None}
    _require_sources(spec, files)
    expected_counts = _expected_counts(spec.analysis_manifest_path)
    consistency = _validate_self_consistency(
        spec,
        contents.tracks,
        files,
        expected_counts,
    )
    _require_golden_oracles(spec, contents.tracks, files)
    selected_tracks, selected_files = _select_goldens(spec, contents.tracks, files)
    if len(selected_tracks) < 4:
        raise ReferenceSeriesError(
            f"{spec.name}: golden center checks require at least four tracks"
        )
    inputs = _fit_inputs(selected_tracks, selected_files)
    fit_set = fit(FitRequest(inputs))
    golden_comparison = _compare_golden_fits(
        spec,
        contents.tracks,
        files,
        fit_set,
    )
    track_ids = tuple(track.track_id for track in selected_tracks)
    geometries = {
        method.value: derive_geometry(
            fit_set,
            GeometryRequest(method=method, center_reference_track_ids=track_ids),
        )
        for method in CENTER_METHODS
    }
    method_metrics = {
        method: _center_metrics(result, files) for method, result in geometries.items()
    }
    circle_metrics = method_metrics[GeometryMethod.CIRCLE_FIT.value]
    circle_error = circle_metrics["error_max_px"]
    if (
        circle_metrics["available_frames"] != len(spec.golden_frame_ids)
        or circle_error is None
        or circle_error > GOLDEN_CENTER_TOLERANCE_PX
    ):
        raise ReferenceSeriesError(
            f"{spec.name}: every golden frame must reproduce its center within "
            f"{GOLDEN_CENTER_TOLERANCE_PX:.1f} px; unavailable frames: "
            f"{circle_metrics['unavailable_frame_ids']}"
        )
    spot_removal = _spot_removal_metrics(fit_set, track_ids, files)
    refusals = _refusal_metrics(fit_set, track_ids)
    ransac = _ransac_metrics(fit_set, track_ids)
    _enforce_acceptance(spec, method_metrics, spot_removal, ransac)
    return {
        "name": spec.name,
        "session": str(spec.session_path.resolve()),
        "analysis_manifest": str(spec.analysis_manifest_path.resolve()),
        "input_sha256": sha256_files(
            (
                spec.session_path,
                spec.analysis_manifest_path,
                *(files[frame_id].filepath for frame_id in spec.golden_frame_ids),
            )
        ),
        "self_consistency": consistency,
        "golden_comparison": golden_comparison,
        "center_methods": method_metrics,
        "spot_removal": spot_removal,
        "explicit_refusals": refusals,
        "ransac_sensitivity": ransac,
        "recovery": _recovery_metrics(inputs),
    }


def _require_sources(spec: ReferenceSeriesSpec, files: dict[int, LeedFile]) -> None:
    missing_frames = [
        frame_id for frame_id in spec.golden_frame_ids if frame_id not in files
    ]
    if missing_frames:
        raise ReferenceSeriesError(
            f"{spec.name}: golden frames are absent from the session: {missing_frames}"
        )
    missing_paths = [
        str(files[frame_id].filepath)
        for frame_id in spec.golden_frame_ids
        if not files[frame_id].filepath.is_file()
    ]
    if missing_paths:
        raise ReferenceSeriesError(
            f"{spec.name}: golden source does not exist: {missing_paths[0]}"
        )


def _require_golden_oracles(
    spec: ReferenceSeriesSpec,
    tracks: list[Track],
    files: dict[int, LeedFile],
) -> None:
    fit_counts = {frame_id: 0 for frame_id in spec.golden_frame_ids}
    for track in tracks:
        for frame_id, result in getattr(track, "analysis_results", {}).items():
            if frame_id in fit_counts and result:
                fit_counts[frame_id] += 1
    for frame_id in spec.golden_frame_ids:
        if fit_counts[frame_id] < 3:
            raise ReferenceSeriesError(
                f"{spec.name}: golden frame {frame_id} needs at least three "
                "approved fit oracle values"
            )
        if not files[frame_id].pattern_center:
            raise ReferenceSeriesError(
                f"{spec.name}: golden frame {frame_id} lacks an approved center oracle"
            )


def _expected_counts(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["counts"]
        result = {
            key: int(counts[key])
            for key in (
                "frames",
                "tracks",
                "tracked_rois",
                "valid_fits",
                "invalid_fits",
            )
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as cause:
        raise ReferenceSeriesError(
            f"Invalid analysis manifest {path}: {cause}"
        ) from cause
    return result


def _validate_self_consistency(
    spec: ReferenceSeriesSpec,
    tracks: list[Track],
    files: dict[int, LeedFile],
    expected: dict[str, int],
) -> dict[str, Any]:
    tracked_rois = sum(len(track.spots_by_image_id) for track in tracks)
    results = [
        (track.track_id, frame_id, result)
        for track in tracks
        for frame_id, result in getattr(track, "analysis_results", {}).items()
    ]
    valid = [(track_id, frame_id, result) for track_id, frame_id, result in results if result]
    actual = {
        "frames": len(files),
        "tracks": len(tracks),
        "tracked_rois": tracked_rois,
        "valid_fits": len(valid),
        "invalid_fits": len(results) - len(valid),
    }
    if actual != expected:
        raise ReferenceSeriesError(
            f"{spec.name}: persisted counts differ: {actual!r} != {expected!r}"
        )
    residuals, unavailable_geometry = _radius_residuals(spec, valid, files)
    return {
        **actual,
        "geometry_unavailable_fits": unavailable_geometry,
        "geometry_comparable_fits": len(residuals),
        "radius_residual_max_px": max(residuals, default=0.0),
        "radius_residual_median_px": median(residuals) if residuals else 0.0,
    }


def _radius_residuals(
    spec: ReferenceSeriesSpec,
    valid: list[tuple[int, int, dict[str, Any]]],
    files: dict[int, LeedFile],
) -> tuple[list[float], int]:
    residuals = []
    unavailable = 0
    for _track_id, frame_id, result in valid:
        source = files.get(frame_id)
        center = None if source is None else source.pattern_center
        has_radius = result.get("radius") is not None
        if not center and not has_radius:
            unavailable += 1
            continue
        if not center or not has_radius:
            raise ReferenceSeriesError(
                f"{spec.name}: valid fit {frame_id} has one-sided geometry"
            )
        expected_radius = np.hypot(
            float(result["global_center_x"]) - float(center["center_x"]),
            float(result["global_center_y"]) - float(center["center_y"]),
        )
        residuals.append(abs(float(result["radius"]) - float(expected_radius)))
    if residuals and not np.allclose(residuals, 0.0, rtol=1e-12, atol=1e-9):
        raise ReferenceSeriesError(
            f"{spec.name}: persisted radius is not self-consistent"
        )
    return residuals, unavailable


def _select_goldens(
    spec: ReferenceSeriesSpec,
    tracks: list[Track],
    files: dict[int, LeedFile],
) -> tuple[list[Track], dict[int, LeedFile]]:
    selected = set(spec.golden_frame_ids)
    selected_tracks = [
        Track(
            track.track_id,
            {
                frame_id: spot
                for frame_id, spot in track.spots_by_image_id.items()
                if frame_id in selected
            },
        )
        for track in tracks
    ]
    selected_tracks = [track for track in selected_tracks if track.spots_by_image_id]
    return selected_tracks, {frame_id: files[frame_id] for frame_id in selected}


def _fit_inputs(
    tracks: list[Track],
    files: dict[int, LeedFile],
) -> tuple[FitObservationInput, ...]:
    inputs = []
    for frame_id in sorted(files):
        for track in tracks:
            spot = track.spots_by_image_id.get(frame_id)
            if spot is None:
                continue
            rect = spot.roi_rect
            source = files[frame_id]
            inputs.append(
                FitObservationInput(
                    track.track_id,
                    frame_id,
                    source.filepath,
                    Rect(rect.x(), rect.y(), rect.width(), rect.height()),
                    energy=source.energy,
                    temperature=source.temperature,
                )
            )
    return tuple(inputs)


def _compare_golden_fits(
    spec: ReferenceSeriesSpec,
    expected_tracks: list[Track],
    files: dict[int, LeedFile],
    fit_set: FitSet,
) -> dict[str, Any]:
    expected = {
        (track.track_id, frame_id): result
        for track in expected_tracks
        for frame_id, result in getattr(track, "analysis_results", {}).items()
        if frame_id in spec.golden_frame_ids and result
    }
    distances = []
    for observation in fit_set.observations:
        target = expected.get((observation.track_id, observation.frame_id))
        if target is None:
            continue
        if observation.result is None:
            raise ReferenceSeriesError(
                f"{spec.name}: approved fit became unavailable for "
                f"{observation.track_id}/{observation.frame_id}"
            )
        distances.append(
            float(
                np.hypot(
                    observation.result.global_center_x - target["global_center_x"],
                    observation.result.global_center_y - target["global_center_y"],
                )
            )
        )
    if len(distances) != len(expected):
        raise ReferenceSeriesError(f"{spec.name}: not every approved fit was rerun")
    if distances and max(distances) > GOLDEN_CENTER_TOLERANCE_PX:
        raise ReferenceSeriesError(
            f"{spec.name}: golden fit center error exceeds "
            f"{GOLDEN_CENTER_TOLERANCE_PX:.1f} px"
        )
    return {
        "frame_count": len(spec.golden_frame_ids),
        "fit_count": len(distances),
        "fit_center_error_max_px": max(distances, default=0.0),
        "fit_center_error_median_px": median(distances) if distances else 0.0,
        "tolerance_px": GOLDEN_CENTER_TOLERANCE_PX,
        "input_file_count": len(files),
    }


def _center_metrics(
    geometry: GeometryResult,
    files: dict[int, LeedFile],
) -> dict[str, Any]:
    errors = []
    errors_by_frame = {}
    ordered_centers = []
    for frame in geometry.frames:
        center = frame.center
        if center is None:
            continue
        expected = files[frame.frame_id].pattern_center
        if expected:
            error = float(
                np.hypot(
                    center.center_x - float(expected["center_x"]),
                    center.center_y - float(expected["center_y"]),
                )
            )
            errors.append(error)
            errors_by_frame[frame.frame_id] = error
        ordered_centers.append((frame.frame_id, center.center_x, center.center_y))
    jumps = [
        float(np.hypot(current[1] - previous[1], current[2] - previous[2]))
        for previous, current in zip(ordered_centers, ordered_centers[1:])
    ]
    available_count = len(ordered_centers)
    return {
        "available_frames": available_count,
        "unavailable_frames": len(geometry.frames) - available_count,
        "unavailable_frame_ids": [
            frame.frame_id for frame in geometry.frames if frame.center is None
        ],
        "error_count": len(errors),
        "error_by_frame_px": errors_by_frame,
        "error_max_px": max(errors, default=None),
        "error_median_px": median(errors) if errors else None,
        "temporal_jump_count": len(jumps),
        "temporal_jump_max_px": max(jumps, default=0.0),
        "status": geometry.status.value,
    }


def _spot_removal_metrics(
    fit_set: FitSet,
    track_ids: tuple[int, ...],
    files: dict[int, LeedFile],
) -> dict[str, Any]:
    baseline = derive_geometry(
        fit_set,
        GeometryRequest(GeometryMethod.CIRCLE_FIT, track_ids),
    )
    removed_track = track_ids[0]
    reduced = FitSet(
        tuple(item for item in fit_set.observations if item.track_id != removed_track)
    )
    after = derive_geometry(
        reduced,
        GeometryRequest(GeometryMethod.CIRCLE_FIT, track_ids[1:]),
    )
    baseline_centers = {
        item.frame_id: item.center for item in baseline.frames if item.center is not None
    }
    shifts = [
        float(
            np.hypot(
                frame.center.center_x - baseline_centers[frame.frame_id].center_x,
                frame.center.center_y - baseline_centers[frame.frame_id].center_y,
            )
        )
        for frame in after.frames
        if frame.center is not None and frame.frame_id in baseline_centers
    ]
    return {
        "removed_track_id": removed_track,
        "available_frames": len(shifts),
        "center_shift_max_px": max(shifts, default=None),
        "center_shift_median_px": median(shifts) if shifts else None,
        "reference_frame_count": len(files),
    }


def _refusal_metrics(fit_set: FitSet, track_ids: tuple[int, ...]) -> dict[str, int]:
    insufficient = FitSet(
        tuple(item for item in fit_set.observations if item.track_id in track_ids[:2])
    )
    result = derive_geometry(
        insufficient,
        GeometryRequest(GeometryMethod.CIRCLE_FIT, track_ids[:2]),
    )
    refused = sum(
        frame.status is GeometryStatus.GEOMETRY_UNAVAILABLE for frame in result.frames
    )
    if refused != len(result.frames):
        raise ReferenceSeriesError("Insufficient center points were not refused")
    return {"insufficient_points": refused}


def _ransac_metrics(fit_set: FitSet, track_ids: tuple[int, ...]) -> dict[str, Any]:
    by_seed = []
    for seed in RANSAC_SEEDS:
        result = derive_geometry(
            fit_set,
            GeometryRequest(
                GeometryMethod.PER_FRAME,
                track_ids,
                GeometryParameters(random_seed=seed),
            ),
        )
        by_seed.append(
            {
                frame.frame_id: (frame.center.center_x, frame.center.center_y)
                for frame in result.frames
                if frame.center is not None
            }
        )
    spreads = []
    for frame_id in sorted(set.intersection(*(set(item) for item in by_seed))):
        centers = by_seed[0][frame_id], by_seed[1][frame_id], by_seed[2][frame_id], by_seed[3][frame_id]
        spreads.append(
            max(
                float(np.hypot(left[0] - right[0], left[1] - right[1]))
                for left in centers
                for right in centers
            )
        )
    return {
        "seeds": list(RANSAC_SEEDS),
        "common_frame_count": len(spreads),
        "max_seed_spread_px": max(spreads, default=None),
        "median_seed_spread_px": median(spreads) if spreads else None,
    }


def _recovery_metrics(inputs: tuple[FitObservationInput, ...]) -> dict[str, Any]:
    if FitRecoveryConfig().enabled:
        raise ReferenceSeriesError("Recovery unexpectedly became enabled by default")
    recovered = fit(
        FitRequest(inputs, recovery=FitRecoveryConfig(enabled=True)),
    )
    statuses = [item.provenance.recovery.status for item in recovered.observations]
    counts = {
        status.value: sum(item is status for item in statuses)
        for status in (
            RecoveryStatus.RECOVERED,
            RecoveryStatus.FAILED,
            RecoveryStatus.AMBIGUOUS,
            RecoveryStatus.NOT_NEEDED,
        )
    }
    return {
        "default_enabled": False,
        "recommended_default": False,
        "counts": counts,
    }


def _enforce_acceptance(
    spec: ReferenceSeriesSpec,
    methods: dict[str, dict[str, Any]],
    spot_removal: dict[str, Any],
    ransac: dict[str, Any],
) -> None:
    criteria = spec.criteria
    for method, metrics in methods.items():
        if metrics["available_frames"] < criteria.minimum_method_available_frames:
            raise ReferenceSeriesError(
                f"{spec.name}/{method}: too few available center observations"
            )
        if (
            metrics["error_count"] < criteria.minimum_method_available_frames
            or metrics["error_max_px"] > criteria.max_center_error_px
        ):
            worst_frame = max(
                metrics["error_by_frame_px"],
                key=metrics["error_by_frame_px"].get,
                default=None,
            )
            raise ReferenceSeriesError(
                f"{spec.name}/{method}: center error {metrics['error_max_px']!r} "
                f"at frame {worst_frame!r} exceeds approved "
                f"{criteria.max_center_error_px!r} px"
            )
        if (
            metrics["temporal_jump_count"] < criteria.minimum_temporal_pairs
            or metrics["temporal_jump_max_px"] > criteria.max_temporal_jump_px
        ):
            raise ReferenceSeriesError(
                f"{spec.name}/{method}: temporal jump "
                f"{metrics['temporal_jump_max_px']!r} exceeds approved "
                f"{criteria.max_temporal_jump_px!r} px"
            )
    removal_shift = spot_removal["center_shift_max_px"]
    if (
        spot_removal["available_frames"] < criteria.minimum_method_available_frames
        or removal_shift is None
        or removal_shift > criteria.max_spot_removal_shift_px
    ):
        raise ReferenceSeriesError(
            f"{spec.name}: approved spot-removal stability criterion failed"
        )
    seed_spread = ransac["max_seed_spread_px"]
    if (
        ransac["common_frame_count"] < criteria.minimum_method_available_frames
        or seed_spread is None
        or seed_spread > criteria.max_ransac_seed_spread_px
    ):
        raise ReferenceSeriesError(
            f"{spec.name}: approved RANSAC sensitivity criterion failed"
        )
