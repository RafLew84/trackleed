"""Native-geometry reduction and replay checks for publication artifacts."""

from __future__ import annotations

import numpy as np

from trackleed.analysis.dam4sam_backend import (
    _mask_to_bbox_xyxy,
    _mask_to_representative_point,
)
from trackleed.analysis.ostrack_backend import (
    _boxes_xywh_to_xyxy,
    _representative_point_from_bbox_xywh,
)


ORIGINAL_TO_CACHE = 0.256
CENTER_TOLERANCE_CACHE_PX = 5e-4


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _require_matching_center(
    actual_original_px: np.ndarray,
    expected_original_px: np.ndarray,
    *,
    geometry_name: str,
) -> float:
    error_original_px = float(
        np.linalg.norm(
            np.asarray(actual_original_px, dtype=np.float64)
            - np.asarray(expected_original_px, dtype=np.float64)
        )
    )
    tolerance_original_px = CENTER_TOLERANCE_CACHE_PX / ORIGINAL_TO_CACHE
    if error_original_px > tolerance_original_px:
        raise ValueError(
            f"{geometry_name} does not reproduce the archived reduced center: "
            f"error {error_original_px:.9f} original px exceeds "
            f"{tolerance_original_px:.9f}."
        )
    return error_original_px


def summarize_ostrack_bbox(
    bbox_xywh_cache_px: np.ndarray,
    *,
    expected_center_original_px: np.ndarray,
) -> dict[str, object]:
    """Preserve an OSTrack bbox and verify its frozen center reduction."""
    bbox = np.asarray(bbox_xywh_cache_px, dtype=np.float64).reshape(-1)
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or bbox[2] <= 0.0
        or bbox[3] <= 0.0
    ):
        raise ValueError("OSTrack native bbox must be finite with positive size")
    bbox_xyxy = np.asarray(
        _boxes_xywh_to_xyxy(bbox[np.newaxis, :])[0],
        dtype=np.float64,
    )
    center = _representative_point_from_bbox_xywh(
        bbox,
        frame=np.zeros((1, 1), dtype=np.uint8),
        mode="bbox_center",
    )
    if center is None:
        raise ValueError("OSTrack bbox center reduction failed")
    center_cache = np.asarray(center, dtype=np.float64)
    center_original = center_cache / ORIGINAL_TO_CACHE
    center_error = _require_matching_center(
        center_original,
        expected_center_original_px,
        geometry_name="OSTrack bbox",
    )
    return {
        "geometry_type": "bounding_box",
        "coordinate_convention": "xywh and half-open xyxy",
        "bbox_xywh_cache_px": _float_list(bbox),
        "bbox_xyxy_cache_px": _float_list(bbox_xyxy),
        "bbox_xywh_original_px": _float_list(bbox / ORIGINAL_TO_CACHE),
        "bbox_xyxy_original_px": _float_list(
            bbox_xyxy / ORIGINAL_TO_CACHE
        ),
        "reduction_rule": "x+w/2, y+h/2 from native bbox xywh",
        "reduced_center_cache_px": _float_list(center_cache),
        "reduced_center_original_px": _float_list(center_original),
        "archived_reduced_center_original_px": _float_list(
            expected_center_original_px
        ),
        "center_consistency_error_original_px": center_error,
    }


def summarize_dam4sam_mask(
    mask_cache: np.ndarray,
    *,
    expected_center_original_px: np.ndarray,
) -> dict[str, object]:
    """Preserve a DAM4SAM mask and verify its frozen centroid reduction."""
    mask = np.asarray(mask_cache, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"DAM4SAM mask must be two-dimensional: {mask.shape}")
    bbox = _mask_to_bbox_xyxy(mask)
    if bbox is None:
        raise ValueError("DAM4SAM native mask is empty")
    bbox_xyxy = np.asarray(bbox, dtype=np.float64)
    bbox_xywh = np.asarray(
        [
            bbox_xyxy[0],
            bbox_xyxy[1],
            bbox_xyxy[2] - bbox_xyxy[0],
            bbox_xyxy[3] - bbox_xyxy[1],
        ],
        dtype=np.float64,
    )
    centroid = _mask_to_representative_point(
        mask,
        frame_gray=np.zeros(mask.shape, dtype=np.uint8),
        mode="mask_centroid",
    )
    if centroid is None:
        raise ValueError("DAM4SAM mask centroid reduction failed")
    centroid_cache = np.asarray(centroid, dtype=np.float64)
    centroid_original = centroid_cache / ORIGINAL_TO_CACHE
    center_error = _require_matching_center(
        centroid_original,
        expected_center_original_px,
        geometry_name="DAM4SAM mask centroid",
    )
    return {
        "geometry_type": "binary_mask",
        "coordinate_convention": (
            "mask[y,x] on the 256x256 tracking cache; bbox is half-open xyxy"
        ),
        "mask_shape_cache_px": [int(mask.shape[0]), int(mask.shape[1])],
        "mask_area_cache_px": int(np.count_nonzero(mask)),
        "mask_bbox_xyxy_cache_px": _float_list(bbox_xyxy),
        "mask_bbox_xywh_cache_px": _float_list(bbox_xywh),
        "mask_bbox_xyxy_original_px": _float_list(
            bbox_xyxy / ORIGINAL_TO_CACHE
        ),
        "mask_bbox_xywh_original_px": _float_list(
            bbox_xywh / ORIGINAL_TO_CACHE
        ),
        "reduction_rule": (
            "arithmetic mean of x indices and y indices of all true mask pixels"
        ),
        "reduced_centroid_cache_px": _float_list(centroid_cache),
        "reduced_centroid_original_px": _float_list(centroid_original),
        "archived_reduced_center_original_px": _float_list(
            expected_center_original_px
        ),
        "center_consistency_error_original_px": center_error,
    }


def compare_reduced_trajectories(
    actual_tracks_xy_cache: np.ndarray,
    actual_available: np.ndarray,
    expected_tracks_xy_cache: np.ndarray,
    expected_available: np.ndarray,
    *,
    method: str,
) -> dict[str, object]:
    """Compare a replayed reduced trajectory with its frozen publication run."""
    actual_tracks = np.asarray(actual_tracks_xy_cache, dtype=np.float64)
    actual_status = np.asarray(actual_available, dtype=bool)
    expected_tracks = np.asarray(expected_tracks_xy_cache, dtype=np.float64)
    expected_status = np.asarray(expected_available, dtype=bool)
    if actual_tracks.shape != expected_tracks.shape:
        raise ValueError(
            f"{method} replay trajectory shape {actual_tracks.shape} differs "
            f"from archived shape {expected_tracks.shape}"
        )
    if actual_status.shape != expected_status.shape:
        raise ValueError(
            f"{method} replay availability shape {actual_status.shape} differs "
            f"from archived shape {expected_status.shape}"
        )
    if actual_tracks.ndim != 2 or actual_tracks.shape[1] != 2:
        raise ValueError(
            f"{method} replay must have trajectory shape [frames,2]"
        )
    if not np.array_equal(actual_status, expected_status):
        mismatch_count = int(
            np.count_nonzero(actual_status != expected_status)
        )
        raise ValueError(
            f"{method} replay availability differs in {mismatch_count} frames"
        )
    distances = np.linalg.norm(
        actual_tracks[expected_status] - expected_tracks[expected_status],
        axis=-1,
    )
    maximum_error = float(np.max(distances)) if distances.size else 0.0
    if maximum_error > CENTER_TOLERANCE_CACHE_PX:
        raise ValueError(
            f"{method} replay trajectory differs from the archived reduction: "
            f"{maximum_error:.9f} cache px"
        )
    return {
        "availability_matches_all_frames": True,
        "reduced_center_matches_all_available_frames": True,
        "available_frame_count": int(np.count_nonzero(expected_status)),
        "maximum_reduced_center_error_cache_px": maximum_error,
        "tolerance_cache_px": CENTER_TOLERANCE_CACHE_PX,
    }
