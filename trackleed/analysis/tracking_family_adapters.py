from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from trackleed.analysis.tracking_filter_utils import (
    DirectionFilterContext,
    build_direction_filter_context,
    read_basic_motion_filter_params,
    read_bbox_quality_filter_params,
    read_mask_quality_filter_params,
)


@dataclass(frozen=True)
class FamilyAdapterContext:
    keep_invisible_as_gaps: bool
    roi_size: int
    window_size: int
    window_overlap: int
    motion_filter_params: dict[str, Any]
    direction_filter: DirectionFilterContext


@dataclass(frozen=True)
class PointFamilyAdapterContext(FamilyAdapterContext):
    recenter_mode: str
    centroid_patch_multiplier: float
    centroid_rel_threshold: float
    centroid_max_expand: int
    centroid_max_shift_ratio: float


@dataclass(frozen=True)
class MaskFamilyAdapterContext(FamilyAdapterContext):
    quality_filter_params: dict[str, Any]


@dataclass(frozen=True)
class BBoxFamilyAdapterContext(FamilyAdapterContext):
    quality_filter_params: dict[str, Any]


def build_point_family_adapter_context(
    params: Mapping[str, Any],
    *,
    all_files_map: Mapping[int, Any],
    subset_frame_ids: Sequence[int],
    anchor_filepath: str | None,
    estimate_pattern_center_fn: Callable[..., Mapping[str, Any] | None],
) -> PointFamilyAdapterContext:
    return PointFamilyAdapterContext(
        keep_invisible_as_gaps=bool(params.get("keep_invisible_points_as_gaps", True)),
        roi_size=int(params.get("roi_size", 32)),
        window_size=int(params.get("window_size", 64)),
        window_overlap=int(params.get("window_overlap", 8)),
        motion_filter_params=read_basic_motion_filter_params(params),
        direction_filter=build_direction_filter_context(
            params,
            all_files_map=all_files_map,
            subset_frame_ids=subset_frame_ids,
            anchor_filepath=anchor_filepath,
            estimate_pattern_center_fn=estimate_pattern_center_fn,
        ),
        recenter_mode=str(
            params.get(
                "recenter_mode",
                "centroid" if bool(params.get("use_centroid_refinement", True)) else "none",
            )
        ).strip().lower(),
        centroid_patch_multiplier=float(
            params.get("recenter_patch_multiplier", params.get("centroid_patch_multiplier", 1.25))
        ),
        centroid_rel_threshold=float(params.get("centroid_rel_threshold", 0.35)),
        centroid_max_expand=int(max(0, params.get("centroid_max_expand", 1))),
        centroid_max_shift_ratio=float(params.get("centroid_max_shift_ratio", 0.45)),
    )


def build_mask_family_adapter_context(
    params: Mapping[str, Any],
    *,
    all_files_map: Mapping[int, Any],
    subset_frame_ids: Sequence[int],
    anchor_filepath: str | None,
    estimate_pattern_center_fn: Callable[..., Mapping[str, Any] | None],
) -> MaskFamilyAdapterContext:
    return MaskFamilyAdapterContext(
        keep_invisible_as_gaps=bool(params.get("keep_invisible_points_as_gaps", True)),
        roi_size=int(params.get("roi_size", 32)),
        window_size=int(params.get("window_size", 64)),
        window_overlap=int(params.get("window_overlap", 8)),
        motion_filter_params=read_basic_motion_filter_params(params),
        direction_filter=build_direction_filter_context(
            params,
            all_files_map=all_files_map,
            subset_frame_ids=subset_frame_ids,
            anchor_filepath=anchor_filepath,
            estimate_pattern_center_fn=estimate_pattern_center_fn,
        ),
        quality_filter_params=read_mask_quality_filter_params(params),
    )


def build_bbox_family_adapter_context(
    params: Mapping[str, Any],
    *,
    all_files_map: Mapping[int, Any],
    subset_frame_ids: Sequence[int],
    anchor_filepath: str | None,
    estimate_pattern_center_fn: Callable[..., Mapping[str, Any] | None],
) -> BBoxFamilyAdapterContext:
    return BBoxFamilyAdapterContext(
        keep_invisible_as_gaps=bool(params.get("keep_invisible_points_as_gaps", True)),
        roi_size=int(params.get("roi_size", 32)),
        window_size=int(params.get("window_size", 64)),
        window_overlap=int(params.get("window_overlap", 8)),
        motion_filter_params=read_basic_motion_filter_params(params),
        direction_filter=build_direction_filter_context(
            params,
            all_files_map=all_files_map,
            subset_frame_ids=subset_frame_ids,
            anchor_filepath=anchor_filepath,
            estimate_pattern_center_fn=estimate_pattern_center_fn,
        ),
        quality_filter_params=read_bbox_quality_filter_params(params),
    )
