#!/usr/bin/env python3
"""Run DEVA seeded VOS inference in a dedicated interpreter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.deva_backend import (  # noqa: E402
    DevaRuntimeSettings,
    _run_deva_eval_vos_offline,
    _build_deva_initial_label_mask,
    _normalize_query_boxes_xyxy,
    _normalize_query_initial_masks,
    _normalize_query_object_ids,
    _normalize_query_points_tyx,
    _prepare_frames_rgb,
    _normalize_representative_point_mode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DEVA in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and prompts.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the DEVA checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local DEVA repo.")
    parser.add_argument("--device", default="auto", help="Torch device; DEVA eval_vos is CUDA-only.")
    parser.add_argument(
        "--representative-point-mode",
        default="mask_centroid",
        help="How to convert a propagated mask into one representative point.",
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=480,
        help="DEVA eval_vos shortest-side size. Use -1 for original resolution.",
    )
    parser.add_argument(
        "--initial-bbox-padding-px",
        type=float,
        default=0.0,
        help="Padding added when converting bbox prompts into seed masks.",
    )
    parser.add_argument(
        "--initial-point-radius-px",
        type=float,
        default=4.0,
        help="Disk radius used when converting point prompts into seed masks.",
    )
    parser.add_argument("--chunk-size", type=int, default=4, help="DEVA chunk_size.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--subprocess-timeout-sec",
        type=float,
        default=1800.0,
        help="Maximum time allowed for the internal DEVA eval_vos subprocess.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()

    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        has_query_points = (
            bool(data["has_query_points"].tolist()) if "has_query_points" in data else False
        )
        query_points_tyx = (
            np.asarray(data["query_points_tyx"], dtype=np.float32)
            if has_query_points
            else None
        )
        has_query_boxes = bool(data["has_query_boxes"].tolist()) if "has_query_boxes" in data else False
        query_boxes_xyxy = (
            np.asarray(data["query_boxes_xyxy"], dtype=np.float32)
            if has_query_boxes
            else None
        )
        has_query_object_ids = (
            bool(data["has_query_object_ids"].tolist())
            if "has_query_object_ids" in data
            else False
        )
        query_object_ids = (
            np.asarray(data["query_object_ids"], dtype=np.int32)
            if has_query_object_ids
            else None
        )
        has_query_initial_masks = (
            bool(data["has_query_initial_masks"].tolist())
            if "has_query_initial_masks" in data
            else False
        )
        query_initial_masks = (
            np.asarray(data["query_initial_masks"], dtype=bool)
            if has_query_initial_masks
            else None
        )

    frames_rgb, original_resolution = _prepare_frames_rgb(frames)
    num_queries = _infer_num_queries_for_runner(
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
    )
    query_points_tyx = _normalize_query_points_tyx(query_points_tyx, num_queries)
    query_boxes_xyxy = _normalize_query_boxes_xyxy(query_boxes_xyxy, num_queries)
    query_initial_masks = _normalize_query_initial_masks(
        query_initial_masks,
        num_queries=num_queries,
        frame_shape=original_resolution,
    )
    object_ids = _normalize_query_object_ids(query_object_ids, num_queries)
    representative_point_mode = _normalize_representative_point_mode(
        args.representative_point_mode
    )
    initial_label_mask, initial_mask_mode = _build_deva_initial_label_mask(
        frame_shape=original_resolution,
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
        object_ids=object_ids,
        initial_bbox_padding_px=float(args.initial_bbox_padding_px),
        initial_point_radius_px=float(args.initial_point_radius_px),
    )

    result = _run_deva_eval_vos_offline(
        frames_rgb,
        initial_label_mask,
        DevaRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            device=args.device,
            backend_mode="direct",
            representative_point_mode=representative_point_mode,
            inference_size=int(args.inference_size),
            initial_bbox_padding_px=float(args.initial_bbox_padding_px),
            initial_point_radius_px=float(args.initial_point_radius_px),
            amp=not bool(args.no_amp),
            chunk_size=int(args.chunk_size),
            subprocess_timeout_sec=float(args.subprocess_timeout_sec),
        ),
        representative_point_mode=representative_point_mode,
        initial_mask_mode=initial_mask_mode,
        backend_mode="subprocess",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        masks=result.masks.astype(bool, copy=False),
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        mask_areas=result.mask_areas.astype(np.float32, copy=False),
        mask_bboxes_xyxy=result.mask_bboxes_xyxy.astype(np.float32, copy=False),
        mask_scores=result.mask_scores.astype(np.float32, copy=False),
        inference_size=np.asarray(result.inference_size, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        representative_point_mode=np.asarray(result.representative_point_mode),
        initial_mask_mode=np.asarray(result.initial_mask_mode),
        backend_mode=np.asarray(result.backend_mode),
    )
    return 0


def _infer_num_queries_for_runner(
    *,
    query_points_tyx: np.ndarray | None,
    query_boxes_xyxy: np.ndarray | None,
    query_initial_masks: np.ndarray | None,
) -> int:
    candidates: list[int] = []
    if query_points_tyx is not None:
        candidates.append(int(np.asarray(query_points_tyx).shape[0]))
    if query_boxes_xyxy is not None:
        candidates.append(int(np.asarray(query_boxes_xyxy).shape[0]))
    if query_initial_masks is not None:
        candidates.append(int(np.asarray(query_initial_masks).shape[0]))
    if not candidates:
        raise ValueError("DEVA requires at least one query point, bbox, or initial mask.")
    if len(set(candidates)) != 1:
        raise ValueError(f"DEVA query inputs disagree on object count: {candidates}.")
    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
