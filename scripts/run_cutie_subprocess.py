#!/usr/bin/env python3
"""Run Cutie offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.cutie_backend import (  # noqa: E402
    CutieRuntimeSettings,
    _run_cutie_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Cutie in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Cutie checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local Cutie repo.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--representative-point-mode",
        default="mask_centroid",
        help="How to convert a propagated mask into one representative point.",
    )
    parser.add_argument("--height", type=int, default=None, help="Optional inference height.")
    parser.add_argument("--width", type=int, default=None, help="Optional inference width.")
    parser.add_argument(
        "--initial-bbox-padding-px",
        type=float,
        default=12.0,
        help="Padding added when Trackleed builds a seed bbox for Cutie.",
    )
    parser.add_argument(
        "--initial-mask-mode",
        default="threshold",
        help="How to convert the ROI bbox into an initial Cutie mask.",
    )
    parser.add_argument(
        "--initial-mask-threshold-rel",
        type=float,
        default=0.7,
        help="Relative threshold used by the threshold-based initial mask builder.",
    )
    parser.add_argument(
        "--initial-mask-disk-radius-ratio",
        type=float,
        default=0.35,
        help="Disk radius relative to the smaller bbox dimension.",
    )
    parser.add_argument(
        "--mem-every",
        type=int,
        default=None,
        help="Optional Cutie memory stride override.",
    )
    parser.add_argument(
        "--subprocess-timeout-sec",
        type=float,
        default=1800.0,
        help="Stored for parity with the outer wrapper; not used internally.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        query_points_tyx = np.asarray(data["query_points_tyx"], dtype=np.float32)
        has_query_boxes = bool(data["has_query_boxes"].tolist()) if "has_query_boxes" in data else False
        query_boxes_xyxy = None
        if has_query_boxes and "query_boxes_xyxy" in data:
            query_boxes_xyxy = np.asarray(data["query_boxes_xyxy"], dtype=np.float32)
        has_query_initial_masks = (
            bool(data["has_query_initial_masks"].tolist())
            if "has_query_initial_masks" in data
            else False
        )
        query_initial_masks = None
        if has_query_initial_masks and "query_initial_masks" in data:
            query_initial_masks = np.asarray(data["query_initial_masks"], dtype=bool)

    inference_resolution = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Both --height and --width must be provided together.")
        inference_resolution = (int(args.height), int(args.width))

    result = _run_cutie_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
        settings=CutieRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            device=args.device,
            backend_mode="direct",
            representative_point_mode=args.representative_point_mode,
            inference_resolution=inference_resolution,
            initial_bbox_padding_px=float(args.initial_bbox_padding_px),
            initial_mask_mode=args.initial_mask_mode,
            initial_mask_threshold_rel=float(args.initial_mask_threshold_rel),
            initial_mask_disk_radius_ratio=float(args.initial_mask_disk_radius_ratio),
            mem_every=None if args.mem_every is None else int(args.mem_every),
            subprocess_timeout_sec=float(args.subprocess_timeout_sec),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        mask_areas=result.mask_areas.astype(np.float32, copy=False),
        mask_bboxes_xyxy=result.mask_bboxes_xyxy.astype(np.float32, copy=False),
        mask_scores=result.mask_scores.astype(np.float32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        representative_point_mode=np.asarray(result.representative_point_mode),
        initial_mask_mode=np.asarray(result.initial_mask_mode),
        backend_mode=np.asarray(result.backend_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
