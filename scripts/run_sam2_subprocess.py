#!/usr/bin/env python3
"""Run SAM 2 offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.sam2_backend import (
    Sam2RuntimeSettings,
    _run_sam2_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM 2 in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the SAM 2 checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local SAM 2 repo.")
    parser.add_argument("--config", default=None, help="Optional SAM 2 config path.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--prompt-mode", default="point+bbox", help="SAM 2 prompt mode.")
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
        help="Padding added when Trackleed builds a seed bbox for SAM 2.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to SAM 2 mask logits.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=float,
        default=1.0,
        help="Reject masks smaller than this pixel area.",
    )
    parser.add_argument(
        "--max-mask-area",
        type=float,
        default=None,
        help="Optional upper mask-area bound.",
    )
    parser.add_argument(
        "--use-vos-optimized",
        action="store_true",
        help="Enable SAM 2 vos_optimized mode.",
    )
    parser.add_argument(
        "--apply-postprocessing",
        dest="apply_postprocessing",
        action="store_true",
        help="Enable SAM 2 postprocessing.",
    )
    parser.add_argument(
        "--disable-postprocessing",
        dest="apply_postprocessing",
        action="store_false",
        help="Disable SAM 2 postprocessing.",
    )
    parser.set_defaults(apply_postprocessing=True)
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Store video frames on CPU inside the SAM 2 state.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Move more of the SAM 2 internal state to CPU memory.",
    )
    parser.add_argument(
        "--async-loading-frames",
        action="store_true",
        help="Enable async frame loading when supported by the runtime.",
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

    inference_resolution = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Both --height and --width must be provided together.")
        inference_resolution = (int(args.height), int(args.width))

    result = _run_sam2_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        settings=Sam2RuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            config_path=args.config,
            device=args.device,
            backend_mode="direct",
            prompt_mode=args.prompt_mode,
            representative_point_mode=args.representative_point_mode,
            inference_resolution=inference_resolution,
            initial_bbox_padding_px=float(args.initial_bbox_padding_px),
            mask_threshold=float(args.mask_threshold),
            min_mask_area=float(args.min_mask_area),
            max_mask_area=None if args.max_mask_area is None else float(args.max_mask_area),
            use_vos_optimized=bool(args.use_vos_optimized),
            apply_postprocessing=bool(args.apply_postprocessing),
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
            async_loading_frames=bool(args.async_loading_frames),
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
        mask_component_counts=result.mask_component_counts.astype(np.int32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        config_name=np.asarray(result.config_name),
        prompt_mode=np.asarray(result.prompt_mode),
        representative_point_mode=np.asarray(result.representative_point_mode),
        use_vos_optimized=np.asarray(bool(result.use_vos_optimized)),
        backend_mode=np.asarray(result.backend_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
