#!/usr/bin/env python3
"""Run ToMP / TaMOs offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_windows_conda_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    env_root = Path(sys.executable).resolve().parent
    dll_dirs = [
        env_root,
        env_root / "Library" / "bin",
        env_root / "DLLs",
        env_root / "Scripts",
        env_root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    existing = [str(path) for path in dll_dirs if path.exists()]
    current_path = os.environ.get("PATH", "")
    if existing:
        os.environ["PATH"] = os.pathsep.join(existing + ([current_path] if current_path else []))
    for directory in existing:
        try:
            os.add_dll_directory(directory)
        except (FileNotFoundError, OSError):
            continue


_bootstrap_windows_conda_dlls()

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.tomp_backend import (
    TompRuntimeSettings,
    _run_tomp_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ToMP / TaMOs in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the ToMP / TaMOs checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local PyTracking repo.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--model-variant",
        default="tomp50",
        help="Tracker variant: tomp50 or tamos_resnet50.",
    )
    parser.add_argument(
        "--representative-point-mode",
        default="bbox_center",
        help="How to convert a tracked bbox into one representative point.",
    )
    parser.add_argument("--height", type=int, default=None, help="Optional inference height.")
    parser.add_argument("--width", type=int, default=None, help="Optional inference width.")
    parser.add_argument(
        "--initial-bbox-padding-px",
        type=float,
        default=0.0,
        help="Padding added around roi_rect before building the initial bbox.",
    )
    parser.add_argument(
        "--use-presence-score-gating",
        action="store_true",
        help="Enable object-presence score gating inside the direct backend.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.0,
        help="Minimum object-presence score if gating is enabled.",
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

    result = _run_tomp_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        settings=TompRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            device=args.device,
            backend_mode="direct",
            model_variant=args.model_variant,
            representative_point_mode=args.representative_point_mode,
            inference_resolution=inference_resolution,
            initial_bbox_padding_px=float(args.initial_bbox_padding_px),
            use_presence_score_gating=bool(args.use_presence_score_gating),
            presence_threshold=float(args.presence_threshold),
            subprocess_timeout_sec=float(args.subprocess_timeout_sec),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        tracked_bboxes_xywh=result.tracked_bboxes_xywh.astype(np.float32, copy=False),
        tracked_bboxes_xyxy=result.tracked_bboxes_xyxy.astype(np.float32, copy=False),
        bbox_scores=result.bbox_scores.astype(np.float32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        model_variant=np.asarray(result.model_variant),
        representative_point_mode=np.asarray(result.representative_point_mode),
        initial_bbox_padding_px=np.asarray(result.initial_bbox_padding_px, dtype=np.float32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
