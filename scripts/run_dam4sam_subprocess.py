#!/usr/bin/env python3
"""Run DAM4SAM offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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

from trackleed.analysis.dam4sam_backend import (  # noqa: E402
    Dam4SamRuntimeSettings,
    _run_dam4sam_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DAM4SAM in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and prompts.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the DAM4SAM/SAM2 checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local DAM4SAM repo.")
    parser.add_argument("--tracker-name", default="sam21pp-L", help="DAM4SAM tracker variant.")
    parser.add_argument("--device", default="auto", help="Torch device; DAM4SAM is CUDA-only.")
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
        help="Padding added when building a seed bbox from a point prompt.",
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

    inference_resolution = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Both --height and --width must be provided together.")
        inference_resolution = (int(args.height), int(args.width))

    result = _run_dam4sam_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        query_boxes_xyxy=query_boxes_xyxy,
        query_initial_masks=query_initial_masks,
        settings=Dam4SamRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            backend_mode="direct",
            tracker_name=args.tracker_name,
            device=args.device,
            representative_point_mode=args.representative_point_mode,
            inference_resolution=inference_resolution,
            initial_bbox_padding_px=float(args.initial_bbox_padding_px),
            subprocess_timeout_sec=float(args.subprocess_timeout_sec),
        ),
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
        mask_bboxes_xywh=result.mask_bboxes_xywh.astype(np.float32, copy=False),
        mask_scores=result.mask_scores.astype(np.float32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        tracker_name=np.asarray(result.tracker_name),
        config_name=np.asarray(result.config_name),
        representative_point_mode=np.asarray(result.representative_point_mode),
        backend_mode=np.asarray(result.backend_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
