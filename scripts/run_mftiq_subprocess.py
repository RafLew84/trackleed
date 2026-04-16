#!/usr/bin/env python3
"""Run MFTIQ offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.mftiq_backend import (
    MftiqRuntimeSettings,
    _run_mftiq_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MFTIQ in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to UOM checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local MFTIQ repo.")
    parser.add_argument("--config", default=None, help="Optional MFTIQ config path.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--height", type=int, default=None, help="Optional inference height.")
    parser.add_argument("--width", type=int, default=None, help="Optional inference width.")
    parser.add_argument(
        "--occlusion-threshold",
        type=float,
        default=0.5,
        help="Threshold used to derive visible_mask from occlusion scores.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        query_points_tyx = np.asarray(data["query_points_tyx"], dtype=np.float32)

    inference_resolution = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("Both --height and --width must be provided together.")
        inference_resolution = (int(args.height), int(args.width))

    result = _run_mftiq_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        settings=MftiqRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            config_path=args.config,
            device=args.device,
            backend_mode="direct",
            inference_resolution=inference_resolution,
            occlusion_threshold=float(args.occlusion_threshold),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        occlusion_scores=result.occlusion_scores.astype(np.float32, copy=False),
        sigma_scores=result.sigma_scores.astype(np.float32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        config_name=np.asarray(result.config_name),
        backend_mode=np.asarray(result.backend_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
