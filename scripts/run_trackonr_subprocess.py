#!/usr/bin/env python3
"""Run Track-On-R offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.trackonr_backend import (
    TrackonrRuntimeSettings,
    _run_trackonr_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track-On-R in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to Track-On-R checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local track_on repo.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--height", type=int, required=True, help="Inference height.")
    parser.add_argument("--width", type=int, required=True, help="Inference width.")
    parser.add_argument("--support-grid-size", type=int, default=20, help="Track-On-R support grid size.")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.8,
        help="Threshold used to derive visible_mask from Track-On-R visibility.",
    )
    parser.add_argument(
        "--disable-visibility-gating",
        action="store_true",
        help="Disable Trackleed visibility gating and mark all points visible.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        query_points_tyx = np.asarray(data["query_points_tyx"], dtype=np.float32)

    result = _run_trackonr_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        settings=TrackonrRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            device=args.device,
            backend_mode="direct",
            inference_resolution=(int(args.height), int(args.width)),
            support_grid_size=int(max(0, args.support_grid_size)),
            use_visibility_gating=not bool(args.disable_visibility_gating),
            visibility_threshold=float(args.visibility_threshold),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        checkpoint_name=np.asarray(result.checkpoint_name),
        support_grid_size=np.asarray(result.support_grid_size, dtype=np.int32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
