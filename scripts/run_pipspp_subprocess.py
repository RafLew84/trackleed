#!/usr/bin/env python3
"""Run PIPs++ offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.pipspp_backend import (
    PipsppRuntimeSettings,
    _run_pipspp_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PIPs++ in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the PIPs++ checkpoint ref/file.")
    parser.add_argument("--repo-path", default=None, help="Path to the local pipspp repo.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--height", type=int, required=True, help="Inference height.")
    parser.add_argument("--width", type=int, required=True, help="Inference width.")
    parser.add_argument("--iters", type=int, default=4, help="Number of PIPs++ refinement iterations.")
    parser.add_argument("--beautify", action="store_true", help="Enable the upstream beautify step.")
    parser.add_argument(
        "--disable-inbounds-visibility",
        action="store_true",
        help="Disable in-bounds visibility heuristic and mark all predicted points visible.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        query_points_tyx = np.asarray(data["query_points_tyx"], dtype=np.float32)

    result = _run_pipspp_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        settings=PipsppRuntimeSettings(
            checkpoint_path=args.checkpoint,
            repo_path=args.repo_path,
            device=args.device,
            backend_mode="direct",
            inference_resolution=(int(args.height), int(args.width)),
            iters=int(max(1, args.iters)),
            beautify=bool(args.beautify),
            use_inbounds_visibility=not bool(args.disable_inbounds_visibility),
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
        num_prediction_stages=np.asarray(result.num_prediction_stages, dtype=np.int32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
