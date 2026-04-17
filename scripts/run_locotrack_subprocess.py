#!/usr/bin/env python3
"""Run LocoTrack offline tracking in a dedicated interpreter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.locotrack_backend import (
    LocotrackRuntimeSettings,
    _run_locotrack_direct_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LocoTrack in a subprocess-friendly mode.")
    parser.add_argument("--input-npz", required=True, help="Input .npz containing frames and queries.")
    parser.add_argument("--output-npz", required=True, help="Output .npz path.")
    parser.add_argument("--checkpoint", required=True, help="Path to the LocoTrack checkpoint.")
    parser.add_argument("--repo-path", default=None, help="Path to the local locotrack repo.")
    parser.add_argument("--model-size", default="auto", help="LocoTrack model size: auto, small, or base.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--inference-resolution",
        type=int,
        default=256,
        help="Square inference resolution used before LocoTrack inference.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=64, help="LocoTrack query chunk size.")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Visibility threshold used by Trackleed post-processing.",
    )
    parser.add_argument(
        "--expected-dist-threshold",
        type=float,
        default=0.5,
        help="Expected-distance threshold used by Trackleed post-processing.",
    )
    parser.add_argument(
        "--disable-visibility-gating",
        action="store_true",
        help="Disable visibility gating and keep all points visible.",
    )
    parser.add_argument(
        "--disable-uncertainty-gating",
        action="store_true",
        help="Disable expected-distance gating and keep all points visible.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_npz).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])
        query_points_tyx = np.asarray(data["query_points_tyx"], dtype=np.float32)

    result = _run_locotrack_direct_offline(
        frames=frames,
        query_points_tyx=query_points_tyx,
        settings=LocotrackRuntimeSettings(
            repo_path=args.repo_path,
            checkpoint_path=args.checkpoint,
            model_size=str(args.model_size),
            device=args.device,
            backend_mode="direct",
            inference_resolution=int(args.inference_resolution),
            query_chunk_size=int(max(1, args.query_chunk_size)),
            use_visibility_gating=not bool(args.disable_visibility_gating),
            visibility_threshold=float(args.visibility_threshold),
            use_uncertainty_gating=not bool(args.disable_uncertainty_gating),
            expected_dist_threshold=float(args.expected_dist_threshold),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tracks_xy=result.tracks_xy.astype(np.float32, copy=False),
        visible_mask=result.visible_mask.astype(bool, copy=False),
        occlusion_logits=result.occlusion_logits.astype(np.float32, copy=False),
        expected_dist_logits=result.expected_dist_logits.astype(np.float32, copy=False),
        inference_resolution=np.asarray(result.inference_resolution, dtype=np.int32),
        original_resolution=np.asarray(result.original_resolution, dtype=np.int32),
        model_size=np.asarray(result.model_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
