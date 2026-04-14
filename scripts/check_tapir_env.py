#!/usr/bin/env python3
"""Print TAPIR runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.tapir_backend import (
    DEFAULT_TAPIR_MODELS_DIR,
    discover_online_tapir_checkpoints,
    format_tapir_environment_report,
    probe_tapir_environment,
    run_tapir_offline,
    run_tapir_online,
    select_default_online_tapir_checkpoint,
    TapirOnlineRuntimeSettings,
    TapirRuntimeSettings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TAPIR runtime availability for Trackleed.")
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_TAPIR_MODELS_DIR),
        help="Directory containing TAPIR .pt checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit TAPIR checkpoint path or filename relative to --models-dir.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "online"),
        default="offline",
        help="Whether to validate the clip-based offline runtime or the causal online runtime.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic offline inference pass after diagnostics.",
    )
    args = parser.parse_args()

    checkpoint_override = args.checkpoint
    if checkpoint_override is None and args.mode == "online":
        checkpoints = discover_online_tapir_checkpoints(args.models_dir)
        selected_checkpoint = select_default_online_tapir_checkpoint(checkpoints)
        checkpoint_override = str(selected_checkpoint) if selected_checkpoint is not None else None

    status = probe_tapir_environment(
        models_dir=args.models_dir,
        checkpoint_path=checkpoint_override,
    )
    print(format_tapir_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        frames = np.zeros((4, 64, 64), dtype=np.uint8)
        frames[:, 30:34, 30:34] = 255
        query_points = np.array([[0.0, 32.0, 32.0]], dtype=np.float32)
        if args.mode == "online":
            result = run_tapir_online(
                frames=frames,
                query_points_tyx=query_points,
                settings=TapirOnlineRuntimeSettings(
                    checkpoint_path=status.selected_checkpoint,
                    device="auto",
                    inference_resolution=64,
                    query_chunk_size=8,
                ),
            )
            print("Smoke test (online):")
            print(f"- tracks shape: {result.tracks_xy.shape}")
            print(f"- visible shape: {result.visible_mask.shape}")
            print(f"- inference resolution: {result.inference_resolution}")
            print(f"- query frame idx: {result.query_frame_idx}")
            print(f"- first track point: {result.tracks_xy[0, result.query_frame_idx].tolist()}")
        else:
            result = run_tapir_offline(
                frames=frames,
                query_points_tyx=query_points,
                settings=TapirRuntimeSettings(
                    checkpoint_path=status.selected_checkpoint,
                    device="auto",
                    inference_resolution=64,
                    query_chunk_size=8,
                ),
            )
            print("Smoke test (offline):")
            print(f"- tracks shape: {result.tracks_xy.shape}")
            print(f"- visible shape: {result.visible_mask.shape}")
            print(f"- inference resolution: {result.inference_resolution}")
            print(f"- first track point: {result.tracks_xy[0, 0].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
