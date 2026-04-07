#!/usr/bin/env python3
"""Print TAPNext runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.tapnext_backend import (
    DEFAULT_TAPNEXT_MODELS_DIR,
    format_tapnext_environment_report,
    probe_tapnext_environment,
    run_tapnext_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TAPNext runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local tapnet repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_TAPNEXT_MODELS_DIR),
        help="Directory containing TAPNext checkpoints (.pt or .npz).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit TAPNext checkpoint path or filename relative to --models-dir.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic sequential inference pass after diagnostics.",
    )
    args = parser.parse_args()

    status = probe_tapnext_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
    )
    print(format_tapnext_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_tapnext_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            device=args.device,
        )
        print("Smoke test:")
        print(f"- device: {result.device}")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- variant: {result.checkpoint_variant}")
        print(f"- tracks shape: {result.tracks_shape}")
        print(f"- track logits shape: {result.track_logits_shape}")
        print(f"- visible logits shape: {result.visible_logits_shape}")
        print(f"- certainty shape: {result.certainty_shape}")
        print(f"- first track point: {result.first_track_point}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
