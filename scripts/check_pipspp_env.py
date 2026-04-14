#!/usr/bin/env python3
"""Print PIPs++ runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.pipspp_backend import (
    DEFAULT_PIPSPP_MODELS_DIR,
    format_pipspp_environment_report,
    probe_pipspp_environment,
    run_pipspp_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PIPs++ runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local PIPs++ repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_PIPSPP_MODELS_DIR),
        help="Directory containing optional PIPs++ checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit PIPs++ checkpoint reference. This can be a directory such as "
            "`reference_model`, a .pth file, or a path relative to --models-dir / --repo-path."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic forward pass after diagnostics.",
    )
    args = parser.parse_args()

    status = probe_pipspp_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
    )
    print(format_pipspp_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_pipspp_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            device=args.device,
        )
        print("Smoke test:")
        print(f"- device: {result.device}")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- prediction stages: {result.num_prediction_stages}")
        print(f"- tracks shape: {result.tracks_shape}")
        print(f"- first track point: {result.first_track_point}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
