#!/usr/bin/env python3
"""Print MFTIQ runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.mftiq_backend import (
    DEFAULT_MFTIQ_MODELS_DIR,
    format_mftiq_environment_report,
    format_mftiq_inference_probe_report,
    probe_mftiq_environment,
    run_mftiq_inference_probe,
    run_mftiq_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MFTIQ runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local MFTIQ repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MFTIQ_MODELS_DIR),
        help="Directory containing optional MFTIQ checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit MFTIQ checkpoint path or name. This can be a .pth file or a path relative "
            "to --models-dir / --repo-path."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional MFTIQ config path. Defaults to configs/MFTIQ4_ROMA_200k_cfg.py.",
    )
    parser.add_argument(
        "--demo-input",
        default=None,
        help="Optional video or image-directory input used for the smoke test.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cuda:0/cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny two-frame point-tracking smoke test after diagnostics.",
    )
    parser.add_argument(
        "--probe-inference",
        action="store_true",
        help="Run a short multi-frame point-tracking probe and print the adapter decision.",
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=4,
        help="Number of frames used by --probe-inference.",
    )
    args = parser.parse_args()

    status = probe_mftiq_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
        config_path=args.config,
        demo_input_path=args.demo_input,
    )
    print(format_mftiq_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_mftiq_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            device=args.device,
            config_path=args.config,
            demo_input_path=args.demo_input,
        )
        print("Smoke test:")
        print(f"- device: {result.device}")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- config: {result.config_name}")
        print(f"- demo input: {result.demo_input_name}")
        print(f"- coords shape: {result.coords_shape}")
        print(f"- occlusions shape: {result.occlusions_shape}")
        print(f"- first coord: {result.first_coord}")
        print(f"- first occlusion: {result.first_occlusion}")

    if args.probe_inference:
        result = run_mftiq_inference_probe(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            device=args.device,
            config_path=args.config,
            demo_input_path=args.demo_input,
            num_frames=args.probe_frames,
        )
        print(format_mftiq_inference_probe_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
