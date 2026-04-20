#!/usr/bin/env python3
"""Print Cutie runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.cutie_backend import (
    DEFAULT_CUTIE_MODELS_DIR,
    format_cutie_environment_report,
    format_cutie_initial_mask_probe_report,
    probe_cutie_environment,
    probe_cutie_initial_mask_builder,
    run_cutie_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Cutie runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local Cutie repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_CUTIE_MODELS_DIR),
        help="Directory containing optional Cutie checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit Cutie checkpoint path or name.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cuda:0/cpu.",
    )
    parser.add_argument(
        "--max-internal-size",
        type=int,
        default=256,
        help="Maximum shortest side used by the smoke-test inference core.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the default example sequence and run a tiny two-frame Cutie propagation smoke test.",
    )
    parser.add_argument(
        "--probe-mask-builder",
        action="store_true",
        help="Compare threshold/full_bbox/disk initial-mask builders on the bundled Cutie examples.",
    )
    parser.add_argument(
        "--mask-builder-threshold",
        type=float,
        default=0.7,
        help="Relative threshold used by the `threshold` mask builder.",
    )
    parser.add_argument(
        "--mask-builder-disk-radius-ratio",
        type=float,
        default=0.35,
        help="Disk radius as a fraction of the shorter ROI side.",
    )
    parser.add_argument(
        "--mask-builder-padding-ratio",
        type=float,
        default=0.15,
        help="Padding ratio applied when deriving the probe ROI from the reference mask bbox.",
    )
    parser.add_argument(
        "--mask-builder-jitter-px",
        type=int,
        default=4,
        help="Minimum absolute jitter used for bbox stability checks.",
    )
    parser.add_argument(
        "--mask-builder-jitter-scale",
        type=float,
        default=0.12,
        help="Relative bbox jitter used for stability checks.",
    )
    args = parser.parse_args()

    status = probe_cutie_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
    )
    print(format_cutie_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_cutie_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            device=args.device,
            max_internal_size=args.max_internal_size,
        )
        print("Smoke test:")
        print(f"- device: {result.device}")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- example sequence: {result.example_sequence_name}")
        print(f"- frame count: {result.frame_count}")
        print(f"- object ids: {result.object_ids}")
        print(f"- first mask shape: {result.first_mask_shape}")
        print(f"- second mask shape: {result.second_mask_shape}")
        print(f"- first mask area: {result.first_mask_area}")
        print(f"- second mask area: {result.second_mask_area}")
        print(f"- max_internal_size: {result.max_internal_size}")

    if args.probe_mask_builder:
        probe = probe_cutie_initial_mask_builder(
            repo_path=status.repo_path,
            threshold_rel=args.mask_builder_threshold,
            disk_radius_ratio=args.mask_builder_disk_radius_ratio,
            bbox_padding_ratio=args.mask_builder_padding_ratio,
            jitter_px=args.mask_builder_jitter_px,
            jitter_scale=args.mask_builder_jitter_scale,
        )
        print(format_cutie_initial_mask_probe_report(probe))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
