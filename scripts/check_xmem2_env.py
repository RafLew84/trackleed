#!/usr/bin/env python3
"""Print XMem2 / XMem++ runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.xmem2_backend import (
    DEFAULT_XMEM2_EXAMPLE_FRAME_LIMIT,
    DEFAULT_XMEM2_MODELS_DIR,
    format_xmem2_environment_report,
    format_xmem2_initial_mask_probe_report,
    probe_xmem2_environment,
    probe_xmem2_initial_mask_builder,
    run_xmem2_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check XMem2 / XMem++ runtime availability for Trackleed."
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local XMem2 repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_XMEM2_MODELS_DIR),
        help="Directory containing optional XMem checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit XMem checkpoint path or name.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny XMem2 propagation smoke test on the bundled `caps` example.",
    )
    parser.add_argument(
        "--smoke-test-frame-limit",
        type=int,
        default=DEFAULT_XMEM2_EXAMPLE_FRAME_LIMIT,
        help="Use only the first N frames from the bundled `caps` example for the smoke test.",
    )
    parser.add_argument(
        "--probe-mask-builder",
        action="store_true",
        help="Compare threshold/full_bbox/disk initial-mask builders on the bundled `caps` example annotations.",
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

    status = probe_xmem2_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
    )
    print(format_xmem2_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_xmem2_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            frame_limit=args.smoke_test_frame_limit,
        )
        print("Smoke test:")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- example sequence: {result.example_sequence_name}")
        print(f"- frame limit: {result.frame_limit}")
        print(f"- frame count: {result.frame_count}")
        print(f"- annotation frame count: {result.annotation_frame_count}")
        print(f"- output mask count: {result.output_mask_count}")
        print(f"- output overlay count: {result.output_overlay_count}")
        print(f"- first output mask shape: {result.first_output_mask_shape}")
        print(f"- first output mask area: {result.first_output_mask_area}")
        print(f"- stats shape: {result.stats_shape}")

    if args.probe_mask_builder:
        probe = probe_xmem2_initial_mask_builder(
            repo_path=status.repo_path,
            threshold_rel=args.mask_builder_threshold,
            disk_radius_ratio=args.mask_builder_disk_radius_ratio,
            bbox_padding_ratio=args.mask_builder_padding_ratio,
            jitter_px=args.mask_builder_jitter_px,
            jitter_scale=args.mask_builder_jitter_scale,
        )
        print(format_xmem2_initial_mask_probe_report(probe))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
