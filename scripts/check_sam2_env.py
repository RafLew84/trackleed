#!/usr/bin/env python3
"""Print SAM 2 runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.sam2_backend import (
    DEFAULT_SAM2_MODELS_DIR,
    format_sam2_environment_report,
    probe_sam2_environment,
    run_sam2_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAM 2 runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local SAM 2 repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_SAM2_MODELS_DIR),
        help="Directory containing optional SAM 2 checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit SAM 2 checkpoint path or name.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional SAM 2 config path or config identifier.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cuda:0/cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build image/video predictors and run a tiny two-frame prompt propagation smoke test.",
    )
    args = parser.parse_args()

    status = probe_sam2_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
        config_path=args.config,
    )
    print(format_sam2_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_sam2_smoke_test(
            checkpoint_path=status.selected_checkpoint,
            repo_path=status.repo_path,
            config_path=args.config,
            device=args.device,
        )
        print("Smoke test:")
        print(f"- device: {result.device}")
        print(f"- checkpoint: {result.checkpoint_name}")
        print(f"- config: {result.config_name}")
        print(f"- image masks shape: {result.image_masks_shape}")
        print(f"- image ious shape: {result.image_ious_shape}")
        print(f"- video prompt masks shape: {result.video_prompt_masks_shape}")
        print(f"- video propagated masks shape: {result.video_propagated_masks_shape}")
        print(f"- propagated frame count: {result.propagated_frame_count}")
        print(f"- first image mask area: {result.first_image_mask_area}")
        print(f"- first video mask area: {result.first_video_mask_area}")
        print(f"- CUDA extension: {'ok' if result.cuda_extension_importable else 'missing'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
