#!/usr/bin/env python3
"""Print ToMP / TaMOs runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.tomp_backend import (
    DEFAULT_TAMOS_MODELS_DIR,
    DEFAULT_TOMP_BBOX_PADDING_CANDIDATES,
    DEFAULT_TOMP_MODELS_DIR,
    expected_tomp_checkpoint_name_for_variant,
    format_tomp_environment_report,
    format_tomp_initial_bbox_probe_report,
    probe_tomp_environment,
    probe_tomp_initial_bbox_padding,
    run_tomp_smoke_test,
    select_default_tamos_checkpoint_for_variant,
    select_default_tomp_checkpoint_for_variant,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ToMP / TaMOs runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local PyTracking repository root.",
    )
    parser.add_argument(
        "--tomp-models-dir",
        default=str(DEFAULT_TOMP_MODELS_DIR),
        help="Directory containing ToMP checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--tamos-models-dir",
        default=str(DEFAULT_TAMOS_MODELS_DIR),
        help="Directory containing TaMOs checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--tomp-checkpoint",
        default=None,
        help="Explicit ToMP checkpoint path or filename.",
    )
    parser.add_argument(
        "--tamos-checkpoint",
        default=None,
        help="Explicit TaMOs checkpoint path or filename.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the optional smoke test, e.g. auto/cuda/cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic forward pass for the selected ToMP / TaMOs variant after diagnostics.",
    )
    parser.add_argument(
        "--probe-bbox-init",
        action="store_true",
        help="Compare several initial bbox padding values on synthetic small-object sequences.",
    )
    parser.add_argument(
        "--bbox-padding-candidates",
        default=",".join(str(v) for v in DEFAULT_TOMP_BBOX_PADDING_CANDIDATES),
        help="Comma-separated padding candidates in pixels used by --probe-bbox-init.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "all",
            "tomp",
            "tomp50",
            "tomp101",
            "tamos",
            "tamos_resnet50",
            "tamos_swin_base",
        ),
        default="all",
        help="Which variant to smoke-test or probe.",
    )
    args = parser.parse_args()

    status = probe_tomp_environment(
        tomp_models_dir=args.tomp_models_dir,
        tamos_models_dir=args.tamos_models_dir,
        tomp_checkpoint_path=args.tomp_checkpoint,
        tamos_checkpoint_path=args.tamos_checkpoint,
        repo_path=args.repo_path,
    )
    print(format_tomp_environment_report(status))
    if not status.is_ready:
        return 1

    requested_smoke_variants = _expand_smoke_variants(args.variant)
    requested_probe_variants = _expand_probe_variants(args.variant)
    exit_code = 0
    did_anything = False

    if args.smoke_test:
        for variant in requested_smoke_variants:
            checkpoint_path = _select_checkpoint_for_variant(status, variant)
            variant_label = variant
            is_tamos = variant.startswith("tamos")
            family_ready = status.tamos_ready if is_tamos else status.tomp_ready
            if family_ready and checkpoint_path is not None:
                did_anything = True
                result = run_tomp_smoke_test(
                    repo_path=status.repo_path,
                    checkpoint_path=checkpoint_path,
                    variant=variant,
                    device=args.device,
                )
                _print_smoke_result(result)
                continue
            print(
                f"Skipping {variant_label} smoke test: runtime or checkpoint "
                f"`{expected_tomp_checkpoint_name_for_variant(variant)}` not ready."
            )
            exit_code = 1 if args.variant != "all" else exit_code

    if args.probe_bbox_init:
        probe_variants = []
        tomp_checkpoint = None
        tamos_checkpoint = None

        for variant in requested_probe_variants:
            checkpoint_path = _select_checkpoint_for_variant(status, variant)
            if checkpoint_path is None:
                print(
                    f"Skipping {variant} bbox probe: checkpoint "
                    f"`{expected_tomp_checkpoint_name_for_variant(variant)}` not ready."
                )
                exit_code = 1 if args.variant != "all" else exit_code
                continue
            if variant.startswith("tamos"):
                if status.tamos_ready:
                    probe_variants.append(variant)
                    tamos_checkpoint = checkpoint_path
                else:
                    print(f"Skipping {variant} bbox probe: runtime not ready.")
                    exit_code = 1 if args.variant != "all" else exit_code
            else:
                if status.tomp_ready:
                    probe_variants.append(variant)
                    tomp_checkpoint = checkpoint_path
                else:
                    print(f"Skipping {variant} bbox probe: runtime not ready.")
                    exit_code = 1 if args.variant != "all" else exit_code

        if probe_variants:
            did_anything = True
            probe = probe_tomp_initial_bbox_padding(
                repo_path=status.repo_path,
                tomp_checkpoint_path=tomp_checkpoint,
                tamos_checkpoint_path=tamos_checkpoint,
                variants=probe_variants,
                candidate_paddings_px=_parse_padding_candidates(args.bbox_padding_candidates),
                device=args.device,
            )
            print(format_tomp_initial_bbox_probe_report(probe))

    if args.smoke_test or args.probe_bbox_init:
        if not did_anything:
            return 1
    return exit_code


def _print_smoke_result(result) -> None:
    print(f"Smoke test ({result.variant}):")
    print(f"- tracker: {result.tracker_name}")
    print(f"- parameter: {result.parameter_name}")
    print(f"- checkpoint: {result.checkpoint_name}")
    print(f"- device: {result.device}")
    print(f"- frame shape: {result.frame_shape}")
    print(f"- init bbox: {result.init_bbox_xywh}")
    print(f"- tracked bbox: {result.tracked_bbox_xywh}")
    print(f"- object presence score: {result.object_presence_score:.6f}")
    print(f"- init time sec: {result.init_time_sec:.6f}")
    print(f"- track time sec: {result.track_time_sec:.6f}")


def _parse_padding_candidates(value: str) -> list[int]:
    candidates: list[int] = []
    for chunk in str(value or "").split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        candidates.append(int(stripped))
    if not candidates:
        raise ValueError("At least one bbox padding candidate is required.")
    return candidates


def _expand_smoke_variants(variant: str) -> tuple[str, ...]:
    if variant == "all":
        return ("tomp50", "tomp101", "tamos_resnet50", "tamos_swin_base")
    if variant == "tomp":
        return ("tomp50", "tomp101")
    if variant == "tamos":
        return ("tamos_resnet50", "tamos_swin_base")
    return (variant,)


def _expand_probe_variants(variant: str) -> tuple[str, ...]:
    if variant == "all":
        return ("tomp50", "tamos_resnet50")
    if variant == "tomp":
        return ("tomp50",)
    if variant == "tamos":
        return ("tamos_resnet50",)
    return (variant,)


def _select_checkpoint_for_variant(status, variant: str):
    if variant.startswith("tamos"):
        return select_default_tamos_checkpoint_for_variant(
            status.available_tamos_checkpoints,
            variant,
        )
    return select_default_tomp_checkpoint_for_variant(
        status.available_tomp_checkpoints,
        variant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
