#!/usr/bin/env python3
"""Print SAMURAI runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKLEED_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKLEED_ROOT))

from trackleed.analysis.samurai_backend import (  # noqa: E402
    DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC,
    DEFAULT_SAMURAI_MODEL_VARIANT,
    DEFAULT_SAMURAI_MODELS_DIR,
    DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC,
    format_samurai_environment_report,
    format_samurai_smoke_test_report,
    probe_samurai_environment,
    run_samurai_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAMURAI runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local SAMURAI repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_SAMURAI_MODELS_DIR),
        help="Directory containing SAMURAI/SAM2.1 checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit SAMURAI/SAM2.1 checkpoint path or filename relative to --models-dir.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable from the dedicated SAMURAI Conda env.",
    )
    parser.add_argument(
        "--model-variant",
        default=DEFAULT_SAMURAI_MODEL_VARIANT,
        help=(
            "SAMURAI/SAM2.1 variant, e.g. sam2.1_hiera_base_plus, "
            "sam2.1_hiera_large, small, or tiny."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for --smoke-test. Use auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the SAMURAI video predictor and run a tiny two-frame RGB/JPG bbox smoke test.",
    )
    parser.add_argument(
        "--import-probe-timeout-sec",
        type=float,
        default=DEFAULT_SAMURAI_IMPORT_PROBE_TIMEOUT_SEC,
        help="Timeout for the subprocess import probe.",
    )
    parser.add_argument(
        "--smoke-timeout-sec",
        type=float,
        default=DEFAULT_SAMURAI_SMOKE_TIMEOUT_SEC,
        help="Timeout for --smoke-test execution.",
    )
    args = parser.parse_args()

    status = probe_samurai_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
        python_executable=args.python_executable,
        model_variant=args.model_variant,
        import_probe_timeout_sec=float(args.import_probe_timeout_sec),
    )
    print(format_samurai_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_samurai_smoke_test(
            repo_path=status.repo_path,
            checkpoint_path=status.selected_checkpoint,
            python_executable=status.python_executable,
            model_variant=status.model_variant,
            device=args.device,
            timeout_sec=float(args.smoke_timeout_sec),
        )
        print(format_samurai_smoke_test_report(result))
        if not result.succeeded:
            return result.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
