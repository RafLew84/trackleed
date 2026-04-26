#!/usr/bin/env python3
"""Print OSTrack runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKLEED_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKLEED_ROOT))

from trackleed.analysis.ostrack_backend import (  # noqa: E402
    DEFAULT_OSTRACK_CONFIG_NAME,
    DEFAULT_OSTRACK_MODELS_DIR,
    DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
    format_ostrack_environment_report,
    probe_ostrack_environment,
    run_ostrack_profile_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OSTrack runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local OSTrack repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_OSTRACK_MODELS_DIR),
        help="Directory containing OSTrack checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit OSTrack checkpoint path or filename relative to --models-dir.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable from the dedicated OSTrack Conda env.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_OSTRACK_CONFIG_NAME,
        help="OSTrack experiment config name without .yaml.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run upstream tracking/profile_model.py after diagnostics.",
    )
    parser.add_argument(
        "--profile-timeout-sec",
        type=float,
        default=DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
        help="Timeout for --smoke-test profile_model.py execution.",
    )
    args = parser.parse_args()

    status = probe_ostrack_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
        python_executable=args.python_executable,
        config_name=args.config,
    )
    print(format_ostrack_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_ostrack_profile_smoke_test(
            repo_path=status.repo_path,
            python_executable=status.python_executable,
            config_name=args.config,
            timeout_sec=float(args.profile_timeout_sec),
        )
        print("Smoke test:")
        print(f"- config: {result.config_name}")
        print(f"- returncode: {result.returncode}")
        print(f"- macs: {result.macs or 'unknown'}")
        print(f"- params: {result.params or 'unknown'}")
        print(
            "- latency ms: "
            + (f"{result.latency_ms:.3f}" if result.latency_ms is not None else "unknown")
        )
        print(f"- FPS: {result.fps:.3f}" if result.fps is not None else "- FPS: unknown")
        if result.returncode != 0:
            if result.stdout.strip():
                print("- stdout:")
                print(result.stdout.strip())
            if result.stderr.strip():
                print("- stderr:")
                print(result.stderr.strip())
            return result.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
