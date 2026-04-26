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
    DEFAULT_OSTRACK_BBOX_PADDING_CANDIDATES,
    DEFAULT_OSTRACK_CONFIG_NAME,
    DEFAULT_OSTRACK_MODELS_DIR,
    DEFAULT_OSTRACK_PROBE_TIMEOUT_SEC,
    DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
    DEFAULT_OSTRACK_SUBPROCESS_TIMEOUT_SEC,
    format_ostrack_backend_smoke_test_report,
    format_ostrack_environment_report,
    format_ostrack_initial_bbox_probe_report,
    probe_ostrack_environment,
    probe_ostrack_initial_bbox_padding,
    run_ostrack_backend_smoke_test,
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
        "--device",
        default="auto",
        help="Torch device for backend smoke tests, e.g. auto/cuda/cuda:0.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run upstream tracking/profile_model.py after diagnostics.",
    )
    parser.add_argument(
        "--backend-smoke-test",
        action="store_true",
        help="Run Trackleed's OSTrack backend on a tiny synthetic sequence.",
    )
    parser.add_argument(
        "--probe-bbox-init",
        action="store_true",
        help="Compare several initial bbox padding values on synthetic small-spot sequences.",
    )
    parser.add_argument(
        "--bbox-padding-candidates",
        default=",".join(str(v) for v in DEFAULT_OSTRACK_BBOX_PADDING_CANDIDATES),
        help="Comma-separated padding candidates in pixels used by --probe-bbox-init.",
    )
    parser.add_argument(
        "--profile-timeout-sec",
        type=float,
        default=DEFAULT_OSTRACK_PROFILE_TIMEOUT_SEC,
        help="Timeout for --smoke-test profile_model.py execution.",
    )
    parser.add_argument(
        "--probe-timeout-sec",
        type=float,
        default=DEFAULT_OSTRACK_PROBE_TIMEOUT_SEC,
        help="Timeout for --probe-bbox-init execution.",
    )
    parser.add_argument(
        "--backend-timeout-sec",
        type=float,
        default=DEFAULT_OSTRACK_SUBPROCESS_TIMEOUT_SEC,
        help="Timeout for --backend-smoke-test execution.",
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

    if args.backend_smoke_test:
        result = run_ostrack_backend_smoke_test(
            repo_path=status.repo_path,
            checkpoint_path=status.selected_checkpoint,
            python_executable=status.python_executable,
            config_name=args.config,
            device=args.device,
            backend_mode="subprocess",
            representative_point_mode="bbox_center",
            initial_bbox_padding_px=0.0,
            timeout_sec=float(args.backend_timeout_sec),
        )
        print(format_ostrack_backend_smoke_test_report(result))

    if args.probe_bbox_init:
        probe = probe_ostrack_initial_bbox_padding(
            repo_path=status.repo_path,
            checkpoint_path=status.selected_checkpoint,
            python_executable=status.python_executable,
            config_name=args.config,
            candidate_paddings_px=_parse_padding_candidates(args.bbox_padding_candidates),
            timeout_sec=float(args.probe_timeout_sec),
        )
        print(format_ostrack_initial_bbox_probe_report(probe))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
