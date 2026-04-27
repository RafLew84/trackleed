#!/usr/bin/env python3
"""Print DAM4SAM runtime diagnostics for the current Trackleed environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


TRACKLEED_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKLEED_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKLEED_ROOT))

from trackleed.analysis.dam4sam_backend import (  # noqa: E402
    DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC,
    DEFAULT_DAM4SAM_MODELS_DIR,
    DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC,
    DEFAULT_DAM4SAM_TRACKER_NAME,
    format_dam4sam_environment_report,
    format_dam4sam_smoke_test_report,
    probe_dam4sam_environment,
    run_dam4sam_smoke_test,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DAM4SAM runtime availability for Trackleed.")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to the local DAM4SAM repository root.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_DAM4SAM_MODELS_DIR),
        help="Directory containing DAM4SAM/SAM2.1 checkpoints for Trackleed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit DAM4SAM/SAM2 checkpoint path or filename relative to --models-dir.",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable from the dedicated DAM4SAM Conda env.",
    )
    parser.add_argument(
        "--tracker-name",
        default=DEFAULT_DAM4SAM_TRACKER_NAME,
        help="DAM4SAM tracker variant, e.g. sam21pp-L or sam21pp-B.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build DAM4SAMTracker and run a tiny non-interactive two-frame RGB smoke test.",
    )
    parser.add_argument(
        "--import-probe-timeout-sec",
        type=float,
        default=DEFAULT_DAM4SAM_IMPORT_PROBE_TIMEOUT_SEC,
        help="Timeout for the subprocess import probe.",
    )
    parser.add_argument(
        "--smoke-timeout-sec",
        type=float,
        default=DEFAULT_DAM4SAM_SMOKE_TIMEOUT_SEC,
        help="Timeout for --smoke-test execution.",
    )
    args = parser.parse_args()

    status = probe_dam4sam_environment(
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint,
        repo_path=args.repo_path,
        python_executable=args.python_executable,
        tracker_name=args.tracker_name,
        import_probe_timeout_sec=float(args.import_probe_timeout_sec),
    )
    print(format_dam4sam_environment_report(status))
    if not status.is_ready:
        return 1

    if args.smoke_test:
        result = run_dam4sam_smoke_test(
            repo_path=status.repo_path,
            python_executable=status.python_executable,
            tracker_name=args.tracker_name,
            timeout_sec=float(args.smoke_timeout_sec),
        )
        print(format_dam4sam_smoke_test_report(result))
        if not result.succeeded:
            return result.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
