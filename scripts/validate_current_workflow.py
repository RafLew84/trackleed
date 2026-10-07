from __future__ import annotations

import argparse
import json
from functools import partial
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.validation.command_log import run_logged_validation  # noqa: E402
from trackleed.validation.current_workflow import validate_current_workflow  # noqa: E402
from trackleed.validation.progress import ValidationProgress, read_validation_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation, printing a short result and the full log path."
    )
    parser.add_argument(
        "--text-base", default=os.environ.get("TRACKLEED_VALIDATION_BASE"),
        help="Compare changed text with this commit, or EMPTY for all checked-out files.",
    )
    parser.add_argument("--status", nargs="?", const="latest", metavar="STATUS_FILE",
                        help="Read an existing run status as JSON without running checks; see docs/validation-progress.md.")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status is not None:
        try:
            print(json.dumps(read_validation_status(REPOSITORY_ROOT, args.status), indent=2))
            return 0
        except (OSError, ValueError, KeyError, TypeError) as cause:
            print(f"Cannot read validation status: {cause}", file=sys.stderr)
            return 1
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return run_logged_validation(REPOSITORY_ROOT, partial(_validate, args))


def _validate(args: argparse.Namespace, progress: ValidationProgress) -> tuple[int, str]:
    if not sys.flags.utf8_mode:
        return 1, "Start validation with: python -X utf8 scripts/validate_current_workflow.py"
    if args.reference_manifest is not None and args.reference_report is None:
        return 1, (
            "Current workflow validation failed: --reference-report is required "
            "with --reference-manifest"
        )
    if args.reference_report is not None and args.reference_manifest is None:
        return 1, (
            "Current workflow validation failed: --reference-manifest is required "
            "with --reference-report"
        )
    if args.reference_manifest is not None and not args.reference_manifest.is_file():
        return 1, (
            "Current workflow validation failed: reference manifest does not exist: "
            f"{args.reference_manifest}"
        )
    report = validate_current_workflow(REPOSITORY_ROOT, text_base=args.text_base, progress=progress)
    if not report.ok:
        print(report.format_errors(), file=sys.stderr)
        return 1, "Current workflow validation failed: " + report.errors[0]
    reference_summary = ""
    if args.reference_manifest is not None:
        from trackleed.validation.reference_workflow import (
            ReferenceValidationConfig,
            validate_reference_workflow,
        )

        progress.start_step("reference_validation")
        reference = validate_reference_workflow(
            ReferenceValidationConfig(
                repository_root=REPOSITORY_ROOT,
                manifest_path=args.reference_manifest,
                report_path=args.reference_report,
            )
        )
        progress.complete_step()
        reference_summary = (
            f", {reference.golden_frame_count} approved golden frames across "
            f"{reference.series_count} series"
        )
    return 0, (
        "Current workflow validation passed: "
        f"{report.checked_python_files} Python files, {report.tests_run} tests"
        f"{reference_summary}; changed text and generator types passed."
    )


if __name__ == "__main__":
    raise SystemExit(main())
