from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.validation.current_workflow import validate_current_workflow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed current workflow validation."
    )
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference_manifest is not None and args.reference_report is None:
        print(
            "Current workflow validation failed: --reference-report is required "
            "with --reference-manifest",
            file=sys.stderr,
        )
        return 1
    if args.reference_report is not None and args.reference_manifest is None:
        print(
            "Current workflow validation failed: --reference-manifest is required "
            "with --reference-report",
            file=sys.stderr,
        )
        return 1
    if args.reference_manifest is not None and not args.reference_manifest.is_file():
        print(
            f"Current workflow validation failed: reference manifest does not exist: "
            f"{args.reference_manifest}",
            file=sys.stderr,
        )
        return 1
    report = validate_current_workflow(REPOSITORY_ROOT)
    if not report.ok:
        print("Current workflow validation failed:", file=sys.stderr)
        print(report.format_errors(), file=sys.stderr)
        return 1
    reference_summary = ""
    if args.reference_manifest is not None:
        try:
            from trackleed.validation.reference_workflow import (
                ReferenceValidationConfig,
                validate_reference_workflow,
            )

            reference = validate_reference_workflow(
                ReferenceValidationConfig(
                    repository_root=REPOSITORY_ROOT,
                    manifest_path=args.reference_manifest,
                    report_path=args.reference_report,
                )
            )
        except Exception as cause:
            print(
                f"Current workflow reference validation failed: {cause}",
                file=sys.stderr,
            )
            return 1
        reference_summary = (
            f", {reference.golden_frame_count} approved golden frames across "
            f"{reference.series_count} series"
        )
    print(
        "Current workflow validation passed: "
        f"{report.checked_python_files} Python files, {report.tests_run} tests"
        f"{reference_summary}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
