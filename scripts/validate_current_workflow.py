from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trackleed.validation.current_workflow import validate_current_workflow  # noqa: E402


def main() -> int:
    report = validate_current_workflow(REPOSITORY_ROOT)
    if not report.ok:
        print("Current workflow validation failed:", file=sys.stderr)
        print(report.format_errors(), file=sys.stderr)
        return 1
    print(
        "Current workflow validation passed: "
        f"{report.checked_python_files} Python files, {report.tests_run} tests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
