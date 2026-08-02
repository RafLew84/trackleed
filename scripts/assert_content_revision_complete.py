#!/usr/bin/env python3
"""Validate all seven gates and write the content-revision-complete report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.content_revision import (  # noqa: E402
    ContentRevisionValidationError,
    validate_content_revision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else workspace / "manuscript/content_revision/validation_report.json"
    )
    try:
        report = validate_content_revision(
            workspace,
            contract_path=arguments.contract,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (ContentRevisionValidationError, OSError, ValueError) as exc:
        print(f"Content-revision assertion failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "gate_count": len(report["gates"]),
                "output": str(output),
                "placeholder_count": report["placeholder_summary"]["total"],
                "status": report["status"],
                "submission_ready": report["submission_ready"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
