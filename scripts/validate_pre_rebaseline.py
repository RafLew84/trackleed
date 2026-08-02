#!/usr/bin/env python3
"""Validate and record the non-mutating pre-rebaseline integration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.pre_rebaseline import (  # noqa: E402
    PreRebaselineValidationError,
    validate_pre_rebaseline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expectation", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.resolve()
    fingerprint = workspace / "manuscript/scientific_core/fingerprint.json"
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else workspace / "manuscript/scientific_core/pre_rebaseline_report.json"
    )
    if output == fingerprint:
        print("Pre-rebaseline report cannot replace fingerprint.json", file=sys.stderr)
        return 2
    try:
        report = validate_pre_rebaseline(
            workspace,
            expectation_path=arguments.expectation,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, PreRebaselineValidationError, ValueError) as exc:
        print(f"Pre-rebaseline validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
