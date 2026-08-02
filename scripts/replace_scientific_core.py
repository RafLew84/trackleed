#!/usr/bin/env python3
"""Replace a scientific-core fingerprint through an authorized workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.scientific_core_replacement import (  # noqa: E402
    ScientificCoreReplacementError,
    replace_fingerprint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-predecessor", required=True)
    parser.add_argument("--decision-record", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = replace_fingerprint(
            arguments.workspace_root,
            expected_predecessor=arguments.expected_predecessor,
            decision_record=arguments.decision_record,
            implementation_commit=arguments.implementation_commit,
            authorization=arguments.authorization,
            check_only=arguments.check_only,
        )
    except (OSError, ScientificCoreReplacementError, ValueError) as exc:
        print(f"Scientific-core replacement failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
