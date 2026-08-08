#!/usr/bin/env python3
"""Freeze once or validate the focused cross-contract main manuscript."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.cross_contract_manuscript import (
    CrossContractManuscriptError,
    freeze_cross_contract_manuscript,
    validate_cross_contract_manuscript,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "manuscript/cross_contract/main_manuscript",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.freeze:
            report = freeze_cross_contract_manuscript(
                arguments.workspace_root,
                arguments.package,
            )
            action = "Froze"
        else:
            report = validate_cross_contract_manuscript(
                arguments.workspace_root,
                arguments.package,
            )
            action = "Validated"
    except (
        CrossContractManuscriptError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cross-contract manuscript package failed: {exc}", file=sys.stderr)
        return 2
    print(f"{action} cross-contract main manuscript:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
