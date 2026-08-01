#!/usr/bin/env python3
"""Freeze once or validate the publication scientific-core fingerprint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.scientific_core import (  # noqa: E402
    ScientificCoreValidationError,
    freeze_fingerprint,
    validate_fingerprint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--fingerprint", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.resolve()
    package = workspace / "manuscript" / "scientific_core"
    contract = (arguments.contract or package / "fingerprint_contract.json").resolve()
    fingerprint = (arguments.fingerprint or package / "fingerprint.json").resolve()
    try:
        if arguments.build:
            report = freeze_fingerprint(workspace, contract, fingerprint)
        else:
            report = validate_fingerprint(workspace, contract, fingerprint)
    except (
        FileExistsError,
        FileNotFoundError,
        ScientificCoreValidationError,
        ValueError,
    ) as exc:
        print(f"Scientific-core fingerprint failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
