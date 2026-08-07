#!/usr/bin/env python3
"""Freeze once or validate the native ViPErLEED pilot reference contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.viperleed_pilot import (  # noqa: E402
    ViperleedPilotValidationError,
    freeze_viperleed_pilot,
    validate_viperleed_pilot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "manuscript/cross_contract/viperleed_pilot",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.freeze:
            report = freeze_viperleed_pilot(
                arguments.workspace_root,
                arguments.package,
            )
            action = "Frozen"
        else:
            report = validate_viperleed_pilot(
                arguments.workspace_root,
                arguments.package,
            )
            action = "Validated"
    except (
        FileExistsError,
        OSError,
        TypeError,
        ViperleedPilotValidationError,
        ValueError,
    ) as exc:
        print(f"ViPErLEED pilot contract failed: {exc}", file=sys.stderr)
        return 2
    print(f"{action} native ViPErLEED pilot contract:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
