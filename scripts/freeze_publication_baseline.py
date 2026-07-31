#!/usr/bin/env python3
"""Build once or validate the frozen publication scientific-baseline contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.scientific_baseline import (  # noqa: E402
    BaselineValidationError,
    build_package,
    validate_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve the relative paths in source_inventory.json.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "manuscript" / "scientific_baseline",
        help="Directory containing source_inventory.json and the frozen package.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build",
        action="store_true",
        help="Create a new package; refuses to replace any frozen output.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate package files and linked sources without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.build:
            report = build_package(args.workspace_root, args.package)
            action = "Built"
        else:
            report = validate_package(args.workspace_root, args.package)
            action = "Validated"
    except (
        BaselineValidationError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Scientific-baseline contract failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"{action} scientific-baseline contract: "
        f"{report.case_count} cases, {report.method_count} methods, "
        f"{report.raw_image_count} raw images, "
        f"{report.visible_opportunity_count} opportunities, "
        f"{report.evaluable_gap_count} gaps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
