#!/usr/bin/env python3
"""Prepare, publish after review, or validate the cross-contract draft package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.cross_contract_draft_package import (
    CrossContractDraftPackageError,
    build_cross_contract_draft_package,
    prepare_cross_contract_draft_package,
    validate_cross_contract_draft_package,
)


def _discover_pandoc(explicit: Path | None) -> Path:
    discovered = shutil.which("pandoc")
    candidates = (
        explicit,
        Path(discovered) if discovered else None,
        Path.home() / "AppData/Local/Pandoc/pandoc.exe",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise CrossContractDraftPackageError(
        "Pandoc is unavailable; pass --pandoc explicitly"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "manuscript/cross_contract/draft_package",
    )
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--reproduction-staging", type=Path)
    parser.add_argument("--pandoc", type=Path)
    parser.add_argument("--reviewer-identity")
    parser.add_argument("--review-date")
    parser.add_argument("--confirm-page-acceptance", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--publish-reviewed", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.prepare:
            if arguments.staging is None:
                raise CrossContractDraftPackageError("--prepare requires --staging")
            report = prepare_cross_contract_draft_package(
                arguments.workspace_root,
                arguments.staging,
                pandoc_path=_discover_pandoc(arguments.pandoc),
            )
            action = "Prepared"
        elif arguments.publish_reviewed:
            if arguments.staging is None or arguments.reproduction_staging is None:
                raise CrossContractDraftPackageError(
                    "--publish-reviewed requires both staging paths"
                )
            report = build_cross_contract_draft_package(
                arguments.workspace_root,
                arguments.staging,
                arguments.reproduction_staging,
                reviewer_identity=arguments.reviewer_identity or "",
                review_date=arguments.review_date or "",
                confirm_page_acceptance=arguments.confirm_page_acceptance,
                package_path=arguments.package,
            )
            action = "Published"
        else:
            report = validate_cross_contract_draft_package(
                arguments.workspace_root,
                arguments.package,
            )
            action = "Validated"
    except (
        CrossContractDraftPackageError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Cross-contract draft package failed: {exc}", file=sys.stderr)
        return 2
    print(f"{action} cross-contract draft package:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
