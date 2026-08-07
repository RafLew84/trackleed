#!/usr/bin/env python3
"""Preview, freeze, or validate the cross-contract manuscript claim boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.publication.cross_contract_claim_boundary import (
    CrossContractClaimBoundaryError,
    freeze_cross_contract_claim_boundary,
    preview_cross_contract_claim_boundary,
    validate_cross_contract_claim_boundary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--package",
        type=Path,
        default=REPO_ROOT / "manuscript/cross_contract/claim_boundary",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--confirm", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirmed-by", default="")
    parser.add_argument("--confirmed-date", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.preview:
            report = preview_cross_contract_claim_boundary(
                args.workspace_root, args.package
            )
        elif args.confirm:
            report = freeze_cross_contract_claim_boundary(
                args.workspace_root,
                args.package,
                confirmed_by=args.confirmed_by,
                confirmed_date=args.confirmed_date,
            )
        else:
            report = validate_cross_contract_claim_boundary(
                args.workspace_root, args.package
            )
    except CrossContractClaimBoundaryError as exc:
        print(
            f"Cross-contract claim-boundary validation failed: {exc}", file=sys.stderr
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
