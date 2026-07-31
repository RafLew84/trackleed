"""Validate the source-linked experimental-metadata package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.analysis.experimental_metadata import (  # noqa: E402
    MetadataValidationError,
    validate_metadata_package,
)


DEFAULT_PACKAGE = Path("manuscript/experimental_metadata")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve provenance sources.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=DEFAULT_PACKAGE,
        help="Experimental-metadata package directory.",
    )
    parser.add_argument(
        "--require-author-confirmation",
        action="store_true",
        help="Fail unless the author checkpoint is complete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    package = args.package
    if not package.is_absolute():
        package = workspace_root / package
    try:
        report = validate_metadata_package(
            workspace_root,
            package,
            require_author_confirmation=args.require_author_confirmation,
        )
    except (MetadataValidationError, OSError, ValueError) as exc:
        print(f"Experimental-metadata validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "Validated experimental metadata: "
        f"{report.field_count} fields "
        f"({report.recorded_count} recorded, "
        f"{report.not_recorded_count} not recorded, "
        f"{report.not_available_count} not available); "
        f"author checkpoint={report.author_checkpoint_status}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
