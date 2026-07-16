#!/usr/bin/env python3
"""Create non-destructive publication copies enriched with stored FWHM data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.io.publication_exporter import create_publication_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy saved LEED analysis exports and sessions into a separate publication "
            "tree, then add FWHM from stored fit results without loading TIFF files."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data"),
        help="Root searched recursively for exports and saved sessions (default: data).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("publication_exports"),
        help=(
            "Output root outside the source tree (default: publication_exports). "
            "Existing matching output is reused; non-matching output is never overwritten."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = create_publication_export(args.source, args.destination)
    except (FileExistsError, NotADirectoryError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 2

    action = "Reused" if result.reused else "Created"
    print(f"{action} publication export: {result.destination_root}")
    print(f"Manifest: {result.destination_root / 'export_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
