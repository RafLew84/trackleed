#!/usr/bin/env python3
"""Create temperature-axis FWHM and amplitude plots from a stored CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.io.temperature_series_plots import (
    export_temperature_series_plots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot effective/profile FWHM and Gaussian amplitude against "
            "temperature from full_analysis_per_spot.csv."
        )
    )
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory; an existing path is never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_temperature_series_plots(
            args.source_csv,
            args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Temperature plot export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Temperature plots written to: {result.output_dir}")
    for plot_path in result.plot_paths:
        print(f"Plot: {plot_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
