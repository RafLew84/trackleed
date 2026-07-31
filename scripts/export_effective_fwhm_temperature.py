#!/usr/bin/env python3
"""Export scalar FWHM plots for separate heating and cooling branches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.io.effective_fwhm_temperature_export import (
    export_effective_fwhm_temperature,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export geometric-mean FWHM data and separate heating/cooling "
            "plots without exposing legacy X/Y width labels."
        )
    )
    parser.add_argument(
        "--fit-csv",
        required=True,
        type=Path,
        help="Canonical full_analysis_per_spot.csv containing stored fit widths.",
    )
    parser.add_argument(
        "--series-csv",
        required=True,
        type=Path,
        help="Existing curated plot-data CSV supplying order and quality status.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory; existing target files are never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_effective_fwhm_temperature(
            fit_csv=args.fit_csv,
            series_csv=args.series_csv,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"FWHM temperature export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Data: {result.data_path}")
    for plot_path in result.plot_paths:
        print(f"Plot: {plot_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
