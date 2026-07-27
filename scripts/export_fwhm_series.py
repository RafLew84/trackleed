#!/usr/bin/env python3
"""Create canonical FWHM tables and plots from an existing per-spot CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.io.fwhm_series_exporter import export_fwhm_series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate axis-order-invariant FWHM metrics from an existing "
            "full_analysis_per_spot.csv and create a non-destructive export."
        )
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        type=Path,
        help="Existing per-spot CSV containing FWHM X, FWHM Y and R-squared.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New output directory; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--min-r-squared",
        type=float,
        default=0.85,
        help="Minimum accepted Gaussian-fit R-squared (default: 0.85).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_fwhm_series(
            args.source_csv,
            args.output_dir,
            min_r_squared=args.min_r_squared,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"FWHM export failed: {exc}", file=sys.stderr)
        return 2

    print(f"Created FWHM export: {result.output_dir}")
    print(f"Per-spot data: {result.enriched_csv}")
    print(f"Per-frame data: {result.per_frame_csv}")
    print(f"Quality report: {result.quality_report_csv}")
    for plot_path in result.plot_paths:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
