#!/usr/bin/env python3
"""Create a non-destructive quality-marked FWHM temperature plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackleed.io.fwhm_temperature_quality_plot import (
    export_quality_marked_fwhm_temperature_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot image-axis FWHM against temperature while retaining "
            "Gaussian fits rejected by explicit quality thresholds as gray "
            "markers. Existing outputs are never overwritten."
        )
    )
    parser.add_argument("--fwhm-csv", required=True, type=Path)
    parser.add_argument("--amplitude-csv", required=True, type=Path)
    parser.add_argument(
        "--output-stem",
        required=True,
        type=Path,
        help="Path without a suffix; PNG, CSV and JSON files are created.",
    )
    parser.add_argument("--min-amplitude", type=float, default=0.10)
    parser.add_argument("--min-r-squared", type=float, default=0.90)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_quality_marked_fwhm_temperature_plot(
            args.fwhm_csv,
            args.amplitude_csv,
            args.output_stem,
            min_amplitude=args.min_amplitude,
            min_r_squared=args.min_r_squared,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Quality-marked FWHM export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Plot: {result.plot_path}")
    print(f"Data: {result.data_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Accepted rows: {result.accepted_count}")
    print(f"Rejected rows: {result.rejected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
