"""Non-destructive FWHM series export from existing per-spot CSV data."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from trackleed.analysis.fwhm_series import (
    DERIVED_COLUMNS,
    enrich_fwhm_rows,
    numeric_sort_key,
    summarize_fwhm_per_frame,
)
from trackleed.io.fwhm_series_plots import write_fwhm_plots


@dataclass(frozen=True)
class FwhmSeriesExportResult:
    """Artifacts produced by one non-destructive FWHM series export."""

    output_dir: Path
    enriched_csv: Path
    per_frame_csv: Path
    quality_report_csv: Path
    plot_paths: tuple[Path, ...]


def export_fwhm_series(
    source_csv: Path | str,
    output_dir: Path | str,
    *,
    min_r_squared: float = 0.85,
) -> FwhmSeriesExportResult:
    """Create a canonical FWHM export without modifying the source CSV."""

    source_path = Path(source_csv)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"fwhm_x", "fwhm_y", "r_squared"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(
                "Source CSV is missing required columns: " + ", ".join(missing)
            )
        rows = enrich_fwhm_rows(
            reader,
            min_r_squared=min_r_squared,
        )

    destination.mkdir(parents=True)
    enriched_csv = destination / "full_analysis_per_spot_fwhm.csv"
    _write_csv(
        enriched_csv,
        fieldnames
        + [column for column in DERIVED_COLUMNS if column not in fieldnames],
        rows,
    )

    per_frame_csv = destination / "fwhm_per_frame.csv"
    _write_csv(
        per_frame_csv,
        [
            "frame_id",
            "temperature",
            "energy",
            "coverage",
            "fwhm_effective_mean",
            "fwhm_valid_count",
            "fwhm_invalid_count",
        ],
        summarize_fwhm_per_frame(rows),
    )

    quality_report_csv = destination / "fwhm_quality_report.csv"
    _write_csv(
        quality_report_csv,
        ["track_id", "quality_reason", "count"],
        _quality_report_rows(rows),
    )
    plot_paths = write_fwhm_plots(destination, rows)

    return FwhmSeriesExportResult(
        output_dir=destination,
        enriched_csv=enriched_csv,
        per_frame_csv=per_frame_csv,
        quality_report_csv=quality_report_csv,
        plot_paths=plot_paths,
    )


def _quality_report_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str | int]]:
    counts = Counter(
        (
            str(row.get("track_id", "")),
            str(row.get("fwhm_quality_reason", "")),
        )
        for row in rows
    )
    return [
        {
            "track_id": track_id,
            "quality_reason": reason,
            "count": count,
        }
        for (track_id, reason), count in sorted(
            counts.items(),
            key=lambda item: (
                numeric_sort_key(item[0][0]),
                item[0][1] != "ok",
                item[0][1],
            ),
        )
    ]


def _write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
