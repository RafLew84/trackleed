"""Non-destructive FWHM series export from existing per-spot CSV data."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import statistics

from trackleed.io.fwhm_series_plots import write_fwhm_plots


DERIVED_COLUMNS = (
    "fwhm_minor",
    "fwhm_major",
    "fwhm_effective",
    "fwhm_anisotropy",
    "fwhm_valid",
    "fwhm_quality_reason",
)
MIN_TEMPORAL_QUALITY_POINTS = 5
MAX_EFFECTIVE_MEDIAN_FACTOR = 3.0
MAX_EFFECTIVE_ROBUST_Z = 8.0


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
        rows = [_enrich_row(row, min_r_squared=min_r_squared) for row in reader]
    _apply_temporal_quality(rows)

    destination.mkdir(parents=True)
    enriched_csv = destination / "full_analysis_per_spot_fwhm.csv"
    output_fields = fieldnames + [
        column for column in DERIVED_COLUMNS if column not in fieldnames
    ]
    with enriched_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    per_frame_csv = destination / "fwhm_per_frame.csv"
    _write_per_frame_csv(per_frame_csv, rows)
    quality_report_csv = destination / "fwhm_quality_report.csv"
    _write_quality_report(quality_report_csv, rows)
    plot_paths = write_fwhm_plots(destination, rows)

    return FwhmSeriesExportResult(
        output_dir=destination,
        enriched_csv=enriched_csv,
        per_frame_csv=per_frame_csv,
        quality_report_csv=quality_report_csv,
        plot_paths=plot_paths,
    )


def _enrich_row(
    row: dict[str, str],
    *,
    min_r_squared: float,
) -> dict[str, str]:
    enriched = dict(row)
    fwhm_x = _finite_float(row.get("fwhm_x"))
    fwhm_y = _finite_float(row.get("fwhm_y"))
    r_squared = _finite_float(row.get("r_squared"))

    if fwhm_x is None or fwhm_y is None:
        return _with_quality(enriched, valid=False, reason="missing_fwhm")
    if fwhm_x <= 0 or fwhm_y <= 0:
        return _with_quality(enriched, valid=False, reason="nonpositive_fwhm")

    minor = min(fwhm_x, fwhm_y)
    major = max(fwhm_x, fwhm_y)
    enriched.update(
        {
            "fwhm_minor": _format_number(minor),
            "fwhm_major": _format_number(major),
            "fwhm_effective": _format_number(math.sqrt(fwhm_x * fwhm_y)),
            "fwhm_anisotropy": _format_number(major / minor),
        }
    )
    if r_squared is None:
        return _with_quality(enriched, valid=False, reason="missing_r_squared")
    if r_squared < min_r_squared:
        return _with_quality(
            enriched,
            valid=False,
            reason=f"r_squared_below_{_format_number(min_r_squared)}",
        )
    return _with_quality(enriched, valid=True, reason="ok")


def _with_quality(
    row: dict[str, str],
    *,
    valid: bool,
    reason: str,
) -> dict[str, str]:
    for column in DERIVED_COLUMNS[:4]:
        row.setdefault(column, "")
    row["fwhm_valid"] = "true" if valid else "false"
    row["fwhm_quality_reason"] = reason
    return row


def _apply_temporal_quality(rows: list[dict[str, str]]) -> None:
    rows_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["fwhm_valid"] == "true":
            rows_by_track[str(row.get("track_id", ""))].append(row)

    for track_rows in rows_by_track.values():
        if len(track_rows) < MIN_TEMPORAL_QUALITY_POINTS:
            continue
        effective_values = [
            float(row["fwhm_effective"]) for row in track_rows
        ]
        effective_median = statistics.median(effective_values)
        log_values = [math.log(value) for value in effective_values]
        log_median = statistics.median(log_values)
        log_mad = statistics.median(
            abs(value - log_median) for value in log_values
        )
        robust_scale = 1.4826 * log_mad

        for row, effective, log_effective in zip(
            track_rows,
            effective_values,
            log_values,
        ):
            factor = max(
                effective / effective_median,
                effective_median / effective,
            )
            robust_z = (
                abs(log_effective - log_median) / robust_scale
                if robust_scale > 1e-12
                else 0.0
            )
            if (
                factor > MAX_EFFECTIVE_MEDIAN_FACTOR
                or robust_z > MAX_EFFECTIVE_ROBUST_Z
            ):
                _with_quality(
                    row,
                    valid=False,
                    reason="temporal_fwhm_outlier",
                )


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    return format(value, ".10g")


def _write_per_frame_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("frame_id", ""))].append(row)

    fieldnames = [
        "frame_id",
        "temperature",
        "energy",
        "coverage",
        "fwhm_effective_mean",
        "fwhm_valid_count",
        "fwhm_invalid_count",
    ]
    summary_rows = []
    for frame_id in sorted(grouped, key=_frame_sort_key):
        frame_rows = grouped[frame_id]
        valid_rows = [row for row in frame_rows if row["fwhm_valid"] == "true"]
        effective_values = [
            float(row["fwhm_effective"]) for row in valid_rows
        ]
        summary_rows.append(
            {
                "frame_id": frame_id,
                "temperature": _first_present(frame_rows, "temperature"),
                "energy": _first_present(frame_rows, "energy"),
                "coverage": _first_present(frame_rows, "coverage"),
                "fwhm_effective_mean": (
                    _format_number(statistics.fmean(effective_values))
                    if effective_values
                    else ""
                ),
                "fwhm_valid_count": len(valid_rows),
                "fwhm_invalid_count": len(frame_rows) - len(valid_rows),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _first_present(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def _frame_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def _write_quality_report(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    counts = Counter(
        (
            str(row.get("track_id", "")),
            str(row.get("fwhm_quality_reason", "")),
        )
        for row in rows
    )
    report_rows = [
        {
            "track_id": track_id,
            "quality_reason": reason,
            "count": count,
        }
        for (track_id, reason), count in sorted(
            counts.items(),
            key=lambda item: (
                _frame_sort_key(item[0][0]),
                item[0][1] != "ok",
                item[0][1],
            ),
        )
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["track_id", "quality_reason", "count"],
        )
        writer.writeheader()
        writer.writerows(report_rows)
