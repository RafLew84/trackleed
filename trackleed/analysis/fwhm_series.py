"""Canonical FWHM values and quality assessment for stored fit series."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import statistics
from typing import Iterable, Mapping


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
class CanonicalFwhm:
    """Axis-order-independent widths derived from one accepted Gaussian fit."""

    minor: float
    major: float
    effective: float
    anisotropy: float

    @classmethod
    def from_axes(cls, fwhm_x: float, fwhm_y: float) -> "CanonicalFwhm":
        minor = min(fwhm_x, fwhm_y)
        major = max(fwhm_x, fwhm_y)
        return cls(
            minor=minor,
            major=major,
            effective=math.sqrt(fwhm_x * fwhm_y),
            anisotropy=major / minor,
        )

    def as_csv_fields(self) -> dict[str, str]:
        return {
            "fwhm_minor": format_number(self.minor),
            "fwhm_major": format_number(self.major),
            "fwhm_effective": format_number(self.effective),
            "fwhm_anisotropy": format_number(self.anisotropy),
        }


def enrich_fwhm_rows(
    source_rows: Iterable[Mapping[str, str]],
    *,
    min_r_squared: float = 0.85,
) -> list[dict[str, str]]:
    """Return source rows enriched with canonical FWHM and quality fields."""

    rows = [
        _enrich_row(row, min_r_squared=min_r_squared) for row in source_rows
    ]
    _apply_temporal_quality(rows)
    return rows


def summarize_fwhm_per_frame(
    rows: Iterable[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Aggregate effective FWHM using only rows accepted by quality checks."""

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("frame_id", ""))].append(row)

    summaries = []
    for frame_id in sorted(grouped, key=numeric_sort_key):
        frame_rows = grouped[frame_id]
        valid_rows = [row for row in frame_rows if row["fwhm_valid"] == "true"]
        effective_values = [
            float(row["fwhm_effective"]) for row in valid_rows
        ]
        summaries.append(
            {
                "frame_id": frame_id,
                "temperature": _first_present(frame_rows, "temperature"),
                "energy": _first_present(frame_rows, "energy"),
                "coverage": _first_present(frame_rows, "coverage"),
                "fwhm_effective_mean": (
                    format_number(statistics.fmean(effective_values))
                    if effective_values
                    else ""
                ),
                "fwhm_valid_count": len(valid_rows),
                "fwhm_invalid_count": len(frame_rows) - len(valid_rows),
            }
        )
    return summaries


def numeric_sort_key(value: str) -> tuple[int, float | str]:
    """Sort numeric identifiers numerically and other identifiers lexically."""

    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def format_number(value: float) -> str:
    return format(value, ".10g")


def _enrich_row(
    source_row: Mapping[str, str],
    *,
    min_r_squared: float,
) -> dict[str, str]:
    row = dict(source_row)
    fwhm_x = _finite_float(row.get("fwhm_x"))
    fwhm_y = _finite_float(row.get("fwhm_y"))
    r_squared = _finite_float(row.get("r_squared"))

    if fwhm_x is None or fwhm_y is None:
        return _with_quality(row, valid=False, reason="missing_fwhm")
    if fwhm_x <= 0 or fwhm_y <= 0:
        return _with_quality(row, valid=False, reason="nonpositive_fwhm")

    row.update(CanonicalFwhm.from_axes(fwhm_x, fwhm_y).as_csv_fields())
    if r_squared is None:
        return _with_quality(row, valid=False, reason="missing_r_squared")
    if r_squared < min_r_squared:
        return _with_quality(
            row,
            valid=False,
            reason=f"r_squared_below_{format_number(min_r_squared)}",
        )
    return _with_quality(row, valid=True, reason="ok")


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


def _first_present(rows: Iterable[Mapping[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if value != "":
            return value
    return ""
