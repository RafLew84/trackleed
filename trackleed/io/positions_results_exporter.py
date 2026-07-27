"""CSV serialization for stored Analyze Positions results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from trackleed.analysis.result_parameters import (
    GAUSSIAN_DERIVED_EXPORT_FIELDS,
)


POSITIONS_ANALYSIS_BASE_FIELDS: tuple[str, ...] = (
    "frame_id",
    "group_id",
    "point_role",
    "track_id",
    "distance_from_reference",
    "x",
    "y",
    "position_source",
    "fit_ok",
    "quality_status",
    "quality_flags",
)

# Keep the historical columns in their original order, then append the new
# unambiguous width and orientation fields.
POSITIONS_ANALYSIS_LEGACY_FIT_FIELDS: tuple[str, ...] = (
    "sigma_x",
    "sigma_y",
    "fwhm_x",
    "fwhm_y",
    "fwhm_x_error",
    "fwhm_y_error",
    "theta",
)
POSITIONS_ANALYSIS_NEW_FIT_FIELDS: tuple[str, ...] = (
    "theta_error",
    *GAUSSIAN_DERIVED_EXPORT_FIELDS,
    "theta_major",
    "theta_major_error",
)
POSITIONS_ANALYSIS_FIELDS: tuple[str, ...] = (
    *POSITIONS_ANALYSIS_BASE_FIELDS,
    *POSITIONS_ANALYSIS_LEGACY_FIT_FIELDS,
    "r_squared",
    *POSITIONS_ANALYSIS_NEW_FIT_FIELDS,
)
POSITIONS_ANALYSIS_FIT_FIELDS: tuple[str, ...] = (
    *POSITIONS_ANALYSIS_LEGACY_FIT_FIELDS,
    *POSITIONS_ANALYSIS_NEW_FIT_FIELDS,
)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_positions_fit_export_fields(
    fit_payload: Mapping[str, object] | None,
    *,
    from_fit: bool,
    raw_fit_is_current: bool = True,
) -> dict[str, object]:
    """Extract exportable fit fields without loading images or repeating fits."""
    payload = fit_payload if isinstance(fit_payload, Mapping) else {}
    raw_fit_results = payload.get("fit_results")
    fit_results = (
        raw_fit_results
        if isinstance(raw_fit_results, Mapping)
        else {}
    )
    fit_ok = bool(payload.get("fit_ok", False)) and from_fit
    export_fit_results = fit_results if raw_fit_is_current else {}
    r_squared = _finite_float(
        payload.get("r_squared", fit_results.get("r_squared"))
    )

    return {
        "position_source": "gaussian_fit" if from_fit else "tracked_center",
        "fit_ok": fit_ok,
        "quality_status": payload.get("quality_status", "failed"),
        "quality_flags": payload.get("quality_flags") or [],
        **{
            field_name: export_fit_results.get(field_name)
            for field_name in POSITIONS_ANALYSIS_FIT_FIELDS
        },
        "r_squared": r_squared,
    }


def _csv_safe_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def write_positions_results_csv(
    output_path: Path,
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write deterministic Analyze Positions rows using the shared schema."""
    ordered_rows = sorted(
        rows,
        key=lambda current: (
            current.get("frame_id", -1),
            current.get("group_id", -1),
            str(current.get("point_role", "")),
            current.get("track_id", -1),
        ),
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSITIONS_ANALYSIS_FIELDS)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    field_name: _csv_safe_value(row.get(field_name))
                    for field_name in POSITIONS_ANALYSIS_FIELDS
                }
            )
    return output_path
