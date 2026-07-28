"""Quality-aware FWHM plots on the temperature axis."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


FWHM_REQUIRED_COLUMNS = {
    "track_id",
    "frame_id",
    "frame_order",
    "temperature_C",
    "thermal_branch",
    "plotted",
    "fwhm_x_px",
    "fwhm_y_px",
    "r_squared",
}
AMPLITUDE_REQUIRED_COLUMNS = {
    "track_id",
    "frame_id",
    "amplitude_au",
}


@dataclass(frozen=True)
class QualityFwhmPlotResult:
    """Artifacts created by a non-destructive quality-aware export."""

    plot_path: Path
    data_path: Path
    manifest_path: Path
    accepted_count: int
    rejected_count: int


# Preserve the public name used by the original quality-filtered exporter.
QualityFilteredFwhmPlotResult = QualityFwhmPlotResult


def export_quality_filtered_fwhm_temperature_plot(
    fwhm_csv: Path | str,
    amplitude_csv: Path | str,
    output_stem: Path | str,
    *,
    min_amplitude: float = 0.10,
    min_r_squared: float = 0.90,
) -> QualityFwhmPlotResult:
    """Create a new FWHM plot with unreliable Gaussian fits shown as gaps."""

    return _export_quality_fwhm_temperature_plot(
        fwhm_csv,
        amplitude_csv,
        output_stem,
        min_amplitude=min_amplitude,
        min_r_squared=min_r_squared,
        render_plot=_render_filtered_plot,
        rejected_values_are_plotted_as="gaps",
    )


def export_quality_marked_fwhm_temperature_plot(
    fwhm_csv: Path | str,
    amplitude_csv: Path | str,
    output_stem: Path | str,
    *,
    min_amplitude: float = 0.10,
    min_r_squared: float = 0.90,
) -> QualityFwhmPlotResult:
    """Create a FWHM plot that retains rejected fits as gray markers."""

    return _export_quality_fwhm_temperature_plot(
        fwhm_csv,
        amplitude_csv,
        output_stem,
        min_amplitude=min_amplitude,
        min_r_squared=min_r_squared,
        render_plot=_render_marked_plot,
        rejected_values_are_plotted_as="gray markers",
    )


def _export_quality_fwhm_temperature_plot(
    fwhm_csv: Path | str,
    amplitude_csv: Path | str,
    output_stem: Path | str,
    *,
    min_amplitude: float,
    min_r_squared: float,
    render_plot: Callable[..., None],
    rejected_values_are_plotted_as: str,
) -> QualityFwhmPlotResult:
    if min_amplitude < 0.0:
        raise ValueError("Minimum amplitude must be non-negative")
    if not 0.0 <= min_r_squared <= 1.0:
        raise ValueError("Minimum R-squared must be between zero and one")

    fwhm_path = Path(fwhm_csv)
    amplitude_path = Path(amplitude_csv)
    stem = Path(output_stem)
    plot_path = stem.with_suffix(".png")
    data_path = stem.parent / f"{stem.name}_data.csv"
    manifest_path = stem.parent / f"{stem.name}_manifest.json"
    output_paths = (plot_path, data_path, manifest_path)
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Quality-aware FWHM export does not overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )

    fwhm_fieldnames, fwhm_rows = _read_csv(
        fwhm_path,
        FWHM_REQUIRED_COLUMNS,
    )
    _, amplitude_rows = _read_csv(
        amplitude_path,
        AMPLITUDE_REQUIRED_COLUMNS,
    )
    amplitudes = {
        _row_key(row): row
        for row in amplitude_rows
    }
    if len(amplitudes) != len(amplitude_rows):
        raise ValueError("Amplitude CSV contains duplicate track/frame rows")

    exported_rows = []
    accepted_count = 0
    for source_row in fwhm_rows:
        key = _row_key(source_row)
        amplitude_row = amplitudes.get(key)
        amplitude = _finite_float(
            amplitude_row.get("amplitude_au") if amplitude_row else None
        )
        amplitude_error = (
            amplitude_row.get("amplitude_error_au", "")
            if amplitude_row
            else ""
        )
        quality_valid, quality_reason = _quality_status(
            source_row,
            amplitude,
            min_amplitude=min_amplitude,
            min_r_squared=min_r_squared,
        )
        exported_row = dict(source_row)
        exported_row["amplitude_au"] = (
            "" if amplitude is None else _format_number(amplitude)
        )
        exported_row["amplitude_error_au"] = amplitude_error
        exported_row["quality_valid"] = str(quality_valid)
        exported_row["quality_reason"] = quality_reason
        exported_rows.append(exported_row)
        accepted_count += int(quality_valid)

    stem.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = list(fwhm_fieldnames)
    for fieldname in (
        "amplitude_au",
        "amplitude_error_au",
        "quality_valid",
        "quality_reason",
    ):
        if fieldname not in output_fieldnames:
            output_fieldnames.append(fieldname)
    _write_csv(data_path, output_fieldnames, exported_rows)
    render_plot(
        plot_path,
        exported_rows,
        min_amplitude=min_amplitude,
        min_r_squared=min_r_squared,
    )

    rejected_count = len(exported_rows) - accepted_count
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fwhm_source_csv": str(fwhm_path.resolve()),
                "fwhm_source_sha256": _sha256(fwhm_path),
                "amplitude_source_csv": str(amplitude_path.resolve()),
                "amplitude_source_sha256": _sha256(amplitude_path),
                "plot": plot_path.name,
                "data": data_path.name,
                "quality_filter": {
                    "minimum_amplitude_au": min_amplitude,
                    "minimum_r_squared": min_r_squared,
                    "rejected_values_are_plotted_as": (
                        rejected_values_are_plotted_as
                    ),
                    "interpolation": False,
                },
                "row_count": len(exported_rows),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return QualityFwhmPlotResult(
        plot_path=plot_path,
        data_path=data_path,
        manifest_path=manifest_path,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )


def _quality_status(
    row: Mapping[str, str],
    amplitude: float | None,
    *,
    min_amplitude: float,
    min_r_squared: float,
) -> tuple[bool, str]:
    if row.get("plotted", "").strip().lower() != "true":
        detail = row.get("missing_reason") or row.get("quality_note")
        return False, f"source_not_plotted{':' + detail if detail else ''}"
    if amplitude is None:
        return False, "amplitude_missing"
    if amplitude < min_amplitude:
        return False, f"amplitude_below_{_format_number(min_amplitude)}"
    r_squared = _finite_float(row.get("r_squared"))
    if r_squared is None:
        return False, "r_squared_missing"
    if r_squared < min_r_squared:
        return False, f"r_squared_below_{_format_number(min_r_squared)}"
    for column in ("temperature_C", "fwhm_x_px", "fwhm_y_px"):
        if _finite_float(row.get(column)) is None:
            return False, f"{column}_missing"
    return True, "accepted"


def _render_filtered_plot(
    destination: Path,
    rows: Iterable[Mapping[str, str]],
    *,
    min_amplitude: float,
    min_r_squared: float,
) -> None:
    grouped: dict[str, dict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["track_id"])][str(row["thermal_branch"])].append(row)

    track_ids = sorted(grouped, key=_numeric_sort_key)
    if not track_ids:
        raise ValueError("FWHM CSV contains no result rows")
    columns = min(3, len(track_ids))
    rows_count = math.ceil(len(track_ids) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(16, 4.8 * rows_count),
        squeeze=False,
    )

    colors = {"x": "#0072B2", "y": "#D55E00"}
    branch_styles = {"heating": "-", "cooling": "--"}
    for axis, track_id in zip(axes.flat, track_ids):
        for branch, branch_rows in grouped[track_id].items():
            branch_rows.sort(key=lambda row: _numeric_sort_key(row["frame_order"]))
            line_style = branch_styles.get(branch.lower(), ":")
            for dimension in ("x", "y"):
                x_values, y_values = _plot_values_with_gaps(
                    branch_rows,
                    f"fwhm_{dimension}_px",
                )
                axis.plot(
                    x_values,
                    y_values,
                    color=colors[dimension],
                    linestyle=line_style,
                    linewidth=1.8,
                )
        axis.set_title(f"Track {track_id}", fontsize=14)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=11)

    for axis in axes.flat[len(track_ids):]:
        axis.set_visible(False)

    figure.supxlabel("Temperature (°C)", fontsize=15, y=0.035)
    figure.supylabel("FWHM (px)", fontsize=15)
    legend_handles = [
        Line2D([0], [0], color=colors["x"], linestyle="-", label="FWHM X – Heating"),
        Line2D([0], [0], color=colors["x"], linestyle="--", label="FWHM X – Cooling"),
        Line2D([0], [0], color=colors["y"], linestyle="-", label="FWHM Y – Heating"),
        Line2D([0], [0], color=colors["y"], linestyle="--", label="FWHM Y – Cooling"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    figure.text(
        0.5,
        0.008,
        (
            f"Quality filter: amplitude ≥ {_format_number(min_amplitude)}, "
            f"R² ≥ {_format_number(min_r_squared)}; gaps are not interpolated."
        ),
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.075, 1.0, 0.93))
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _render_marked_plot(
    destination: Path,
    rows: Iterable[Mapping[str, str]],
    *,
    min_amplitude: float,
    min_r_squared: float,
) -> None:
    grouped: dict[str, dict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["track_id"])][str(row["thermal_branch"])].append(row)

    track_ids = sorted(grouped, key=_numeric_sort_key)
    if not track_ids:
        raise ValueError("FWHM CSV contains no result rows")
    columns = min(3, len(track_ids))
    rows_count = math.ceil(len(track_ids) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(16, 4.8 * rows_count),
        squeeze=False,
        sharey=True,
    )

    colors = {"x": "#0072B2", "y": "#D55E00"}
    rejected_markers = {"x": "o", "y": "^"}
    branch_styles = {"heating": "-", "cooling": "--"}
    for axis, track_id in zip(axes.flat, track_ids):
        for branch, branch_rows in grouped[track_id].items():
            branch_rows.sort(key=lambda row: _numeric_sort_key(row["frame_order"]))
            line_style = branch_styles.get(branch.lower(), ":")
            for dimension in ("x", "y"):
                value_column = f"fwhm_{dimension}_px"
                x_values, y_values = _plot_values_with_gaps(
                    branch_rows,
                    value_column,
                )
                axis.plot(
                    x_values,
                    y_values,
                    color=colors[dimension],
                    linestyle=line_style,
                    linewidth=1.7,
                    zorder=2,
                )
                rejected_x, rejected_y = _rejected_plot_values(
                    branch_rows,
                    value_column,
                )
                axis.scatter(
                    rejected_x,
                    rejected_y,
                    color="#777777",
                    marker=rejected_markers[dimension],
                    s=13,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )
        axis.set_title(f"Track {track_id}", fontsize=14)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=11)

    for axis in axes.flat[len(track_ids):]:
        axis.set_visible(False)

    figure.supxlabel("Temperature (°C)", fontsize=15, y=0.035)
    figure.supylabel("FWHM (px)", fontsize=15)
    legend_handles = [
        Line2D([0], [0], color=colors["x"], linestyle="-", label="FWHM X – Heating"),
        Line2D([0], [0], color=colors["x"], linestyle="--", label="FWHM X – Cooling"),
        Line2D([0], [0], color=colors["y"], linestyle="-", label="FWHM Y – Heating"),
        Line2D([0], [0], color=colors["y"], linestyle="--", label="FWHM Y – Cooling"),
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="o",
            linestyle="none",
            label="Rejected by quality criteria",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.008,
        (
            f"Quality criteria: amplitude ≥ {_format_number(min_amplitude)}, "
            f"R² ≥ {_format_number(min_r_squared)}. Gray values are retained "
            "but not connected."
        ),
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.075, 1.0, 0.93))
    figure.savefig(destination, dpi=600, bbox_inches="tight")
    plt.close(figure)


def _rejected_plot_values(
    rows: Iterable[Mapping[str, str]],
    value_column: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        if row.get("quality_valid") == "True":
            continue
        temperature = _finite_float(row.get("temperature_C"))
        value = _finite_float(row.get(value_column))
        if temperature is not None and value is not None:
            x_values.append(temperature)
            y_values.append(value)
    return x_values, y_values


def _plot_values_with_gaps(
    rows: list[Mapping[str, str]],
    value_column: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    previous_order: float | None = None
    for row in rows:
        order = _finite_float(row.get("frame_order"))
        temperature = _finite_float(row.get("temperature_C"))
        value = _finite_float(row.get(value_column))
        if (
            previous_order is not None
            and order is not None
            and order > previous_order + 1
        ):
            x_values.append(math.nan)
            y_values.append(math.nan)
        x_values.append(temperature if temperature is not None else math.nan)
        y_values.append(
            value
            if row.get("quality_valid") == "True" and value is not None
            else math.nan
        )
        previous_order = order
    return x_values, y_values


def _read_csv(
    path: Path,
    required_columns: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(required_columns.difference(fieldnames))
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )
        return fieldnames, list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row["track_id"]), str(row["frame_id"])


def _numeric_sort_key(value: object) -> tuple[int, float | str]:
    try:
        return 0, float(str(value))
    except ValueError:
        return 1, str(value)


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    return f"{value:g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
