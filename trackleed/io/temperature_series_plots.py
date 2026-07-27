"""Temperature-axis plots for stored TrackLEED Gaussian-fit results."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from trackleed.analysis.fwhm_series import numeric_sort_key
from trackleed.io.fwhm_series_plots import (
    PALETTE,
    PlotSeries,
    _render_line_plot,
)


REQUIRED_COLUMNS = {
    "track_id",
    "frame_id",
    "temperature",
    "fwhm_effective",
    "fwhm_profile_x",
    "fwhm_profile_y",
    "amplitude",
}


@dataclass(frozen=True)
class TemperatureSeriesPlotResult:
    """Files created by a non-destructive temperature-series plot export."""

    output_dir: Path
    plot_paths: tuple[Path, ...]
    manifest_path: Path


def export_temperature_series_plots(
    source_csv: Path | str,
    output_dir: Path | str,
) -> TemperatureSeriesPlotResult:
    """Plot FWHM and amplitude against temperature for every stored track."""

    source_path = Path(source_csv)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                "Source CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Source CSV contains no result rows")

    rows_by_track_frame: dict[
        str,
        dict[str, dict[str, str]],
    ] = defaultdict(dict)
    frame_temperatures: dict[str, float] = {}
    for row in rows:
        track_id = str(row["track_id"])
        frame_id = str(row["frame_id"])
        temperature = _finite_float(row["temperature"])
        if temperature is None:
            raise ValueError(
                f"Invalid temperature for track {track_id}, frame {frame_id}"
            )
        previous_temperature = frame_temperatures.setdefault(
            frame_id,
            temperature,
        )
        if not math.isclose(previous_temperature, temperature):
            raise ValueError(
                f"Inconsistent temperatures stored for frame {frame_id}"
            )
        rows_by_track_frame[track_id][frame_id] = row

    frame_ids = sorted(frame_temperatures, key=numeric_sort_key)
    track_ids = sorted(rows_by_track_frame, key=numeric_sort_key)
    temperatures = [frame_temperatures[frame_id] for frame_id in frame_ids]

    destination.mkdir(parents=True)
    effective_path = destination / (
        "fwhm_effective_vs_temperature_by_track.png"
    )
    _render_line_plot(
        effective_path,
        title="Effective FWHM vs temperature",
        x_label="Temperature (deg C)",
        y_label="Effective FWHM (px)",
        series=[
            PlotSeries(
                label=f"Track {track_id}",
                color=PALETTE[index % len(PALETTE)],
                line_width=3,
                x_values=temperatures,
                y_values=_track_values(
                    rows_by_track_frame[track_id],
                    frame_ids,
                    "fwhm_effective",
                ),
            )
            for index, track_id in enumerate(track_ids)
        ],
    )

    profile_path = destination / (
        "fwhm_profile_x_y_vs_temperature_by_track.png"
    )
    profile_series = []
    for index, track_id in enumerate(track_ids):
        track_rows = rows_by_track_frame[track_id]
        color = PALETTE[index % len(PALETTE)]
        profile_series.extend(
            [
                PlotSeries(
                    label=f"Track {track_id} profile X",
                    color=color,
                    line_width=4,
                    x_values=temperatures,
                    y_values=_track_values(
                        track_rows,
                        frame_ids,
                        "fwhm_profile_x",
                    ),
                ),
                PlotSeries(
                    label=f"Track {track_id} profile Y",
                    color=color,
                    line_width=1,
                    x_values=temperatures,
                    y_values=_track_values(
                        track_rows,
                        frame_ids,
                        "fwhm_profile_y",
                    ),
                ),
            ]
        )
    _render_line_plot(
        profile_path,
        title="Image-axis FWHM profiles vs temperature",
        x_label="Temperature (deg C)",
        y_label="FWHM (px)",
        series=profile_series,
    )

    amplitude_path = destination / "amplitude_vs_temperature_by_track.png"
    _render_line_plot(
        amplitude_path,
        title="Gaussian amplitude vs temperature",
        x_label="Temperature (deg C)",
        y_label="Amplitude",
        series=[
            PlotSeries(
                label=f"Track {track_id}",
                color=PALETTE[index % len(PALETTE)],
                line_width=3,
                x_values=temperatures,
                y_values=_track_values(
                    rows_by_track_frame[track_id],
                    frame_ids,
                    "amplitude",
                ),
            )
            for index, track_id in enumerate(track_ids)
        ],
    )

    manifest_path = destination / "temperature_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_csv": str(source_path.resolve()),
                "source_sha256": _sha256(source_path),
                "x_axis": "temperature",
                "fwhm_scalar": "fwhm_effective",
                "fwhm_image_axes": [
                    "fwhm_profile_x",
                    "fwhm_profile_y",
                ],
                "amplitude": "amplitude",
                "track_count": len(track_ids),
                "frame_count": len(frame_ids),
                "row_count": len(rows),
                "plots": [
                    effective_path.name,
                    profile_path.name,
                    amplitude_path.name,
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TemperatureSeriesPlotResult(
        output_dir=destination,
        plot_paths=(effective_path, profile_path, amplitude_path),
        manifest_path=manifest_path,
    )


def _track_values(
    rows_by_frame: dict[str, dict[str, str]],
    frame_ids: list[str],
    column: str,
) -> list[float]:
    values = []
    for frame_id in frame_ids:
        row = rows_by_frame.get(frame_id)
        value = _finite_float(row.get(column)) if row else None
        values.append(value if value is not None else math.nan)
    return values


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
