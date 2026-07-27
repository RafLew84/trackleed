"""Dependency-light PNG plots for canonical FWHM series exports."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Iterable


PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def write_fwhm_plots(
    output_dir: Path,
    rows: list[dict[str, str]],
) -> tuple[Path, ...]:
    """Write the three standard quality-filtered FWHM plots."""

    rows_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_track[str(row.get("track_id", ""))].append(row)
    for track_rows in rows_by_track.values():
        track_rows.sort(key=lambda row: _sort_key(row.get("frame_id", "")))
    track_ids = sorted(rows_by_track, key=_sort_key)

    effective_path = output_dir / "fwhm_effective_by_track.png"
    _render_line_plot(
        effective_path,
        title="Equivalent FWHM by track",
        y_label="Equivalent FWHM (px)",
        series=[
            (
                f"Track {track_id}",
                PALETTE[index % len(PALETTE)],
                3,
                _numeric_frames(rows_by_track[track_id]),
                _valid_values(rows_by_track[track_id], "fwhm_effective"),
            )
            for index, track_id in enumerate(track_ids)
        ],
    )

    major_minor_path = output_dir / "fwhm_major_minor_by_track.png"
    major_minor_series = []
    for index, track_id in enumerate(track_ids):
        track_rows = rows_by_track[track_id]
        frames = _numeric_frames(track_rows)
        color = PALETTE[index % len(PALETTE)]
        major_minor_series.extend(
            [
                (
                    f"Track {track_id} major",
                    color,
                    4,
                    frames,
                    _valid_values(track_rows, "fwhm_major"),
                ),
                (
                    f"Track {track_id} minor",
                    color,
                    1,
                    frames,
                    _valid_values(track_rows, "fwhm_minor"),
                ),
            ]
        )
    _render_line_plot(
        major_minor_path,
        title="Major and minor FWHM by track",
        y_label="FWHM (px)",
        series=major_minor_series,
    )

    anisotropy_path = output_dir / "fwhm_anisotropy_by_track.png"
    _render_line_plot(
        anisotropy_path,
        title="FWHM anisotropy by track",
        y_label="Major / minor",
        reference_y=1.0,
        series=[
            (
                f"Track {track_id}",
                PALETTE[index % len(PALETTE)],
                3,
                _numeric_frames(rows_by_track[track_id]),
                _valid_values(rows_by_track[track_id], "fwhm_anisotropy"),
            )
            for index, track_id in enumerate(track_ids)
        ],
    )

    return effective_path, major_minor_path, anisotropy_path


def _numeric_frames(rows: Iterable[dict[str, str]]) -> list[float]:
    return [float(row["frame_id"]) for row in rows]


def _valid_values(
    rows: Iterable[dict[str, str]],
    column: str,
) -> list[float]:
    return [
        float(row[column]) if row["fwhm_valid"] == "true" else math.nan
        for row in rows
    ]


def _render_line_plot(
    output_path: Path,
    *,
    title: str,
    y_label: str,
    series: list[tuple[str, str, int, list[float], list[float]]],
    reference_y: float | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 900
    left, top, right, bottom = 130, 100, 1240, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_font = _load_font(ImageFont, size=22)
    small_font = _load_font(ImageFont, size=18)
    title_font = _load_font(ImageFont, size=32, bold=True)

    all_x = [value for _, _, _, x_values, _ in series for value in x_values]
    all_y = [
        value
        for _, _, _, _, y_values in series
        for value in y_values
        if math.isfinite(value)
    ]
    x_min, x_max = _plot_limits(all_x)
    y_min, y_max = _plot_limits(all_y, include=reference_y)

    def map_x(value: float) -> float:
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def map_y(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for tick_index in range(6):
        fraction = tick_index / 5.0
        x_value = x_min + fraction * (x_max - x_min)
        x_pixel = map_x(x_value)
        draw.line((x_pixel, top, x_pixel, bottom), fill="#E5E7EB", width=1)
        draw.text(
            (x_pixel - 24, bottom + 12),
            _format_tick(x_value),
            fill="#111827",
            font=small_font,
        )

        y_value = y_min + fraction * (y_max - y_min)
        y_pixel = map_y(y_value)
        draw.line((left, y_pixel, right, y_pixel), fill="#E5E7EB", width=1)
        draw.text(
            (30, y_pixel - 11),
            _format_tick(y_value),
            fill="#111827",
            font=small_font,
        )

    draw.line((left, top, left, bottom), fill="#111827", width=2)
    draw.line((left, bottom, right, bottom), fill="#111827", width=2)
    if reference_y is not None and y_min <= reference_y <= y_max:
        y_pixel = map_y(reference_y)
        draw.line((left, y_pixel, right, y_pixel), fill="#6B7280", width=2)

    for _, color, line_width, x_values, y_values in series:
        previous: tuple[float, float] | None = None
        for x_value, y_value in zip(x_values, y_values):
            if not math.isfinite(y_value):
                previous = None
                continue
            current = map_x(x_value), map_y(y_value)
            if previous is not None:
                draw.line((*previous, *current), fill=color, width=line_width)
            previous = current

    draw.text((left, 35), title, fill="#111827", font=title_font)
    draw.text(
        ((left + right) / 2 - 50, bottom + 58),
        "Frame ID",
        fill="#111827",
        font=regular_font,
    )
    draw.text((30, 65), y_label, fill="#111827", font=regular_font)

    legend_x, legend_y = right + 35, top
    for label, color, line_width, _, _ in series:
        draw.line(
            (legend_x, legend_y + 10, legend_x + 42, legend_y + 10),
            fill=color,
            width=max(2, line_width),
        )
        draw.text(
            (legend_x + 55, legend_y),
            label,
            fill="#111827",
            font=small_font,
        )
        legend_y += 30

    image.save(output_path, format="PNG", dpi=(180, 180))


def _plot_limits(
    values: list[float],
    *,
    include: float | None = None,
) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if include is not None:
        finite_values.append(include)
    if not finite_values:
        return 0.0, 1.0
    minimum, maximum = min(finite_values), max(finite_values)
    if math.isclose(minimum, maximum):
        padding = max(1.0, abs(minimum) * 0.05)
    else:
        padding = 0.05 * (maximum - minimum)
    return minimum - padding, maximum + padding


def _format_tick(value: float) -> str:
    return format(value, ".4g")


def _load_font(font_module, *, size: int, bold: bool = False):
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return font_module.truetype(font_name, size)
    except OSError:
        return font_module.load_default()


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value
