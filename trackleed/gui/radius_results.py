"""Radius plots and tables with independent mean/circle summary selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from trackleed.core.data_models import LeedFile, Track


RADIUS_SUMMARIES = (("Average radius", "average"), ("Circle-fit radius", "circle_fit"))
_X_FIELDS = {"Temperature": "temperature", "Energy": "energy", "Coverage": "coverage"}


@dataclass(frozen=True)
class _Series:
    x: np.ndarray
    radius: np.ndarray
    error: np.ndarray


def draw_radius_plot(
    axes: Axes,
    tracks: Sequence[Track],
    files: Mapping[int, LeedFile],
    *,
    selected_track_ids: set[int],
    x_axis: str,
    summary_mode: str,
    show_errors: bool,
    highlighted_track_id: int | None = None,
) -> str:
    """Render existing measurements only; return a user-facing summary note."""
    axes.clear()
    colors = colormaps["tab10"].colors
    for index, track in enumerate(tracks):
        if track.track_id not in selected_track_ids:
            continue
        points = []
        for frame_id, result in sorted(getattr(track, "analysis_results", {}).items()):
            result = result or {}
            x = _x_value(frame_id, files, x_axis)
            radius = _number(result.get("radius"))
            if x is not None and radius is not None:
                points.append((x, radius, _number(result.get("radius_error"))))
        series = _series(points)
        highlighted = track.track_id == highlighted_track_id
        style = dict(color=colors[index % len(colors)], label=f"Track {track.track_id}",
                     linewidth=3.0 if highlighted else 1.5, alpha=1.0 if highlighted else 0.7)
        if show_errors:
            axes.errorbar(series.x, series.radius, yerr=series.error, fmt="o",
                          markersize=4, capsize=3, **style)
        else:
            axes.plot(series.x, series.radius, linestyle="-", **style)
    summary = _summary_series(tracks, files, selected_track_ids, x_axis, summary_mode)
    label = "Circle-fit radius" if summary_mode == "circle_fit" else "Average Radius"
    if summary.x.size:
        axes.plot(summary.x, summary.radius, color="black", linewidth=2,
                  marker="o" if summary_mode == "circle_fit" else None, label=label)
        if show_errors:
            finite = np.isfinite(summary.error)
            axes.fill_between(summary.x, summary.radius - summary.error,
                              summary.radius + summary.error, where=finite,
                              color="black", alpha=0.2)
            if summary_mode == "circle_fit" and np.any(finite):
                axes.errorbar(summary.x[finite], summary.radius[finite],
                              yerr=summary.error[finite], fmt="none", color="black", capsize=3)
    axes.set_xlabel(x_axis)
    axes.set_ylabel("Radius (pixels)")
    axes.set_title(f"Spot Radius vs. {x_axis}")
    axes.grid(True)
    return _summary_note(summary, summary_mode)


def populate_radius_table(
    table: QTableWidget,
    tracks: Sequence[Track],
    files: Mapping[int, LeedFile],
    *,
    include_circle: bool,
) -> None:
    """Show per-track radii and an optional published circle R +/- u column."""
    frame_ids = {frame_id for track in tracks for frame_id in getattr(track, "analysis_results", {})}
    if include_circle:
        frame_ids.update(frame_id for frame_id, frame in files.items() if frame.pattern_center)
    headers = [f"Track {track.track_id}" for track in tracks]
    if include_circle:
        headers.append("Circle-fit R +/- u (px)")
    table.clear()
    table.setRowCount(len(frame_ids))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setVerticalHeaderLabels([str(frame_id) for frame_id in sorted(frame_ids)])
    for row, frame_id in enumerate(sorted(frame_ids)):
        values = []
        for track in tracks:
            result = getattr(track, "analysis_results", {}).get(frame_id) or {}
            values.append((result.get("radius"), result.get("radius_error")))
        if include_circle:
            frame = files.get(frame_id)
            center = (frame.pattern_center or {}) if frame is not None else {}
            values.append((center.get("circle_radius"), center.get("circle_radius_error")))
        for column, (radius, error) in enumerate(values):
            item = QTableWidgetItem(_radius_text(radius, error))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, column, item)
    table.resizeColumnsToContents()


def _summary_series(
    tracks: Sequence[Track],
    files: Mapping[int, LeedFile],
    selected: set[int],
    x_axis: str,
    mode: str,
) -> _Series:
    points = []
    for frame_id, frame in sorted(files.items()):
        x = _x_value(frame_id, files, x_axis)
        if x is None:
            continue
        if mode == "circle_fit":
            center = frame.pattern_center or {}
            radius = _number(center.get("circle_radius"))
            error = _number(center.get("circle_radius_error"))
        else:
            radii = [
                value for track in tracks if track.track_id in selected
                if (value := _number((getattr(track, "analysis_results", {}).get(frame_id) or {}).get("radius")))
                is not None
            ]
            if len(radii) < 2:
                continue
            radius = float(np.mean(radii))
            error = float(np.std(radii) / np.sqrt(len(radii)))
        if radius is not None:
            points.append((x, radius, error))
    return _series(sorted(points, key=lambda point: point[0]))


def _series(points: list[tuple[float, float, float | None]]) -> _Series:
    values = np.asarray(
        [(x, radius, np.nan if error is None else error) for x, radius, error in points],
        dtype=float,
    ).reshape((-1, 3))
    return _Series(values[:, 0], values[:, 1], values[:, 2])


def _summary_note(series: _Series, mode: str) -> str:
    if mode != "circle_fit":
        return "Average of selected tracks; at least two valid radii per frame."
    if not series.x.size:
        return "No fitted-circle radius. Run Analysis with a circle-fit center mode."
    note = ("Circle uses all analyzed tracks. Display selection affects individual curves only. "
            "Use spots from one ring. Error bars show fit uncertainty (1 sigma).")
    missing = int(np.sum(~np.isfinite(series.error)))
    if missing:
        note += (f" Uncertainty unavailable for {missing} frame(s): three-point, "
                 "smoothed or older results. Run Analysis to update older results.")
    return note


def _x_value(frame_id: int, files: Mapping[int, LeedFile], x_axis: str) -> float | None:
    if x_axis == "Frame ID":
        return float(frame_id)
    frame = files.get(frame_id)
    return _number(getattr(frame, _X_FIELDS.get(x_axis, ""), None))


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _radius_text(radius: object, error: object) -> str:
    radius, error = _number(radius), _number(error)
    if radius is None:
        return "N/A"
    return f"{radius:.2f} +/- {error:.2f}" if error is not None else f"{radius:.2f} (u: N/A)"
