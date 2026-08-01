"""Five-panel renderer for publication Figure 4."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from figure_contract import (
    ADVERSE_COLOR,
    DISPLAY_NAMES,
    EXPECTED_COLOR,
    FIGURE_BASENAME,
    FIGURE_SIZE_INCHES,
    FONT_FAMILY,
    FONT_SIZES_PT,
    METHOD_COLORS,
    METHOD_ORDER,
    NEUTRAL_COLOR,
    OUTPUT_DIRECTORY,
    PNG_DPI,
    REGIME_COLORS,
    load_aggregate_source_data,
    load_representative_case_source_data,
    load_representative_pair_source_data,
    load_visibility_source_data,
)


@dataclass(frozen=True)
class RepresentativeTrajectorySummary:
    """Framewise cross-method summary of valid pair measurements."""

    frames: tuple[int, ...]
    energies_ev: tuple[float, ...]
    valid_method_counts: tuple[int, ...]
    median_distances_px: tuple[float, ...]
    minimum_distances_px: tuple[float, ...]
    maximum_distances_px: tuple[float, ...]

    @property
    def maximum_range_px(self) -> float:
        return max(
            maximum - minimum
            for minimum, maximum in zip(
                self.minimum_distances_px,
                self.maximum_distances_px,
            )
        )


def summarize_representative_trajectory(
    rows: Sequence[dict[str, object]],
) -> RepresentativeTrajectorySummary:
    """Summarize available valid methods without collapsing acquisitions."""

    by_frame: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        frame = int(row["frame"])
        method = str(row["method"])
        method_rows = by_frame.setdefault(frame, {})
        if method in method_rows:
            raise ValueError(
                f"Duplicate representative measurement at frame {frame}, "
                f"method {method}"
            )
        method_rows[method] = row
    if not by_frame:
        raise ValueError("Representative trajectory is empty")

    frames = tuple(sorted(by_frame))
    energies: list[float] = []
    counts: list[int] = []
    medians: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []
    for frame in frames:
        method_rows = by_frame[frame]
        unexpected = sorted(set(method_rows) - set(METHOD_ORDER))
        if unexpected:
            raise ValueError(
                f"Unexpected representative methods at frame {frame}: {unexpected}"
            )
        frame_energies = {float(row["energy_ev"]) for row in method_rows.values()}
        if len(frame_energies) != 1:
            raise ValueError(f"Inconsistent representative energy at frame {frame}")
        distances = [float(row["pair_distance_px"]) for row in method_rows.values()]
        energies.append(frame_energies.pop())
        counts.append(len(distances))
        medians.append(float(statistics.median(distances)))
        minima.append(min(distances))
        maxima.append(max(distances))

    return RepresentativeTrajectorySummary(
        frames=frames,
        energies_ev=tuple(energies),
        valid_method_counts=tuple(counts),
        median_distances_px=tuple(medians),
        minimum_distances_px=tuple(minima),
        maximum_distances_px=tuple(maxima),
    )


def _panel_label(
    axis: object, label: str, *, x: float = -0.12, y: float = 1.05
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["panel_label"],
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _style_numeric_axis(axis: object) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.7)


def _expected_pair_distance(
    visibility_rows: Sequence[dict[str, object]],
) -> tuple[list[int], list[float], list[float]]:
    by_frame: dict[int, dict[str, dict[str, object]]] = {}
    for row in visibility_rows:
        by_frame.setdefault(int(row["frame"]), {})[str(row["branch"])] = row
    frames = sorted(by_frame)
    energies: list[float] = []
    distances: list[float] = []
    for frame in frames:
        branches = by_frame[frame]
        if set(branches) != {"B1", "B2"}:
            raise ValueError(f"Incomplete representative pair at frame {frame}")
        first = branches["B1"]
        second = branches["B2"]
        energies.append(float(first["energy_ev"]))
        distances.append(
            math.hypot(
                float(first["predicted_x"]) - float(second["predicted_x"]),
                float(first["predicted_y"]) - float(second["predicted_y"]),
            )
        )
    return frames, energies, distances


def _render_visibility_ribbon(
    axis: object,
    rows: Sequence[dict[str, object]],
    branch: str,
) -> None:
    from matplotlib.patches import Rectangle

    branch_rows = sorted(
        (row for row in rows if row["branch"] == branch),
        key=lambda row: int(row["frame"]),
    )
    rows_by_energy: dict[float, list[dict[str, object]]] = {}
    for row in branch_rows:
        rows_by_energy.setdefault(float(row["energy_ev"]), []).append(row)
    energies = sorted(rows_by_energy)
    boundaries = [
        energies[0] - 0.5 * (energies[1] - energies[0]),
        *[0.5 * (lower + upper) for lower, upper in zip(energies[:-1], energies[1:])],
        energies[-1] + 0.5 * (energies[-1] - energies[-2]),
    ]
    for energy_index, energy in enumerate(energies):
        acquisitions = rows_by_energy[energy]
        height = 1.0 / len(acquisitions)
        for acquisition_index, row in enumerate(acquisitions):
            axis.add_patch(
                Rectangle(
                    (boundaries[energy_index], acquisition_index * height),
                    boundaries[energy_index + 1] - boundaries[energy_index],
                    height,
                    facecolor=REGIME_COLORS[str(row["status"])],
                    edgecolor="white",
                    linewidth=0.25,
                )
            )
    axis.set_xlim(boundaries[0], boundaries[-1])
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.5])
    axis.set_yticklabels([branch], fontweight="bold")
    axis.tick_params(axis="y", length=0, pad=4)
    axis.set_xticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _render_panel_a(
    figure: object,
    spec: object,
    trajectory_rows: Sequence[dict[str, object]],
    visibility_rows: Sequence[dict[str, object]],
    representative: dict[str, object],
) -> None:
    from matplotlib.patches import Patch

    grid = spec.subgridspec(
        3,
        1,
        height_ratios=(5.0, 0.28, 0.28),
        hspace=0.08,
    )
    axis = figure.add_subplot(grid[0, 0])
    ribbon_b1 = figure.add_subplot(grid[1, 0], sharex=axis)
    ribbon_b2 = figure.add_subplot(grid[2, 0], sharex=axis)
    _panel_label(axis, "a", x=-0.065, y=1.07)
    axis.set_title(
        "Representative pair trajectory and frozen visibility regimes",
        loc="left",
    )
    axis.text(
        1.0,
        1.04,
        (
            f"{representative['scientific_series_label']}, "
            f"{representative['scientific_pair_label']}"
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="right",
        va="bottom",
    )

    _frames, energies, expected_distances = _expected_pair_distance(visibility_rows)
    expected_line = axis.plot(
        energies,
        expected_distances,
        color=EXPECTED_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=1.15,
        label=r"Expected $D(E) \propto 1/\sqrt{E}$",
        zorder=2,
    )[0]
    trajectory = summarize_representative_trajectory(trajectory_rows)
    lower_errors = [
        median - minimum
        for median, minimum in zip(
            trajectory.median_distances_px,
            trajectory.minimum_distances_px,
        )
    ]
    upper_errors = [
        maximum - median
        for median, maximum in zip(
            trajectory.median_distances_px,
            trajectory.maximum_distances_px,
        )
    ]
    median_handle = axis.errorbar(
        trajectory.energies_ev,
        trajectory.median_distances_px,
        yerr=(lower_errors, upper_errors),
        fmt="o",
        markersize=4.0,
        color=NEUTRAL_COLOR,
        markerfacecolor=NEUTRAL_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.4,
        ecolor="#7896AB",
        elinewidth=0.9,
        capsize=2.0,
        capthick=0.8,
        label="Cross-method median; min-max",
        zorder=3,
    )
    axis.set_xlim(38.0, 223.0)
    axis.set_ylabel(r"Pair distance $D(E)$ [px]")
    axis.grid(alpha=0.16, linewidth=0.55)
    tick_energies = (40.0, 66.0, 93.5, 121.5, 149.5, 188.5, 220.5)
    axis.set_xticks(tick_energies)
    axis.tick_params(labelbottom=False)
    trajectory_legend = axis.legend(
        handles=[expected_line, median_handle],
        frameon=False,
        ncol=2,
        loc="upper right",
        columnspacing=0.9,
        handletextpad=0.4,
    )
    axis.add_artist(trajectory_legend)
    axis.text(
        0.985,
        0.865,
        (
            "available valid methods per image: "
            f"n={min(trajectory.valid_method_counts)}-"
            f"{max(trajectory.valid_method_counts)}\n"
            f"maximum min-max range = {trajectory.maximum_range_px:.3f} px "
            "(smaller than marker)"
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="right",
        va="top",
    )
    axis.legend(
        handles=[
            Patch(
                facecolor=REGIME_COLORS["visible_fit"],
                edgecolor="none",
                label="baseline-visible",
            ),
            Patch(
                facecolor=REGIME_COLORS["iv_visibility_gap"],
                edgecolor="none",
                label="I(V) gap",
            ),
            Patch(
                facecolor=REGIME_COLORS["beam_onset_or_termination"],
                edgecolor="none",
                label="beam onset/termination",
            ),
            Patch(
                facecolor=REGIME_COLORS["apparatus_absence"],
                edgecolor="none",
                label="apparatus absence",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.4,
    )

    _render_visibility_ribbon(ribbon_b1, visibility_rows, "B1")
    _render_visibility_ribbon(ribbon_b2, visibility_rows, "B2")
    ribbon_b2.set_xticks(tick_energies)
    ribbon_b2.set_xticklabels([f"{energy:g}" for energy in tick_energies])
    ribbon_b1.tick_params(axis="x", bottom=False, labelbottom=False)
    ribbon_b2.set_xlabel("Electron energy [eV]", labelpad=1.5)


def _render_lollipop_panel(
    axis: object,
    rows: Sequence[dict[str, object]],
    *,
    panel_label: str,
    title: str,
    value_field: str,
    denominator_field: str,
    x_limits: tuple[float, float],
    x_label: str,
    numerator_field: str | None = None,
    value_multiplier: float = 1.0,
    stem_color: str = "#D7D7D7",
    format_value: str = ".0f",
    conditional: bool = False,
    footer: tuple[str, str] | None = None,
) -> None:
    import numpy as np

    y_positions = np.arange(len(rows))[::-1]
    lower = x_limits[0]
    for y, row in zip(y_positions, rows):
        method = str(row["method"])
        value = float(row[value_field]) * value_multiplier
        denominator = int(row[denominator_field])
        axis.plot(
            [lower, value],
            [y, y],
            color=stem_color,
            linewidth=1.15,
            zorder=1,
        )
        axis.scatter(
            [value],
            [y],
            s=39,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        if conditional:
            label = f"{value:{format_value}}  (n={denominator})"
        else:
            numerator = int(row[numerator_field or value_field])
            label = f"{numerator}/{denominator}"
        axis.text(
            value + 0.018 * (x_limits[1] - x_limits[0]),
            y,
            label,
            fontsize=FONT_SIZES_PT["value_annotation"],
            va="center",
            ha="left",
        )
    axis.set_xlim(*x_limits)
    axis.set_ylim(-0.55, len(rows) - 0.45)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([DISPLAY_NAMES[str(row["method"])] for row in rows])
    axis.set_xlabel(x_label)
    axis.set_title(title, loc="left")
    _panel_label(axis, panel_label)
    _style_numeric_axis(axis)
    if footer is not None:
        text, color = footer
        axis.text(
            0.99,
            0.02,
            text,
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["minor_annotation"],
            color=color,
            ha="right",
            va="bottom",
        )


def render_figure(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_rows = load_aggregate_source_data()
    trajectory_rows = load_representative_pair_source_data()
    visibility_rows = load_visibility_source_data()
    representative = load_representative_case_source_data()

    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZES_PT["base"],
            "axes.titlesize": FONT_SIZES_PT["axis_title"],
            "axes.labelsize": FONT_SIZES_PT["axis_label"],
            "xtick.labelsize": FONT_SIZES_PT["tick_label"],
            "ytick.labelsize": FONT_SIZES_PT["tick_label"],
            "legend.fontsize": FONT_SIZES_PT["legend"],
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.55, 1.0, 1.0),
        left=0.13,
        right=0.985,
        bottom=0.045,
        top=0.965,
        hspace=0.48,
        wspace=0.42,
    )
    _render_panel_a(
        figure,
        grid[0, :],
        trajectory_rows,
        visibility_rows,
        representative,
    )
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])
    axis_d = figure.add_subplot(grid[2, 0])
    axis_e = figure.add_subplot(grid[2, 1])

    _render_lollipop_panel(
        axis_b,
        aggregate_rows,
        panel_label="b",
        title="Branch-consistent measurement coverage",
        value_field="branch_consistent_measurements",
        denominator_field="visible_opportunities",
        x_limits=(180.0, 223.5),
        x_label="Valid measurements [count; common denominator 221]",
    )
    _render_lollipop_panel(
        axis_c,
        aggregate_rows,
        panel_label="c",
        title="Post-gap recovery without reinitialization",
        value_field="recovered_gaps",
        denominator_field="evaluated_gaps",
        x_limits=(15.0, 25.2),
        x_label="Recovered natural I(V) gaps [count; denominator 24]",
    )
    _render_lollipop_panel(
        axis_d,
        aggregate_rows,
        panel_label="d",
        title="Geometry-inconsistent accepted fits",
        value_field="geometry_inconsistent_fraction",
        numerator_field="geometry_inconsistent_accepted_fits",
        denominator_field="accepted_measurements_all_frames",
        value_multiplier=100.0,
        stem_color="#E2D4D4",
        x_limits=(0.0, 16.2),
        x_label="Outside branch gate [% of accepted fits]",
        footer=("possible false continuity", ADVERSE_COLOR),
    )
    _render_lollipop_panel(
        axis_e,
        aggregate_rows,
        panel_label="e",
        title="Conditional median pair-distance residual\n(both branches pass all gates)",
        value_field="median_abs_distance_residual_px",
        denominator_field="jointly_consistent_pair_frames",
        x_limits=(1.70, 1.89),
        x_label=r"Median $|D_{\mathrm{measured}}-D_{\mathrm{expected}}|$ [px]",
        format_value=".3f",
        conditional=True,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{FIGURE_BASENAME}.png"
    pdf_path = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure 4 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Independent Ag(111) energy-scan stress test",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure 4",
            "Creator": "TrackLEED Figure 4 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}
