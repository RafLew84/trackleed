"""Five-panel renderer for publication Figure 5."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from figure_contract import (
    ACCEPTED_COLOR,
    ADVERSE_COLOR,
    DISPLAY_NAMES,
    FIGURE_BASENAME,
    FIGURE_SIZE_INCHES,
    FONT_FAMILY,
    FONT_SIZES_PT,
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_ORDER,
    NEUTRAL_COLOR,
    OUTPUT_DIRECTORY,
    PNG_DPI,
    load_aggregate_source_data,
    load_pair_source_data,
    load_representative_crops,
    load_representative_gap_metadata,
    load_representative_gap_source_data,
    shared_crop_intensity_limits,
)


PAIR_ORDER = ("pair 1", "pair 2")
PAIR_LABELS = {"pair 1": "P1", "pair 2": "P2"}


def _panel_label(
    axis: object,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.06,
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


def _ordered_pair_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_identity = {
        (str(row["pair"]), str(row["method"])): row for row in rows
    }
    return [
        by_identity[(pair, method)]
        for pair in PAIR_ORDER
        for method in METHOD_ORDER
    ]


def _pair_row_layout(
    rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[float], list[str]]:
    ordered = _ordered_pair_rows(rows)
    y_positions = [6.0, 5.0, 4.0, 2.6, 1.6, 0.6]
    labels = [
        f"{PAIR_LABELS[str(row['pair'])]}  {DISPLAY_NAMES[str(row['method'])]}"
        for row in ordered
    ]
    return ordered, y_positions, labels


def _render_panel_a(
    figure: object,
    spec: object,
    gap_rows: Sequence[dict[str, object]],
    metadata: dict[str, object],
    crops: dict[float, object],
) -> None:
    from matplotlib.lines import Line2D

    grid = spec.subgridspec(
        5,
        4,
        height_ratios=(0.42, 1.0, 1.0, 1.0, 0.25),
        hspace=0.10,
        wspace=0.04,
    )
    header = figure.add_subplot(grid[0, :])
    header.set_axis_off()
    _panel_label(header, "a", x=-0.055, y=0.42)
    header.text(
        0.0,
        0.72,
        "Visibility-gap mechanism: proposal versus common Gaussian fit",
        transform=header.transAxes,
        fontsize=FONT_SIZES_PT["axis_title"],
        fontweight="bold",
        va="center",
        ha="left",
    )
    header.text(
        1.0,
        0.72,
        "Controlled scan, P1-B1, decreasing E",
        transform=header.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        va="center",
        ha="right",
    )

    energies = [float(value) for value in metadata["display_energies_ev"]]
    role_by_energy: dict[float, str] = {}
    for energy in energies:
        roles = {
            str(row["sequence_role"])
            for row in gap_rows
            if float(row["energy_ev"]) == energy
        }
        if len(roles) != 1:
            raise ValueError(
                f"Figure 5 sequence role is ambiguous at {energy:g} eV"
            )
        role_by_energy[energy] = roles.pop()
    vmin, vmax = shared_crop_intensity_limits()
    x0, y0, _x1, _y1 = [
        float(value) for value in metadata["crop_box_xyxy"]
    ]
    by_identity = {
        (str(row["method"]), float(row["energy_ev"])): row
        for row in gap_rows
    }

    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, energy in enumerate(energies):
            axis = figure.add_subplot(grid[row_index + 1, column_index])
            axis.imshow(
                crops[energy],
                cmap="gray",
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            source = by_identity[(method, energy)]
            native_x = source["native_x"]
            native_y = source["native_y"]
            gaussian_x = source["gaussian_center_x"]
            gaussian_y = source["gaussian_center_y"]

            if bool(source["native_visible"]) and native_x is not None:
                axis.scatter(
                    [float(native_x) - x0],
                    [float(native_y) - y0],
                    s=34,
                    marker="o",
                    facecolors="none",
                    edgecolors=METHOD_COLORS[method],
                    linewidths=1.25,
                    zorder=4,
                )
            else:
                axis.text(
                    0.96,
                    0.06,
                    "no proposal",
                    transform=axis.transAxes,
                    fontsize=FONT_SIZES_PT["minor_annotation"],
                    color="white",
                    ha="right",
                    va="bottom",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.58,
                        "edgecolor": "none",
                        "pad": 1.2,
                    },
                    zorder=5,
                )

            if gaussian_x is not None and gaussian_y is not None:
                accepted = bool(source["measurement_valid"])
                axis.scatter(
                    [float(gaussian_x) - x0],
                    [float(gaussian_y) - y0],
                    s=46 if accepted else 34,
                    marker="+" if accepted else "x",
                    color=ACCEPTED_COLOR if accepted else ADVERSE_COLOR,
                    linewidths=1.45,
                    zorder=5,
                )
            axis.set_xlim(-0.5, 127.5)
            axis.set_ylim(127.5, -0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#B7B7B7")
                spine.set_linewidth(0.55)
            if column_index == 0:
                axis.set_ylabel(
                    DISPLAY_NAMES[method],
                    rotation=90,
                    labelpad=4.5,
                    fontweight="bold",
                    color=METHOD_COLORS[method],
                )
            if row_index == 0:
                axis.set_title(
                    f"{energy:.1f} eV\n{role_by_energy[energy]}",
                    fontsize=FONT_SIZES_PT["tick_label"],
                    pad=2.5,
                )

    legend_axis = figure.add_subplot(grid[4, :])
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=NEUTRAL_COLOR,
                markeredgewidth=1.1,
                markersize=5.2,
                label="native proposal",
            ),
            Line2D(
                [0],
                [0],
                marker="+",
                color=ACCEPTED_COLOR,
                linewidth=0,
                markeredgewidth=1.4,
                markersize=6.5,
                label="accepted Gaussian center",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color=ADVERSE_COLOR,
                linewidth=0,
                markeredgewidth=1.3,
                markersize=5.5,
                label="rejected Gaussian center",
            ),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.0, 0.45),
        ncol=3,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    legend_axis.text(
        1.0,
        0.45,
        (
            f"shared linear scale {vmin:g}-{vmax:g}; explanatory case only"
        ),
        transform=legend_axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="right",
        va="center",
    )


def _render_panel_b(
    axis: object,
    pair_rows: Sequence[dict[str, object]],
) -> None:
    ordered, y_positions, labels = _pair_row_layout(pair_rows)
    for row, y in zip(ordered, y_positions):
        method = str(row["method"])
        coverage = 100.0 * float(row["fit_coverage"])
        axis.plot([65.0, coverage], [y, y], color="#D9D9D9", linewidth=1.15)
        axis.scatter(
            [coverage],
            [y],
            s=39,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.text(
            coverage - 0.8 if coverage > 94.0 else coverage + 0.8,
            y,
            (
                f"{int(row['branch_consistent_measurements'])}/"
                f"{int(row['visible_opportunities'])}"
            ),
            fontsize=FONT_SIZES_PT["value_annotation"],
            ha="right" if coverage > 94.0 else "left",
            va="center",
        )
    axis.axhline(3.3, color="#C9C9C9", linewidth=0.6)
    axis.set_xlim(65.0, 102.0)
    axis.set_ylim(0.0, 6.6)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Branch-consistent coverage [%; denominator 91 per pair]")
    axis.set_title("Coverage by frozen pair", loc="left")
    _panel_label(axis, "b", x=-0.18)
    _style_numeric_axis(axis)


def _render_panel_c(
    axis: object,
    pair_rows: Sequence[dict[str, object]],
) -> None:
    from matplotlib.lines import Line2D

    ordered, y_positions, labels = _pair_row_layout(pair_rows)
    for row, y in zip(ordered, y_positions):
        method = str(row["method"])
        denominator = int(row["evaluated_gaps"])
        recovered = int(row["recovered_gaps"])
        zero_delay = int(row["zero_delay_recoveries"])
        recovery_rate = 100.0 * recovered / denominator
        zero_delay_rate = 100.0 * zero_delay / denominator
        axis.plot(
            [zero_delay_rate, recovery_rate],
            [y, y],
            color=METHOD_COLORS[method],
            linewidth=1.25,
            zorder=2,
        )
        axis.scatter(
            [recovery_rate],
            [y],
            s=47,
            marker=METHOD_MARKERS[method],
            facecolor="white",
            edgecolor=METHOD_COLORS[method],
            linewidth=1.2,
            zorder=3,
        )
        axis.scatter(
            [zero_delay_rate],
            [y],
            s=19,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.3,
            zorder=4,
        )
        axis.text(
            recovery_rate - 0.5 if recovery_rate > 94.0 else recovery_rate,
            y + 0.27,
            f"{recovered}/{denominator}; now {zero_delay}/{denominator}",
            fontsize=FONT_SIZES_PT["value_annotation"],
            ha="right" if recovery_rate > 94.0 else "left",
            va="bottom",
        )
    axis.axhline(3.3, color="#C9C9C9", linewidth=0.6)
    axis.set_xlim(60.0, 102.0)
    axis.set_ylim(0.0, 6.6)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel(
        "Post-gap branch-consistent recovery\n[% of evaluable I(V) gaps]"
    )
    axis.set_title(
        "Total and zero-delay\nbranch-consistent recovery",
        loc="left",
    )
    _panel_label(axis, "c", x=-0.18)
    _style_numeric_axis(axis)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=NEUTRAL_COLOR,
                markerfacecolor="white",
                markeredgecolor=NEUTRAL_COLOR,
                linewidth=0,
                markersize=5.0,
                label="total",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=NEUTRAL_COLOR,
                markerfacecolor=NEUTRAL_COLOR,
                linewidth=0,
                markersize=3.7,
                label="zero delay",
            ),
        ],
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.60, 0.50),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )


def _render_panel_d(
    figure: object,
    spec: object,
    pair_rows: Sequence[dict[str, object]],
) -> None:
    from matplotlib.lines import Line2D

    grid = spec.subgridspec(1, 2, wspace=0.35)
    left = figure.add_subplot(grid[0, 0])
    right = figure.add_subplot(grid[0, 1])
    ordered, y_positions, labels = _pair_row_layout(pair_rows)

    for row, y in zip(ordered, y_positions):
        method = str(row["method"])
        fraction = 100.0 * float(row["geometry_inconsistent_fraction"])
        left.plot([0.0, fraction], [y, y], color="#E2D4D4", linewidth=1.15)
        left.scatter(
            [fraction],
            [y],
            s=37,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        left.text(
            fraction + 0.55,
            y,
            (
                f"{int(row['geometry_inconsistent_accepted_fits'])}/"
                f"{int(row['accepted_measurements_all_frames'])}"
            ),
            fontsize=FONT_SIZES_PT["value_annotation"],
            va="center",
        )

        median = float(row["conditional_median_abs_distance_residual_px"])
        p95 = float(row["conditional_p95_abs_distance_residual_px"])
        right.plot(
            [median, p95],
            [y, y],
            color=METHOD_COLORS[method],
            linewidth=1.35,
            alpha=0.82,
        )
        right.scatter(
            [median],
            [y],
            s=29,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        right.scatter(
            [p95],
            [y],
            s=32,
            marker="|",
            color=METHOD_COLORS[method],
            linewidth=1.5,
            zorder=3,
        )
        right.text(
            p95 + 0.08,
            y,
            f"{median:.2f}-{p95:.2f}  n={int(row['jointly_consistent_pair_frames'])}",
            fontsize=FONT_SIZES_PT["value_annotation"],
            va="center",
        )

    for axis in (left, right):
        axis.axhline(3.3, color="#C9C9C9", linewidth=0.6)
        axis.set_ylim(0.0, 6.6)
        axis.set_yticks(y_positions)
        _style_numeric_axis(axis)
    left.set_yticklabels(labels)
    right.set_yticklabels([])
    left.set_xlim(0.0, 19.0)
    right.set_xlim(0.0, 5.25)
    left.set_title("Possible false continuity", loc="left")
    right.set_title(r"Conditional pair-distance residual $D(E)$", loc="left")
    left.set_xlabel("Outside geometry gate [% of accepted fits]")
    right.set_xlabel("Median to P95 absolute residual [px]")
    _panel_label(left, "d", x=-0.18)
    right.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=NEUTRAL_COLOR,
                linewidth=0,
                markersize=4.4,
                label="median",
            ),
            Line2D(
                [0],
                [0],
                marker="|",
                color=NEUTRAL_COLOR,
                linewidth=0,
                markeredgewidth=1.5,
                markersize=6.0,
                label="P95",
            ),
        ],
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.80, 0.50),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )


def _render_panel_e(
    axis: object,
    aggregate_rows: Sequence[dict[str, object]],
    pair_rows: Sequence[dict[str, object]],
) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    axis.set_axis_off()
    _panel_label(axis, "e", x=-0.055, y=1.02)
    axis.text(
        0.0,
        1.03,
        "Mechanistic readout under a matched SAM2-family control",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["axis_title"],
        fontweight="bold",
        va="bottom",
    )
    zero_delay_by_method = {
        method: sum(
            int(row["zero_delay_recoveries"])
            for row in pair_rows
            if row["method"] == method
        )
        for method in METHOD_ORDER
    }
    mechanism_lines = {
        "sam2": "Continuity loss after gaps",
        "samurai": "Motion-aware persistence",
        "dam4sam": "Distractor-aware suppression",
    }
    by_method = {str(row["method"]): row for row in aggregate_rows}
    card_width = 0.295
    card_x = (0.005, 0.3525, 0.700)
    for method, x in zip(METHOD_ORDER, card_x):
        row = by_method[method]
        color = METHOD_COLORS[method]
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.18),
                card_width,
                0.68,
                boxstyle="round,pad=0.009,rounding_size=0.018",
                transform=axis.transAxes,
                facecolor=to_rgba(color, 0.08),
                edgecolor=color,
                linewidth=0.9,
            )
        )
        axis.text(
            x + 0.018,
            0.75,
            DISPLAY_NAMES[method],
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["card_header"],
            fontweight="bold",
            color=color,
        )
        axis.text(
            x + card_width - 0.018,
            0.75,
            str(row["memory_strategy"]),
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["minor_annotation"],
            color=NEUTRAL_COLOR,
            ha="right",
        )
        metrics = (
            (
                "coverage",
                f"{int(row['branch_consistent_measurements'])}/"
                f"{int(row['visible_opportunities'])}",
            ),
            (
                "branch-consistent\nrecovery",
                f"{int(row['recovered_gaps'])}/{int(row['evaluated_gaps'])}"
                f" (now {zero_delay_by_method[method]}/13)",
            ),
            (
                "inconsistent",
                f"{int(row['geometry_inconsistent_accepted_fits'])}/"
                f"{int(row['accepted_measurements_all_frames'])}",
            ),
        )
        for index, (label, value) in enumerate(metrics):
            y = 0.60 - 0.14 * index
            axis.text(
                x + 0.018,
                y,
                label,
                transform=axis.transAxes,
                fontsize=FONT_SIZES_PT["minor_annotation"],
                color=NEUTRAL_COLOR,
            )
            axis.text(
                x + card_width - 0.018,
                y,
                value,
                transform=axis.transAxes,
                fontsize=FONT_SIZES_PT["card_value"],
                fontweight="bold",
                ha="right",
            )
        axis.text(
            x + card_width / 2.0,
            0.235,
            mechanism_lines[method],
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["card_value"],
            fontweight="bold",
            color=color,
            ha="center",
        )
    axis.text(
        0.5,
        0.08,
        (
            "Identical checkpoint weights - one bbox seed - common 2D Gaussian "
            "fit - no correction or reinitialization"
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="center",
    )
    axis.text(
        0.5,
        0.005,
        (
            "One series and two disjoint pairs: mechanism evidence, not a "
            "general method ranking."
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        fontweight="bold",
        color=NEUTRAL_COLOR,
        ha="center",
    )


def render_figure(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_rows = load_aggregate_source_data()
    pair_rows = load_pair_source_data()
    gap_rows = load_representative_gap_source_data()
    metadata = load_representative_gap_metadata()
    crops = load_representative_crops()

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
        4,
        2,
        height_ratios=(2.35, 1.08, 1.12, 1.05),
        left=0.125,
        right=0.985,
        bottom=0.035,
        top=0.975,
        hspace=0.43,
        wspace=0.43,
    )
    _render_panel_a(figure, grid[0, :], gap_rows, metadata, crops)
    _render_panel_b(figure.add_subplot(grid[1, 0]), pair_rows)
    _render_panel_c(figure.add_subplot(grid[1, 1]), pair_rows)
    _render_panel_d(figure, grid[2, :], pair_rows)
    _render_panel_e(
        figure.add_subplot(grid[3, :]),
        aggregate_rows,
        pair_rows,
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
        metadata={"Software": "TrackLEED Figure 5 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Controlled SAM2-family continuity-consistency trade-off",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure 5",
            "Creator": "TrackLEED Figure 5 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}
