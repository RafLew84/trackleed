"""Five-panel renderer for publication Figure 5."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from figure_contract import (
    ACCEPTED_COLOR,
    ADVERSE_COLOR,
    DISPLAY_NAMES,
    FIGURE_BASENAME,
    FIGURE_SIZE_INCHES,
    FONT_FAMILY,
    FONT_SIZES_PT,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_ORDER,
    NEUTRAL_COLOR,
    OUTPUT_DIRECTORY,
    PAIR_COLORS,
    PNG_DPI,
    load_between_method_spread,
    load_cycle_landmarks,
    load_pair_2_4_diagnostic,
    load_pair_endpoint_timeseries,
    load_temperature_profile,
)


@dataclass(frozen=True)
class EndpointPanelData:
    """Cross-method physical trajectory and deviations for one spot pair."""

    frame_orders: tuple[int, ...]
    median_percentage_points: tuple[float, ...]
    minimum_percentage_points: tuple[float, ...]
    maximum_percentage_points: tuple[float, ...]
    method_residuals_milli_pp: Mapping[str, tuple[float, ...]]


def prepare_endpoint_panel_data(
    rows: Sequence[dict[str, object]],
    pair: str,
) -> EndpointPanelData:
    """Build a median trajectory, min-max envelope, and method residuals."""

    by_frame: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["pair"] != pair:
            continue
        frame_order = int(row["frame_order"])
        method = str(row["method"])
        values = by_frame.setdefault(frame_order, {})
        if method in values:
            raise ValueError(
                f"Duplicate endpoint value for pair {pair}, "
                f"frame {frame_order}, method {method}"
            )
        values[method] = float(row["delta_r_over_r_percent"])
    if not by_frame:
        raise ValueError(f"No endpoint rows found for pair {pair}")

    frame_orders = tuple(sorted(by_frame))
    medians: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []
    residuals: dict[str, list[float]] = {
        method: [] for method in METHOD_ORDER
    }
    expected_methods = set(METHOD_ORDER)
    for frame_order in frame_orders:
        values = by_frame[frame_order]
        if set(values) != expected_methods:
            missing = sorted(expected_methods - set(values))
            unexpected = sorted(set(values) - expected_methods)
            raise ValueError(
                f"Incomplete endpoint panel data for pair {pair}, "
                f"frame {frame_order}: missing={missing}, "
                f"unexpected={unexpected}"
            )
        ordered_values = [values[method] for method in METHOD_ORDER]
        median = float(statistics.median(ordered_values))
        medians.append(median)
        minima.append(min(ordered_values))
        maxima.append(max(ordered_values))
        for method in METHOD_ORDER:
            residuals[method].append(1000.0 * (values[method] - median))

    return EndpointPanelData(
        frame_orders=frame_orders,
        median_percentage_points=tuple(medians),
        minimum_percentage_points=tuple(minima),
        maximum_percentage_points=tuple(maxima),
        method_residuals_milli_pp={
            method: tuple(residuals[method]) for method in METHOD_ORDER
        },
    )


def _progress(
    frame_order: int | float,
    last_frame_order: int,
) -> float:
    return 100.0 * float(frame_order) / last_frame_order


def _panel_label(
    axis: object,
    label: str,
    *,
    x: float = -0.11,
    y: float = 1.05,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["panel_label"],
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _style_numeric_axis(axis: object, *, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.5,
        alpha=0.55,
    )


def _landmarks_by_name(
    landmarks: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {str(row["landmark"]): row for row in landmarks}


def _reader_label(
    landmarks: Sequence[dict[str, object]],
    landmark: str,
) -> str:
    return str(_landmarks_by_name(landmarks)[landmark]["reader_label"])


def _decorate_cycle_axis(
    axis: object,
    landmarks: Sequence[dict[str, object]],
    *,
    last_frame_order: int,
    show_reference: bool = True,
) -> None:
    by_name = _landmarks_by_name(landmarks)
    if show_reference:
        reference = by_name["reference"]
        axis.axvspan(
            _progress(int(reference["frame_start"]), last_frame_order),
            _progress(int(reference["frame_end"]), last_frame_order),
            color="#BDBDBD",
            alpha=0.22,
            linewidth=0,
            zorder=0,
        )
    plateau = by_name["plateau"]
    axis.axvspan(
        _progress(int(plateau["frame_start"]), last_frame_order),
        _progress(int(plateau["frame_end"]), last_frame_order),
        color="#E69F00",
        alpha=0.16,
        linewidth=0,
        zorder=0,
    )
    minimum = by_name["post_plateau_minimum"]
    axis.axvline(
        _progress(int(minimum["frame_start"]), last_frame_order),
        color="#7B3294",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        alpha=0.9,
        zorder=1,
    )
    final = by_name["final"]
    axis.axvline(
        _progress(int(final["frame_start"]), last_frame_order),
        color="#008080",
        linewidth=0.8,
        linestyle=(0, (4, 2)),
        alpha=0.9,
        zorder=1,
    )


def _render_panel_a(
    axis: object,
    profile: Sequence[dict[str, object]],
    landmarks: Sequence[dict[str, object]],
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    last_frame_order = max(int(row["frame_order"]) for row in profile)
    x = [
        _progress(int(row["frame_order"]), last_frame_order)
        for row in profile
    ]
    temperature = [float(row["temperature_c"]) for row in profile]
    axis.fill_between(x, temperature, 0.0, color="#DCEAF4", alpha=0.75)
    axis.plot(x, temperature, color="#2F4B5C", linewidth=1.15)
    _decorate_cycle_axis(
        axis,
        landmarks,
        last_frame_order=last_frame_order,
    )
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(0.0, 875.0)
    axis.set_ylabel("Temperature [°C]")
    axis.set_xticks((0, 25, 50, 75, 100))
    axis.tick_params(axis="x", labelbottom=False)
    axis.set_title("Cu(111) thermal-cycle landmarks at fixed 56 eV", loc="left")
    _panel_label(axis, "a", x=-0.055, y=1.04)
    _style_numeric_axis(axis)

    by_name = _landmarks_by_name(landmarks)
    axis.annotate(
        (
            f"{_reader_label(landmarks, 'post_plateau_minimum')}\n"
            "(outcome-selected)"
        ),
        xy=(
            _progress(
                int(by_name["post_plateau_minimum"]["frame_start"]),
                last_frame_order,
            ),
            797.0,
        ),
        xytext=(61.5, 685.0),
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color="#6A1B78",
        ha="left",
        arrowprops={"arrowstyle": "-", "color": "#7B3294", "lw": 0.65},
    )
    axis.text(
        99.1,
        104.0,
        _reader_label(landmarks, "final").replace(" (", "\n("),
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color="#006C6C",
        ha="right",
        va="bottom",
    )
    axis.legend(
        handles=[
            Patch(
                facecolor="#BDBDBD",
                alpha=0.35,
                edgecolor="none",
                label=_reader_label(landmarks, "reference"),
            ),
            Patch(
                facecolor="#E69F00",
                alpha=0.30,
                edgecolor="none",
                label=_reader_label(landmarks, "plateau"),
            ),
            Line2D(
                [0],
                [0],
                color="#7B3294",
                linestyle=(0, (2, 2)),
                label=_reader_label(landmarks, "post_plateau_minimum"),
            ),
            Line2D(
                [0],
                [0],
                color="#008080",
                linestyle=(0, (4, 2)),
                label=_reader_label(landmarks, "final"),
            ),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.45,
    )


def _render_endpoint_panel(
    main_axis: object,
    residual_axis: object,
    panel: EndpointPanelData,
    landmarks: Sequence[dict[str, object]],
    *,
    pair: str,
    panel_label: str,
    residual_limit: float,
) -> None:
    last_frame_order = max(panel.frame_orders)
    x = [
        _progress(frame_order, last_frame_order)
        for frame_order in panel.frame_orders
    ]
    main_axis.fill_between(
        x,
        panel.minimum_percentage_points,
        panel.maximum_percentage_points,
        color="#80B1D3",
        alpha=0.65,
        linewidth=0,
        label="min-max envelope",
        zorder=2,
    )
    main_axis.plot(
        x,
        panel.median_percentage_points,
        color="#263F4B",
        linewidth=1.15,
        label="five-method median",
        zorder=3,
    )
    _decorate_cycle_axis(
        main_axis,
        landmarks,
        last_frame_order=last_frame_order,
    )
    values = (
        panel.minimum_percentage_points + panel.maximum_percentage_points
    )
    margin = 0.06 * (max(values) - min(values))
    main_axis.set_ylim(min(values) - margin, max(values) + margin)
    main_axis.set_xlim(0.0, 100.0)
    main_axis.axhline(0.0, color="#808080", linewidth=0.6, alpha=0.7)
    main_axis.set_xticks((0, 25, 50, 75, 100))
    main_axis.tick_params(axis="x", labelbottom=False)
    main_axis.set_ylabel(r"$\Delta R/R$ [%]")
    main_axis.set_title(f"Pair {pair}: physical trajectory", loc="left")
    _panel_label(main_axis, panel_label, x=-0.15)
    _style_numeric_axis(main_axis)
    main_axis.legend(
        frameon=False,
        loc="lower left",
        ncol=1,
        handlelength=1.5,
        handletextpad=0.45,
    )

    for method in METHOD_ORDER:
        residual_axis.plot(
            x,
            panel.method_residuals_milli_pp[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=0.8,
            alpha=0.92,
            label=DISPLAY_NAMES[method],
            zorder=2,
        )
    _decorate_cycle_axis(
        residual_axis,
        landmarks,
        last_frame_order=last_frame_order,
    )
    residual_axis.set_ylim(-residual_limit, residual_limit)
    residual_axis.set_xlim(0.0, 100.0)
    residual_axis.axhline(
        0.0,
        color="#555555",
        linewidth=0.65,
        alpha=0.85,
        zorder=1,
    )
    residual_axis.set_xticks((0, 25, 50, 75, 100))
    residual_axis.set_xlabel("Thermal-cycle progress [%]")
    residual_axis.set_ylabel("Residual")
    residual_axis.text(
        0.015,
        0.94,
        r"method - median; $\times 10^{-3}$ pp",
        transform=residual_axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="left",
        va="top",
    )
    _style_numeric_axis(residual_axis)


def _render_panel_d(
    axis: object,
    rows: Sequence[dict[str, object]],
    landmarks: Sequence[dict[str, object]],
) -> None:
    from matplotlib.patches import Patch

    point_order = ("plateau", "post_plateau_minimum", "final")
    point_labels = (
        _reader_label(landmarks, "plateau"),
        _reader_label(landmarks, "post_plateau_minimum").replace(
            " minimum (", "\nminimum\n("
        ),
        _reader_label(landmarks, "final").replace(
            " state (", "\nstate\n("
        ),
    )
    by_identity = {
        (str(row["pair"]), str(row["summary_point"])): row
        for row in rows
    }
    y_base = [2.0, 1.0, 0.0]
    offsets = {"1-6": 0.16, "3-5": -0.16}
    for pair in ("1-6", "3-5"):
        for point, y in zip(point_order, y_base):
            row = by_identity[(pair, point)]
            value = 1000.0 * float(row["spread_percentage_points"])
            position = y + offsets[pair]
            axis.barh(
                [position],
                [value],
                height=0.26,
                color=PAIR_COLORS[pair],
                alpha=0.90,
                edgecolor="white",
                linewidth=0.35,
            )
            axis.text(
                value + 0.012,
                position,
                f"{value:.3f}",
                fontsize=FONT_SIZES_PT["value_annotation"],
                va="center",
            )
    sequence = {
        str(row["pair"]): float(row["spread_percentage_points"])
        for row in rows
        if row["summary_point"] == "entire_sequence_maximum"
    }
    axis.set_xlim(0.0, 0.62)
    axis.set_ylim(-0.55, 2.55)
    axis.set_yticks(y_base)
    axis.set_yticklabels(point_labels)
    axis.set_xlabel(
        "Between-method range\n"
        r"[$\times 10^{-3}$ percentage points]"
    )
    axis.set_title(
        "Endpoint spread after common\nGaussian refinement",
        loc="left",
    )
    _panel_label(axis, "d", x=-0.16)
    _style_numeric_axis(axis, grid_axis="x")
    axis.legend(
        handles=[
            Patch(facecolor=PAIR_COLORS[pair], label=f"pair {pair}")
            for pair in ("1-6", "3-5")
        ],
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    axis.text(
        0.98,
        0.22,
        (
            "Full-sequence maximum [pp]\n"
            f"pair 1-6 = {sequence['1-6']:.6f} pp\n"
            f"pair 3-5 = {sequence['3-5']:.6f} pp"
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="right",
        va="center",
        bbox={
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
            "pad": 1.1,
        },
    )


def _render_panel_e(
    axis: object,
    rows: Sequence[dict[str, object]],
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["frame_id"]))
    x = [
        100.0 * index / (len(ordered) - 1)
        for index, _row in enumerate(ordered)
    ]
    amplitudes = [float(row["amplitude"]) for row in ordered]
    accepted = [bool(row["refit_accepted"]) for row in ordered]
    threshold = float(ordered[0]["amplitude_threshold"])
    axis.plot(x, amplitudes, color="#7A7A7A", linewidth=0.75, zorder=1)
    axis.scatter(
        [value for value, state in zip(x, accepted) if not state],
        [
            value
            for value, state in zip(amplitudes, accepted)
            if not state
        ],
        s=23,
        marker="o",
        color=ADVERSE_COLOR,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    axis.scatter(
        [value for value, state in zip(x, accepted) if state],
        [value for value, state in zip(amplitudes, accepted) if state],
        s=31,
        marker="D",
        color=ACCEPTED_COLOR,
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
    )
    axis.axhline(
        threshold,
        color="#222222",
        linestyle=(0, (4, 2)),
        linewidth=0.85,
        zorder=2,
    )
    plateau_indices = [
        index
        for index, row in enumerate(ordered)
        if float(row["temperature_c"]) == 800.0
    ]
    if plateau_indices:
        axis.axvspan(
            x[min(plateau_indices)] - 1.5,
            x[max(plateau_indices)] + 1.5,
            color="#E69F00",
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    axis.set_xlim(-1.0, 101.0)
    axis.set_ylim(0.039, 0.054)
    axis.set_xticks((0.0, 72.0, 100.0))
    axis.set_xticklabels(("before", "800 °C plateau", "after"))
    axis.set_xlabel("Local sequence around\nthe intensity minimum")
    axis.set_ylabel("Gaussian amplitude")
    axis.set_title(
        "Separate pair 2-4 missing-fit diagnostic\n"
        "(saved YOLO; outside the five-method run)",
        loc="left",
    )
    _panel_label(axis, "e", x=-0.15)
    _style_numeric_axis(axis)

    temperature_axis = axis.twinx()
    temperature_axis.plot(
        x,
        [float(row["temperature_c"]) for row in ordered],
        color="#9A9A9A",
        linestyle=(0, (1.5, 2)),
        linewidth=0.75,
        alpha=0.75,
        zorder=0,
    )
    temperature_axis.set_ylim(775.0, 805.0)
    temperature_axis.set_ylabel("Temperature [°C]", color="#777777")
    temperature_axis.tick_params(axis="y", colors="#777777")
    temperature_axis.spines["top"].set_visible(False)
    temperature_axis.spines["right"].set_color("#B5B5B5")

    axis.text(
        0.50,
        0.92,
        "proposal present 26/26\n24/24 rejected fits\nfail only amplitude gate",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        fontweight="bold",
        color=ADVERSE_COLOR,
        ha="center",
        va="top",
        bbox={
            "facecolor": "white",
            "alpha": 0.78,
            "edgecolor": "none",
            "pad": 1.0,
        },
    )
    axis.text(
        0.98,
        0.69,
        "amplitude threshold 0.05",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color="#222222",
        ha="right",
        va="top",
    )


def render_figure(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    profile = load_temperature_profile()
    landmarks = load_cycle_landmarks()
    endpoints = load_pair_endpoint_timeseries()
    spread = load_between_method_spread()
    diagnostic = load_pair_2_4_diagnostic()
    endpoint_panels = {
        pair: prepare_endpoint_panel_data(endpoints, pair)
        for pair in ("1-6", "3-5")
    }
    residual_limit = 1.08 * max(
        abs(value)
        for panel in endpoint_panels.values()
        for residuals in panel.method_residuals_milli_pp.values()
        for value in residuals
    )

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
        height_ratios=(1.10, 1.82, 1.52),
        left=0.115,
        right=0.945,
        bottom=0.065,
        top=0.975,
        hspace=0.56,
        wspace=0.50,
    )
    axis_a = figure.add_subplot(grid[0, :])
    grid_b = grid[1, 0].subgridspec(
        2, 1, height_ratios=(2.7, 1.0), hspace=0.06
    )
    grid_c = grid[1, 1].subgridspec(
        2, 1, height_ratios=(2.7, 1.0), hspace=0.06
    )
    axis_b = figure.add_subplot(grid_b[0])
    axis_b_residual = figure.add_subplot(grid_b[1], sharex=axis_b)
    axis_c = figure.add_subplot(grid_c[0])
    axis_c_residual = figure.add_subplot(grid_c[1], sharex=axis_c)
    axis_d = figure.add_subplot(grid[2, 0])
    axis_e = figure.add_subplot(grid[2, 1])

    _render_panel_a(axis_a, profile, landmarks)
    _render_endpoint_panel(
        axis_b,
        axis_b_residual,
        endpoint_panels["1-6"],
        landmarks,
        pair="1-6",
        panel_label="b",
        residual_limit=residual_limit,
    )
    _render_endpoint_panel(
        axis_c,
        axis_c_residual,
        endpoint_panels["3-5"],
        landmarks,
        pair="3-5",
        panel_label="c",
        residual_limit=residual_limit,
    )
    _render_panel_d(axis_d, spread, landmarks)
    _render_panel_e(axis_e, diagnostic)
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.0,
                label=DISPLAY_NAMES[method],
            )
            for method in METHOD_ORDER
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.756),
        ncol=5,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
        title="Method residuals",
        title_fontsize=FONT_SIZES_PT["legend"],
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
            "Title": "Cu(111) pairwise physical endpoint",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure 5",
            "Creator": "TrackLEED Figure 5 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}
