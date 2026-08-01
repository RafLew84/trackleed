"""Four-panel renderer for publication Figure 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from figure_contract import (
    FIGURE_BASENAME,
    FONT_SIZES_PT,
    METHOD_COLORS,
    METHOD_ORDER,
    NEUTRAL_COLOR,
    ORACLE_METHOD,
    SOURCE_DATA_DIRECTORY,
    as_float,
    as_int,
    load_all_method_metrics,
    load_translation_source_data,
)


def _panel_label(axis: object, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["panel_label"],
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _style_axis(axis: object) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _render_panel_a(
    axis: object,
    crop: object,
    crop_metadata: dict[str, object],
    translation_rows: Sequence[dict[str, object]],
) -> None:
    from matplotlib.patches import FancyArrowPatch, Rectangle

    axis.imshow(crop, cmap="gray", origin="upper", vmin=0, vmax=255)
    seed_x = as_float(crop_metadata["seed_x_crop_px"])
    seed_y = as_float(crop_metadata["seed_y_crop_px"])
    path_x = [seed_x + as_float(row["dx_px"]) for row in translation_rows]
    path_y = [seed_y + as_float(row["dy_px"]) for row in translation_rows]
    axis.plot(path_x, path_y, color="black", linewidth=2.8, alpha=0.65)
    axis.plot(path_x, path_y, color="#56B4E9", linewidth=1.4)
    for start, end in zip(range(len(path_x) - 1), range(1, len(path_x))):
        axis.add_patch(
            FancyArrowPatch(
                (path_x[start], path_y[start]),
                (path_x[end], path_y[end]),
                arrowstyle="-|>",
                mutation_scale=7.5,
                linewidth=0.9,
                color="#56B4E9",
                shrinkA=3,
                shrinkB=3,
            )
        )
    axis.scatter(
        path_x[:-1],
        path_y[:-1],
        s=22,
        facecolor="#56B4E9",
        edgecolor="black",
        linewidth=0.55,
        zorder=4,
    )
    axis.scatter(
        [path_x[-1]],
        [path_y[-1]],
        s=42,
        facecolor="none",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    labels = ("0/5", "1", "2", "3", "4")
    offsets = ((3, -5), (-11, -4), (-11, 8), (3, -4), (3, 8))
    for x, y, label, (offset_x, offset_y) in zip(
        path_x[:-1],
        path_y[:-1],
        labels,
        offsets,
    ):
        axis.text(
            x + offset_x,
            y + offset_y,
            label,
            color="white",
            fontsize=FONT_SIZES_PT["legend"],
            fontweight="bold",
        )
    roi_size = as_float(crop_metadata["roi_size_original_px"])
    axis.add_patch(
        Rectangle(
            (seed_x - roi_size / 2.0, seed_y - roi_size / 2.0),
            roi_size,
            roi_size,
            fill=False,
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            edgecolor="white",
            alpha=0.9,
        )
    )
    tick_offsets = (-40, 0, 40)
    axis.set_xticks([seed_x + offset for offset in tick_offsets])
    axis.set_xticklabels([str(offset) for offset in tick_offsets])
    axis.set_yticks([seed_y + offset for offset in tick_offsets])
    axis.set_yticklabels([str(offset) for offset in tick_offsets])
    axis.set_xlabel("Detector x offset [px]")
    axis.set_ylabel("Detector y offset [px]")
    axis.set_title(
        r"Closed translation path on Cu(111), 10 $^\circ$C",
        loc="left",
    )
    _panel_label(axis, "a")


def _render_panel_b(
    axis: object,
    method_rows: Sequence[dict[str, object]],
    oracle_row: dict[str, object],
) -> None:
    import numpy as np
    from matplotlib.lines import Line2D

    x_positions = np.arange(len(method_rows))
    for index, row in enumerate(method_rows):
        method = str(row["method"])
        native = as_float(row["native_error_median_px"])
        measured = as_float(row["measurement_error_median_px"])
        axis.plot(
            [index, index],
            [native, measured],
            color=METHOD_COLORS[method],
            linewidth=1.1,
            alpha=0.8,
        )
        axis.scatter(
            [index],
            [native],
            s=29,
            facecolor="white",
            edgecolor=METHOD_COLORS[method],
            linewidth=1.1,
            zorder=3,
        )
        axis.scatter(
            [index],
            [measured],
            s=29,
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    axis.axhline(
        as_float(oracle_row["measurement_error_median_px"]),
        color=NEUTRAL_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
    )
    axis.set_yscale("log")
    axis.set_ylim(1e-3, 50.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [str(row["display_method"]) for row in method_rows],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Median equivariance error [px]")
    axis.set_title("Proposal error vs final position error", loc="left")
    axis.grid(axis="y", which="both", alpha=0.16, linewidth=0.55)
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
                label="Native proposal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=NEUTRAL_COLOR,
                markerfacecolor=NEUTRAL_COLOR,
                linewidth=0,
                label="Gaussian position",
            ),
            Line2D(
                [0],
                [0],
                color=NEUTRAL_COLOR,
                linestyle=(0, (3, 2)),
                linewidth=0.9,
                label="Oracle ROI median",
            ),
        ],
        frameon=False,
        loc="upper left",
        handlelength=1.5,
    )
    _panel_label(axis, "b")


def _render_panel_c(
    axis: object,
    method_rows: Sequence[dict[str, object]],
) -> None:
    import numpy as np

    y_positions = np.arange(len(method_rows))[::-1]
    for y, row in zip(y_positions, method_rows):
        method = str(row["method"])
        coverage = 100.0 * as_float(row["measurement_coverage"])
        valid = as_int(row["valid_gaussian_measurements"])
        total = as_int(row["evaluation_spot_opportunities"])
        axis.plot([90.0, coverage], [y, y], color="#D9D9D9", linewidth=1.1)
        axis.scatter(
            [coverage],
            [y],
            s=34,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.text(
            coverage + 0.25,
            y,
            f"{valid}/{total}",
            va="center",
            fontsize=FONT_SIZES_PT["value_annotation"],
        )
    axis.axvline(100.0, color=NEUTRAL_COLOR, linewidth=0.6, alpha=0.5)
    axis.set_xlim(90.0, 102.7)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row["display_method"] for row in method_rows])
    axis.set_xlabel("Valid Gaussian measurements [%]")
    axis.set_title("Measurement coverage (80 opportunities/method)", loc="left")
    axis.grid(axis="x", alpha=0.15, linewidth=0.55)
    _panel_label(axis, "c")


def _render_panel_d(
    axis: object,
    method_rows: Sequence[dict[str, object]],
    oracle_row: dict[str, object],
) -> None:
    import numpy as np

    y_positions = np.arange(len(method_rows))[::-1]
    for y, row in zip(y_positions, method_rows):
        method = str(row["method"])
        value = as_float(row["abs_delta_d_over_d_max_pp"])
        valid = as_int(row["valid_pair_measurements"])
        total = as_int(row["pair_opportunities"])
        axis.scatter(
            [value],
            [y],
            s=34,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.text(
            value / 1.10 if value > 0.03 else value * 1.10,
            y,
            f"{value:.4f} ({valid}/{total})",
            va="center",
            ha="right" if value > 0.03 else "left",
            fontsize=FONT_SIZES_PT["value_annotation"],
        )
    axis.axvline(
        as_float(oracle_row["abs_delta_d_over_d_max_pp"]),
        color=NEUTRAL_COLOR,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        label="Oracle ROI",
    )
    axis.set_xscale("log")
    axis.set_xlim(7e-4, 1.45e-1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row["display_method"] for row in method_rows])
    axis.set_xlabel(r"Maximum $|\Delta D/D|$ [percentage points]")
    axis.set_title("Pair invariance under common translation", loc="left")
    axis.grid(axis="x", which="both", alpha=0.16, linewidth=0.55)
    axis.legend(frameon=False, loc="upper right")
    _panel_label(axis, "d")


def render_figure(output_directory: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    required_source_files = (
        "translation_path.csv",
        "method_metrics.csv",
        "representative_cu_crop.png",
        "representative_cu_crop.json",
        "source_provenance.json",
    )
    for filename in required_source_files:
        path = SOURCE_DATA_DIRECTORY / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing figure source data: {path}; use --refresh-source-data"
            )

    translation_rows = load_translation_source_data()
    all_method_rows = load_all_method_metrics(
        SOURCE_DATA_DIRECTORY / "method_metrics.csv"
    )
    method_rows = [
        row for row in all_method_rows if row["method"] in METHOD_ORDER
    ]
    oracle_row = next(
        row for row in all_method_rows if row["method"] == ORACLE_METHOD
    )
    crop_metadata = json.loads(
        (SOURCE_DATA_DIRECTORY / "representative_cu_crop.json").read_text(
            encoding="utf-8"
        )
    )
    with Image.open(SOURCE_DATA_DIRECTORY / "representative_cu_crop.png") as image:
        crop = np.asarray(image)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
    figure = plt.figure(figsize=(7.2, 5.3), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.985,
        bottom=0.11,
        top=0.94,
        hspace=0.52,
        wspace=0.42,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    _render_panel_a(axis_a, crop, crop_metadata, translation_rows)
    _render_panel_b(axis_b, method_rows, oracle_row)
    _render_panel_c(axis_c, method_rows)
    _render_panel_d(axis_d, method_rows, oracle_row)
    for axis in (axis_a, axis_b, axis_c, axis_d):
        _style_axis(axis)

    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{FIGURE_BASENAME}.png"
    pdf_path = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png_path,
        dpi=400,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure 2 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Controlled estimator behavior",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure 2",
            "Creator": "TrackLEED Figure 2 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}
