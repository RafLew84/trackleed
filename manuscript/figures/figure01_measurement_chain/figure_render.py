"""Four-panel renderer for publication Figure 1."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from figure_contract import (
    DISPLAY_NAMES,
    EVIDENCE_ROLES,
    FIGURE_BASENAME,
    FIGURE_SIZE_INCHES,
    FONT_FAMILY,
    FONT_SIZES_PT,
    LIGHT_NEUTRAL,
    MEASUREMENT_COLOR,
    METHOD_COLORS,
    NEUTRAL_COLOR,
    PNG_DPI,
    REPRESENTATION_LABELS,
    gaussian_in_crop,
    load_contract,
    load_crop_metadata,
    load_dam4sam_mask,
    load_method_outputs,
    load_native_geometry,
    mask_extent_in_crop,
    ostrack_bbox_in_crop,
    proposal_in_crop,
)


def _panel_heading(
    axis: object,
    label: str,
    title: str,
    *,
    title_x: float = 0.037,
) -> None:
    axis.text(
        0.0,
        1.055,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["panel_label"],
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axis.text(
        title_x,
        1.055,
        title,
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["axis_title"],
        va="bottom",
        ha="left",
    )


def _show_crop(axis: object, crop_image: object) -> None:
    axis.imshow(
        crop_image,
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=153,
        interpolation="nearest",
    )
    axis.set_xlim(0, 128)
    axis.set_ylim(128, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#B7B7B7")
        spine.set_linewidth(0.55)


def _draw_point_proposal(
    axis: object,
    x: float,
    y: float,
    method: str,
) -> None:
    marker = {
        "locotrack_base_offline": "o",
        "lucas_kanade_skip_anchor": "^",
        "easyleed_kalman": "D",
    }[method]
    axis.scatter(
        [x],
        [y],
        marker=marker,
        s=43,
        facecolor=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )
    axis.scatter(
        [x],
        [y],
        marker="+",
        s=29,
        color="black",
        linewidth=0.65,
        zorder=6,
    )


def _render_panel_a(
    figure: object,
    spec: object,
    crop_image: object,
    crop_metadata: dict[str, object],
    method_rows: Sequence[dict[str, object]],
    native_geometry: dict[str, object],
    dam4sam_mask: object,
) -> object:
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    outer = figure.add_subplot(spec)
    outer.set_axis_off()
    _panel_heading(
        outer,
        "a",
        "Heterogeneous native outputs for the same Cu(111) spot",
    )
    outer.text(
        1.0,
        1.055,
        r"56 eV, 92 $^\circ$C",
        transform=outer.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        va="bottom",
        ha="right",
    )

    child_grid = spec.subgridspec(1, 5, wspace=0.10)
    mask_extent = mask_extent_in_crop(crop_metadata)
    for index, row in enumerate(method_rows):
        method = str(row["method"])
        axis = figure.add_subplot(child_grid[0, index])
        _show_crop(axis, crop_image)
        proposal_x, proposal_y = proposal_in_crop(row, crop_metadata)

        if method == "ostrack_vitb_256_mae_ce":
            bbox_x, bbox_y, bbox_w, bbox_h = ostrack_bbox_in_crop(
                native_geometry,
                crop_metadata,
            )
            axis.add_patch(
                Rectangle(
                    (bbox_x, bbox_y),
                    bbox_w,
                    bbox_h,
                    fill=False,
                    edgecolor=METHOD_COLORS[method],
                    linewidth=1.6,
                    zorder=5,
                )
            )
        elif method == "dam4sam_sam21pp_l":
            visible_mask = np.ma.masked_where(~dam4sam_mask, dam4sam_mask)
            axis.imshow(
                visible_mask,
                extent=mask_extent,
                origin="upper",
                interpolation="nearest",
                cmap=ListedColormap([METHOD_COLORS[method]]),
                alpha=0.62,
                vmin=0,
                vmax=1,
                zorder=4,
            )
            axis.scatter(
                [proposal_x],
                [proposal_y],
                marker="+",
                s=32,
                color="white",
                linewidth=0.8,
                zorder=6,
            )
        else:
            _draw_point_proposal(axis, proposal_x, proposal_y, method)

        axis.set_title(
            f"{DISPLAY_NAMES[method]}\n{REPRESENTATION_LABELS[method]}",
            fontsize=FONT_SIZES_PT["method"],
            linespacing=1.2,
            pad=3.5,
        )
        if index == 0:
            axis.plot([8, 28], [118, 118], color="white", linewidth=1.5)
            axis.text(
                18,
                113,
                "20 px",
                color="white",
                fontsize=FONT_SIZES_PT["minor_annotation"],
                ha="center",
                va="bottom",
            )
    outer.text(
        0.5,
        -0.055,
        (
            "Empirical point, bbox, mask, optical-flow point and Kalman "
            "position; same image and spot identity."
        ),
        transform=outer.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="center",
        va="top",
    )
    return outer


def _render_panel_b(
    axis: object,
    crop_image: object,
    crop_metadata: dict[str, object],
    method_rows: Sequence[dict[str, object]],
    contract: dict[str, object],
) -> None:
    import numpy as np
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

    axis.set_axis_off()
    _panel_heading(
        axis,
        "b",
        "Common proposal-to-measurement conversion",
    )
    inset_bounds = (
        (0.045, 0.14, 0.215, 0.72),
        (0.392, 0.14, 0.215, 0.72),
        (0.739, 0.14, 0.215, 0.72),
    )
    stages = (
        "1  Reduced proposal",
        r"2  Fixed $48 \times 48$ px ROI",
        "3  Elliptical Gaussian position",
    )
    insets = []
    for bounds, title in zip(inset_bounds, stages):
        inset = axis.inset_axes(bounds)
        _show_crop(inset, crop_image)
        inset.set_title(
            title,
            fontsize=FONT_SIZES_PT["stage"],
            fontweight="bold",
            pad=2.7,
        )
        insets.append(inset)

    proposal_axis, roi_axis, gaussian_axis = insets
    for row in method_rows:
        method = str(row["method"])
        proposal_x, proposal_y = proposal_in_crop(row, crop_metadata)
        proposal_axis.scatter(
            [proposal_x],
            [proposal_y],
            s=24,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        roi_axis.add_patch(
            Rectangle(
                (
                    float(row["roi_x0_original_px"])
                    - float(crop_metadata["crop_x0_original_px"]),
                    float(row["roi_y0_original_px"])
                    - float(crop_metadata["crop_y0_original_px"]),
                ),
                float(row["roi_size_px"]),
                float(row["roi_size_px"]),
                fill=False,
                edgecolor=METHOD_COLORS[method],
                linewidth=0.85,
                alpha=0.92,
            )
        )
        roi_axis.scatter(
            [proposal_x],
            [proposal_y],
            s=10,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )

    gaussian_x = np.mean(
        [gaussian_in_crop(row, crop_metadata)[0] for row in method_rows]
    )
    gaussian_y = np.mean(
        [gaussian_in_crop(row, crop_metadata)[1] for row in method_rows]
    )
    sigma_x = np.mean([float(row["sigma_x_px"]) for row in method_rows])
    sigma_y = np.mean([float(row["sigma_y_px"]) for row in method_rows])
    for multiplier, alpha in ((4.0, 0.34), (3.0, 0.55), (2.0, 0.85)):
        gaussian_axis.add_patch(
            Ellipse(
                (gaussian_x, gaussian_y),
                width=multiplier * sigma_x,
                height=multiplier * sigma_y,
                fill=False,
                edgecolor=MEASUREMENT_COLOR,
                linewidth=0.75,
                alpha=alpha,
            )
        )
    for row in method_rows:
        method = str(row["method"])
        center_x, center_y = gaussian_in_crop(row, crop_metadata)
        gaussian_axis.scatter(
            [center_x],
            [center_y],
            s=16,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )
    gaussian_axis.scatter(
        [gaussian_x],
        [gaussian_y],
        marker="+",
        s=48,
        color="white",
        linewidth=1.0,
        zorder=6,
    )
    gaussian_axis.set_xlim(38, 90)
    gaussian_axis.set_ylim(90, 38)

    for start, end, label in (
        ((0.276, 0.48), (0.377, 0.48), "centers ROI"),
        ((0.623, 0.48), (0.724, 0.48), "fits raw pixels"),
    ):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color=NEUTRAL_COLOR,
            )
        )
        axis.text(
            (start[0] + end[0]) / 2,
            0.56,
            label,
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["minor_annotation"],
            color=NEUTRAL_COLOR,
            ha="center",
            va="bottom",
        )

    proposal_spread = contract["descriptive_geometry"][
        "maximum_pairwise_native_proposal_separation_px"
    ]
    gaussian_spread = contract["descriptive_geometry"][
        "maximum_pairwise_gaussian_center_separation_px"
    ]
    proposal_axis.text(
        0.5,
        0.04,
        f"max separation {proposal_spread:.2f} px",
        transform=proposal_axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color="white",
        ha="center",
        va="bottom",
    )
    gaussian_axis.text(
        0.5,
        0.04,
        f"max separation {gaussian_spread:.4f} px",
        transform=gaussian_axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color="white",
        ha="center",
        va="bottom",
    )
    axis.text(
        0.5,
        0.015,
        (
            "The tracker locates the measurement region; the common Gaussian "
            "fit defines the quantitative spot position."
        ),
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["representation"],
        fontweight="bold",
        color=NEUTRAL_COLOR,
        ha="center",
        va="bottom",
    )


def _render_panel_c(axis: object) -> None:
    from matplotlib.patches import Ellipse, FancyArrowPatch

    axis.set_axis_off()
    _panel_heading(axis, "c", "Pairwise physical endpoint", title_x=0.06)

    first = (0.22, 0.60)
    second = (0.78, 0.60)
    for center in (first, second):
        for width, height, alpha in (
            (0.23, 0.17, 0.10),
            (0.16, 0.12, 0.16),
            (0.09, 0.07, 0.25),
        ):
            axis.add_patch(
                Ellipse(
                    center,
                    width=width,
                    height=height,
                    transform=axis.transAxes,
                    facecolor=MEASUREMENT_COLOR,
                    edgecolor=MEASUREMENT_COLOR,
                    linewidth=0.55,
                    alpha=alpha,
                )
            )
        axis.scatter(
            [center[0]],
            [center[1]],
            transform=axis.transAxes,
            s=28,
            color=MEASUREMENT_COLOR,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )

    axis.add_patch(
        FancyArrowPatch(
            first,
            second,
            transform=axis.transAxes,
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=1.0,
            color=NEUTRAL_COLOR,
            shrinkA=7,
            shrinkB=7,
        )
    )
    axis.text(
        first[0],
        first[1] + 0.14,
        r"$\mathbf{p}_{i,t}$",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["formula"],
        ha="center",
    )
    axis.text(
        second[0],
        second[1] + 0.14,
        r"$\mathbf{p}_{j,t}$",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["formula"],
        ha="center",
    )
    axis.text(
        0.50,
        0.64,
        r"$D_t$",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["formula"],
        color=NEUTRAL_COLOR,
        ha="center",
        va="bottom",
    )
    axis.text(
        0.50,
        0.35,
        r"$D_t=\|\mathbf{p}_{i,t}-\mathbf{p}_{j,t}\|$",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["formula"],
        ha="center",
        va="center",
    )
    axis.text(
        0.50,
        0.18,
        r"$\Delta D/D=(D_t-D_{\mathrm{ref}})/D_{\mathrm{ref}}$",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["formula"],
        ha="center",
        va="center",
    )
    axis.text(
        0.50,
        0.035,
        "Common image translation cancels; non-rigid distortion does not.",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="center",
        va="bottom",
    )


def _render_panel_d(axis: object) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.set_axis_off()
    _panel_heading(
        axis,
        "d",
        "Four complementary evidence roles",
        title_x=0.06,
    )
    role_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    row_centers = (0.79, 0.58, 0.37, 0.16)
    for index, (item, color, center_y) in enumerate(
        zip(EVIDENCE_ROLES, role_colors, row_centers),
        start=1,
    ):
        axis.add_patch(
            FancyBboxPatch(
                (0.04, center_y - 0.075),
                0.92,
                0.15,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                transform=axis.transAxes,
                facecolor="#F7F7F7",
                edgecolor=LIGHT_NEUTRAL,
                linewidth=0.55,
            )
        )
        axis.scatter(
            [0.095],
            [center_y],
            transform=axis.transAxes,
            s=125,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.text(
            0.095,
            center_y,
            str(index),
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["representation"],
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=4,
        )
        axis.text(
            0.16,
            center_y + 0.024,
            str(item["role"]),
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["stage"],
            fontweight="bold",
            ha="left",
            va="center",
        )
        axis.text(
            0.16,
            center_y - 0.034,
            str(item["regime"]),
            transform=axis.transAxes,
            fontsize=FONT_SIZES_PT["representation"],
            color=NEUTRAL_COLOR,
            ha="left",
            va="center",
        )
    axis.text(
        0.5,
        0.015,
        "No pooled score: separate questions and denominators.",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_PT["minor_annotation"],
        color=NEUTRAL_COLOR,
        ha="center",
        va="bottom",
    )


def render_figure(output_directory: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    contract = load_contract()
    crop_metadata = load_crop_metadata()
    method_rows = load_method_outputs()
    native_geometry = load_native_geometry()
    dam4sam_mask = load_dam4sam_mask()
    with Image.open(
        Path(__file__).resolve().parent
        / "source_data"
        / "representative_spot_crop.png"
    ) as image:
        crop_image = np.asarray(image.convert("L"))

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.38, 1.45, 1.36),
        left=0.055,
        right=0.985,
        bottom=0.052,
        top=0.958,
        hspace=0.36,
        wspace=0.22,
    )
    _render_panel_a(
        figure,
        grid[0, :],
        crop_image,
        crop_metadata,
        method_rows,
        native_geometry,
        dam4sam_mask,
    )
    axis_b = figure.add_subplot(grid[1, :])
    axis_c = figure.add_subplot(grid[2, 0])
    axis_d = figure.add_subplot(grid[2, 1])
    _render_panel_b(
        axis_b,
        crop_image,
        crop_metadata,
        method_rows,
        contract,
    )
    _render_panel_c(axis_c)
    _render_panel_d(axis_d)

    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{FIGURE_BASENAME}.png"
    pdf_path = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure 1 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Measurement chain and evidence ladder",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure 1",
            "Creator": "TrackLEED Figure 1 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}
