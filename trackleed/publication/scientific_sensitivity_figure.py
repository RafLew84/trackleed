"""Deterministic Figure S2 for measurement-window and capture-contract sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from trackleed.analysis.scientific_sensitivity_records import (
    SensitivityResultRecord,
    WidthTriggerRecord,
)
from trackleed.analysis.scientific_tracer_records import read_csv
from trackleed.publication.scientific_sensitivity_sources import (
    GATE_MARGIN_SPECS,
    GateMarginSummaryRecord,
    read_json,
    sha256_file,
    validate_sensitivity_source_data,
)


FIGURE_BASENAME = "figure_s2_measurement_sensitivity"
PART_A_D_BASENAME = f"{FIGURE_BASENAME}_a-d"
PART_E_F_BASENAME = f"{FIGURE_BASENAME}_e-f_continued"
PART_A_D_SIZE_INCHES = (7.2, 10.0)
PART_E_F_SIZE_INCHES = (7.2, 6.0)
PLACEMENT_WIDTH_INCHES = 6.25
PNG_DPI = 300
ANNOTATION_FONT_SIZE_PT = 7.0
MINIMUM_SOURCE_FONT_SIZE_PT = ANNOTATION_FONT_SIZE_PT
OTHER_FONT_SIZE_PT = 7.2
MINIMUM_EFFECTIVE_FONT_SIZE_PT = (
    MINIMUM_SOURCE_FONT_SIZE_PT * PLACEMENT_WIDTH_INCHES / PART_A_D_SIZE_INCHES[0]
)
MINIMUM_HEATMAP_CELL_SIZE_MM = 4.5
CODEBOOK = (
    "Codebook: R = ROI [px]; P = proposal-to-fit gate [px]; "
    "B = branch tolerance [px];\nR² = fit threshold; A = amplitude multiplier; "
    "* = frozen central contract."
)
METHOD_ORDER = ("locotrack", "ostrack", "dam4sam", "lk", "easyleed")
METHOD_DISPLAY_NAMES = {
    "locotrack": "LocoTrack",
    "ostrack": "OSTrack",
    "dam4sam": "DAM4SAM",
    "lk": "Lucas-Kanade",
    "easyleed": "EasyLEED",
}
MATRIX_VARIANTS = tuple(
    f"roi_{roi}_proposal_{proposal}"
    for roi in (32, 48, 64)
    for proposal in (12, 15, 18)
)
OFAT_VARIANTS = (
    "branch_tolerance_12",
    "branch_tolerance_24",
    "r_squared_0p80",
    "r_squared_0p90",
    "amplitude_multiplier_0p75",
    "amplitude_multiplier_1p25",
)
VARIANT_ORDER = MATRIX_VARIANTS + OFAT_VARIANTS
VARIANT_LABELS = {
    **{
        f"roi_{roi}_proposal_{proposal}": (
            f"R{roi}\nP{proposal}*"
            if (roi, proposal) == (48, 15)
            else f"R{roi}\nP{proposal}"
        )
        for roi in (32, 48, 64)
        for proposal in (12, 15, 18)
    },
    "branch_tolerance_12": "B12",
    "branch_tolerance_24": "B24",
    "r_squared_0p80": "R².80",
    "r_squared_0p90": "R².90",
    "amplitude_multiplier_0p75": "A.75",
    "amplitude_multiplier_1p25": "A1.25",
}
PANEL_ORDER = (
    "coverage",
    "recovery",
    "geometry_inconsistency",
    "conditional_pair_residual",
    "signed_gate_margins",
    "width_trigger_audit",
)
IMPLEMENTATION_PATHS = (
    Path("trackleed/publication/scientific_sensitivity_figure.py"),
    Path("trackleed/publication/scientific_sensitivity_sources.py"),
)


@dataclass(frozen=True)
class FigureS2Report:
    variant_count: int
    roi_matrix_variant_count: int
    ofat_variant_count: int
    method_count: int
    gate_audit_record_count: int
    gate_margin_summary_record_count: int
    width_sensitivity_triggered: bool
    part_count: int
    display_item_count: int
    minimum_effective_font_size_pt: float
    minimum_heatmap_cell_width_mm: float
    minimum_heatmap_cell_height_mm: float


def _load_figure_sources(
    source_directory: Path,
) -> tuple[
    list[SensitivityResultRecord],
    list[WidthTriggerRecord],
    list[GateMarginSummaryRecord],
    dict[str, Any],
]:
    validate_sensitivity_source_data(source_directory)
    _, results = read_csv(
        source_directory / "sensitivity_results.csv",
        SensitivityResultRecord,
    )
    method_results = [row for row in results if row["scope"] == "method"]
    _, audit = read_csv(
        source_directory / "gate_margin_audit.csv",
        WidthTriggerRecord,
    )
    _, gate_margins = read_csv(
        source_directory / "gate_margin_summary.csv",
        GateMarginSummaryRecord,
    )
    summary = read_json(source_directory / "sensitivity_summary.json")
    result_keys = {(row["variant_id"], row["method"]) for row in method_results}
    if (
        len(method_results) != 75
        or result_keys
        != {(variant, method) for variant in VARIANT_ORDER for method in METHOD_ORDER}
        or len(audit) != 25
        or len(gate_margins) != 40
        or {(row["method"], row["margin_id"]) for row in gate_margins}
        != {
            (method, margin_id)
            for method in METHOD_ORDER
            for margin_id, *_ in GATE_MARGIN_SPECS
        }
        or {row["scope"] for row in audit} != {"case", "method"}
        or set(summary.get("variants", {})) != set(VARIANT_ORDER)
    ):
        raise ValueError("Figure S2 source matrix changed")
    return method_results, audit, gate_margins, summary


def _result_matrix(
    rows: Sequence[SensitivityResultRecord],
    field: str,
) -> list[list[float]]:
    by_key = {(row["variant_id"], row["method"]): row for row in rows}
    return [
        [float(by_key[(variant, method)][field]) for variant in VARIANT_ORDER]
        for method in METHOD_ORDER
    ]


def _annotate_heatmap(
    axis: Any,
    annotations: Sequence[Sequence[str]],
    values: Sequence[Sequence[float]],
) -> None:
    midpoint = (min(min(row) for row in values) + max(max(row) for row in values)) / 2
    for y, row in enumerate(annotations):
        for x, label in enumerate(row):
            color = "white" if values[y][x] > midpoint else "#222222"
            axis.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE_PT,
                color=color,
            )


def _render_heatmap(
    figure: Any,
    axis: Any,
    rows: Sequence[SensitivityResultRecord],
    *,
    panel_label: str,
    title: str,
    value_field: str,
    annotation: Any,
    color_map: str,
    colorbar_label: str,
) -> None:
    from matplotlib.patches import Rectangle

    values = _result_matrix(rows, value_field)
    image = axis.imshow(values, aspect="auto", cmap=color_map)
    by_key = {(row["variant_id"], row["method"]): row for row in rows}
    annotations = [
        [annotation(by_key[(variant, method)]) for variant in VARIANT_ORDER]
        for method in METHOD_ORDER
    ]
    _annotate_heatmap(axis, annotations, values)
    axis.set_yticks(range(len(METHOD_ORDER)))
    axis.set_yticklabels([METHOD_DISPLAY_NAMES[method] for method in METHOD_ORDER])
    axis.set_xticks(range(len(VARIANT_ORDER)))
    axis.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
    axis.set_xticks(
        [index - 0.5 for index in range(1, len(VARIANT_ORDER))], minor=True
    )
    axis.grid(which="minor", axis="x", color="white", linewidth=0.45, alpha=0.45)
    axis.tick_params(which="minor", bottom=False)
    axis.axvline(8.5, color="white", linewidth=1.5)
    central_index = VARIANT_ORDER.index("roi_48_proposal_15")
    axis.add_patch(
        Rectangle(
            (central_index - 0.5, -0.5),
            1.0,
            len(METHOD_ORDER),
            fill=False,
            edgecolor="#D55E00",
            linewidth=1.5,
            zorder=4,
        )
    )
    axis.set_title(title, loc="left", fontsize=8.5)
    axis.text(
        -0.07,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
    )
    axis.text(
        0.29,
        -0.37,
        "ROI × proposal-to-fit matrix",
        transform=axis.transAxes,
        ha="center",
        fontsize=OTHER_FONT_SIZE_PT,
    )
    axis.text(
        0.81,
        -0.37,
        "one-factor-at-a-time perturbations",
        transform=axis.transAxes,
        ha="center",
        fontsize=OTHER_FONT_SIZE_PT,
    )
    axis.tick_params(labelsize=OTHER_FONT_SIZE_PT, length=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.024, pad=0.015)
    colorbar.set_label(colorbar_label, fontsize=OTHER_FONT_SIZE_PT)
    colorbar.ax.tick_params(labelsize=OTHER_FONT_SIZE_PT)


def _render_gate_margins(
    figure: Any,
    axis: Any,
    rows: Sequence[GateMarginSummaryRecord],
) -> None:
    import numpy as np

    margin_order = tuple(margin_id for margin_id, *_ in GATE_MARGIN_SPECS)
    by_key = {(row["method"], row["margin_id"]): row for row in rows}
    normalized = np.asarray(
        [
            [
                by_key[(method, margin)]["p05"] / by_key[(method, margin)]["normalizer"]
                for margin in margin_order
            ]
            for method in METHOD_ORDER
        ],
        dtype=float,
    )
    image = axis.imshow(
        np.clip(normalized, -1.0, 1.0),
        aspect="auto",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    for y, method in enumerate(METHOD_ORDER):
        for x, margin in enumerate(margin_order):
            row = by_key[(method, margin)]
            axis.text(
                x,
                y,
                f"{row['p05']:.2g}\n{row['negative_count']}/{row['finite_margin_n']}",
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE_PT,
                color="white" if abs(normalized[y, x]) > 0.55 else "#222222",
            )
    axis.set_xticks(range(len(margin_order)))
    axis.set_xticklabels(
        [
            "proposal\nto fit",
            "R²",
            "amplitude",
            "σx low",
            "σx high",
            "σy low",
            "σy high",
            "branch",
        ]
    )
    axis.set_yticks(range(len(METHOD_ORDER)))
    axis.set_yticklabels([METHOD_DISPLAY_NAMES[method] for method in METHOD_ORDER])
    axis.set_title(
        "Signed central-contract gate margins (P05; negative count / n)",
        loc="left",
        fontsize=8.5,
    )
    axis.text(
        -0.08,
        1.05,
        "e",
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
    )
    axis.tick_params(labelsize=OTHER_FONT_SIZE_PT, length=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("normalized signed P05 (clipped)", fontsize=OTHER_FONT_SIZE_PT)
    colorbar.ax.tick_params(labelsize=OTHER_FONT_SIZE_PT)


def _render_gate_audit(axis: Any, audit: Sequence[WidthTriggerRecord]) -> None:
    import numpy as np

    method_rows = [row for row in audit if row["scope"] == "method"]
    by_method = {row["method"]: row for row in method_rows}
    rows = [by_method[method] for method in METHOD_ORDER]
    x = np.arange(len(rows), dtype=float)
    width = 0.32
    near = [row["near_width_boundary_count"] for row in rows]
    near_threshold = [row["near_width_boundary_threshold_count"] for row in rows]
    width_only = [row["width_only_rejection_count"] for row in rows]
    width_threshold = [row["width_only_rejection_threshold_count"] for row in rows]
    axis.bar(x - width / 2, near, width, color="#56B4E9", label="near width boundary")
    axis.bar(
        x + width / 2, width_only, width, color="#CC79A7", label="width-only rejection"
    )
    axis.scatter(
        x - width / 2,
        near_threshold,
        marker="_",
        s=150,
        linewidths=1.6,
        color="#0072B2",
        label="ceil(5% finite-fit n)",
        zorder=3,
    )
    axis.scatter(
        x + width / 2,
        width_threshold,
        marker="_",
        s=150,
        linewidths=1.6,
        color="#A23B72",
        label="12/221 width-only threshold",
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels([METHOD_DISPLAY_NAMES[method] for method in METHOD_ORDER])
    axis.set_ylabel("Candidate/rejection count")
    axis.set_title(
        "Width-trigger audit: neither per-method materiality trigger activated",
        loc="left",
        fontsize=8.5,
    )
    axis.text(
        -0.035,
        1.05,
        "f",
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
    )
    axis.text(
        0.99,
        0.48,
        "validated decision: width sensitivity not run",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=OTHER_FONT_SIZE_PT,
        color="#555555",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=OTHER_FONT_SIZE_PT)
    axis.legend(frameon=False, ncol=1, fontsize=OTHER_FONT_SIZE_PT, loc="upper left")


def render_figure_s2(
    source_directory: Path,
    output_directory: Path,
) -> FigureS2Report:
    """Render Figure S2 independently of ambient Matplotlib settings."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        return _render_figure_s2(source_directory, output_directory, plt)


def _render_figure_s2(
    source_directory: Path,
    output_directory: Path,
    plt: Any,
) -> FigureS2Report:
    rows, audit, gate_margins, summary = _load_figure_sources(
        source_directory.resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        obsolete = output_directory / f"{FIGURE_BASENAME}{suffix}"
        if obsolete.exists():
            obsolete.unlink()

    part_a_d, axes = plt.subplots(4, 1, figsize=PART_A_D_SIZE_INCHES)
    part_a_d.subplots_adjust(
        left=0.14,
        right=0.88,
        top=0.93,
        bottom=0.11,
        hspace=0.95,
    )
    _render_heatmap(
        part_a_d,
        axes[0],
        rows,
        panel_label="a",
        title="Branch-consistent coverage (x / 221 fixed opportunities)",
        value_field="branch_consistent_measurements",
        annotation=lambda row: f"{row['branch_consistent_measurements']}/\n221",
        color_map="Blues",
        colorbar_label="branch-consistent measurements",
    )
    _render_heatmap(
        part_a_d,
        axes[1],
        rows,
        panel_label="b",
        title="Post-gap branch-consistent recovery (x / 24 fixed gaps)",
        value_field="branch_consistent_recovered_gap_count",
        annotation=lambda row: f"{row['branch_consistent_recovered_gap_count']}/\n24",
        color_map="Greens",
        colorbar_label="recovered gaps",
    )
    _render_heatmap(
        part_a_d,
        axes[2],
        rows,
        panel_label="c",
        title="Geometry-inconsistent continuation (events / accepted fits)",
        value_field="geometry_inconsistent_fraction",
        annotation=lambda row: (
            f"{row['geometry_inconsistent_accepted_fits']}/\n"
            f"{row['geometry_inconsistency_denominator']}"
        ),
        color_map="Oranges",
        colorbar_label="fraction of accepted fits",
    )
    _render_heatmap(
        part_a_d,
        axes[3],
        rows,
        panel_label="d",
        title="Conditional median pair-distance residual (value; pair-frame n)",
        value_field="median_absolute_pair_distance_residual_px",
        annotation=lambda row: (
            f"{row['median_absolute_pair_distance_residual_px']:.2f}\n"
            f"n={row['pair_residual_denominator']}"
        ),
        color_map="Purples",
        colorbar_label="median absolute residual [px]",
    )
    part_a_d.suptitle(
        "Figure S2a–d. Measurement-window and capture-contract sensitivity",
        fontsize=10.0,
        fontweight="bold",
    )
    part_a_d.text(
        0.5,
        0.025,
        CODEBOOK,
        ha="center",
        va="bottom",
        fontsize=OTHER_FONT_SIZE_PT,
    )

    part_e_f, continued_axes = plt.subplots(2, 1, figsize=PART_E_F_SIZE_INCHES)
    part_e_f.subplots_adjust(
        left=0.14,
        right=0.90,
        top=0.84,
        bottom=0.17,
        hspace=0.75,
    )
    _render_gate_margins(
        part_e_f,
        continued_axes[0],
        gate_margins,
    )
    _render_gate_audit(continued_axes[1], audit)
    part_e_f.suptitle(
        "Figure S2 (continued). Panels e–f",
        fontsize=10.0,
        fontweight="bold",
    )
    part_e_f.text(
        0.5,
        0.035,
        CODEBOOK,
        ha="center",
        va="bottom",
        fontsize=OTHER_FONT_SIZE_PT,
    )

    part_a_d.canvas.draw()
    part_e_f.canvas.draw()
    heatmap_axes = [*axes, continued_axes[0]]
    heatmap_shapes = [(5, 15)] * 4 + [(5, 8)]
    cell_dimensions = []
    for axis, (row_count, column_count) in zip(
        heatmap_axes, heatmap_shapes, strict=True
    ):
        bounds = axis.get_position()
        figure_width, figure_height = axis.figure.get_size_inches()
        cell_dimensions.append(
            (
                bounds.width * figure_width * 25.4 / column_count,
                bounds.height * figure_height * 25.4 / row_count,
            )
        )
    minimum_cell_width = min(width for width, _ in cell_dimensions)
    minimum_cell_height = min(height for _, height in cell_dimensions)
    if (
        minimum_cell_width < MINIMUM_HEATMAP_CELL_SIZE_MM
        or minimum_cell_height < MINIMUM_HEATMAP_CELL_SIZE_MM
        or MINIMUM_EFFECTIVE_FONT_SIZE_PT < 6.0
    ):
        raise ValueError("Figure S2 rendering violates the readability contract")

    for figure, basename, title in (
        (
            part_a_d,
            PART_A_D_BASENAME,
            "Figure S2a–d: measurement-window and capture-contract sensitivity",
        ),
        (
            part_e_f,
            PART_E_F_BASENAME,
            "Figure S2e–f (continued): gate margins and width-trigger audit",
        ),
    ):
        figure.savefig(
            output_directory / f"{basename}.png",
            dpi=PNG_DPI,
            facecolor="white",
            metadata={"Software": "TrackLEED Figure S2 builder"},
        )
        figure.savefig(
            output_directory / f"{basename}.pdf",
            facecolor="white",
            metadata={
                "Title": title,
                "Author": "TrackLEED collaboration",
                "Subject": "Publication Figure S2",
                "Creator": "TrackLEED Figure S2 builder",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
    width = summary["conditional_width_sensitivity"]
    return FigureS2Report(
        variant_count=len(VARIANT_ORDER),
        roi_matrix_variant_count=len(MATRIX_VARIANTS),
        ofat_variant_count=len(OFAT_VARIANTS),
        method_count=len(METHOD_ORDER),
        gate_audit_record_count=len(audit),
        gate_margin_summary_record_count=len(gate_margins),
        width_sensitivity_triggered=width["decision"] == "run_all_width_contracts",
        part_count=2,
        display_item_count=1,
        minimum_effective_font_size_pt=MINIMUM_EFFECTIVE_FONT_SIZE_PT,
        minimum_heatmap_cell_width_mm=minimum_cell_width,
        minimum_heatmap_cell_height_mm=minimum_cell_height,
    )


def build_figure_s2_artifact(
    workspace_root: Path,
    source_directory: Path,
    output_directory: Path,
    manifest_path: Path,
) -> FigureS2Report:
    """Render Figure S2 and write its source/code/output manifest."""

    workspace_root = workspace_root.resolve()
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    report = render_figure_s2(source_directory, output_directory)
    manifest = {
        "schema_version": 2,
        "figure": "Figure S2",
        "title": "Measurement-window and capture-contract sensitivity",
        "status": "completed",
        "display_item_count": report.display_item_count,
        "continued_figure": True,
        "placement_width_inches": PLACEMENT_WIDTH_INCHES,
        "codebook_repeated_on_parts": True,
        "parts": [
            {
                "label": "Figure S2a–d",
                "panels": ["a", "b", "c", "d"],
                "source_size_inches": list(PART_A_D_SIZE_INCHES),
                "png": f"{PART_A_D_BASENAME}.png",
                "pdf": f"{PART_A_D_BASENAME}.pdf",
            },
            {
                "label": "Figure S2e–f (continued)",
                "panels": ["e", "f"],
                "source_size_inches": list(PART_E_F_SIZE_INCHES),
                "png": f"{PART_E_F_BASENAME}.png",
                "pdf": f"{PART_E_F_BASENAME}.pdf",
            },
        ],
        "panel_order": list(PANEL_ORDER),
        "variant_count": report.variant_count,
        "roi_matrix_variant_count": report.roi_matrix_variant_count,
        "ofat_variant_count": report.ofat_variant_count,
        "method_count": report.method_count,
        "gate_audit_record_count": report.gate_audit_record_count,
        "gate_margin_summary_record_count": (report.gate_margin_summary_record_count),
        "width_sensitivity_triggered": report.width_sensitivity_triggered,
        "fixed_denominators": [221, 24],
        "conditional_denominators": [
            "accepted fits",
            "jointly consistent pair frames",
        ],
        "source_data_sha256": {
            path.name: sha256_file(path)
            for path in sorted(source_directory.iterdir())
            if path.is_file()
        },
        "implementation_sha256": {
            path.as_posix(): sha256_file(workspace_root / path)
            for path in IMPLEMENTATION_PATHS
        },
        "outputs_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_directory.iterdir())
            if path.is_file()
        },
        "rendering": {
            "png_dpi": PNG_DPI,
            "font_family": "DejaVu Sans",
            "central_variant": {
                "id": "roi_48_proposal_15",
                "label": "R48-P15*",
                "outline": True,
            },
            "minimum_source_font_size_pt": MINIMUM_SOURCE_FONT_SIZE_PT,
            "fixed_denominators_in_cells": True,
            "minimum_effective_font_size_pt": round(
                report.minimum_effective_font_size_pt, 3
            ),
            "minimum_heatmap_cell_width_mm": round(
                report.minimum_heatmap_cell_width_mm, 3
            ),
            "minimum_heatmap_cell_height_mm": round(
                report.minimum_heatmap_cell_height_mm, 3
            ),
        },
        "automatic_configuration_selection": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


__all__ = ["FigureS2Report", "build_figure_s2_artifact", "render_figure_s2"]
