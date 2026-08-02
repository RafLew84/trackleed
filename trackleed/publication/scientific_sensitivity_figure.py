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
FIGURE_SIZE_INCHES = (11.0, 9.5)
PNG_DPI = 300
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
        f"roi_{roi}_proposal_{proposal}": f"R{roi}\nP{proposal}"
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
            axis.text(x, y, label, ha="center", va="center", fontsize=4.4, color=color)


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
    axis.axvline(8.5, color="white", linewidth=1.5)
    central_index = VARIANT_ORDER.index("roi_48_proposal_15")
    axis.axvline(central_index - 0.48, color="#D55E00", linewidth=0.8)
    axis.axvline(central_index + 0.48, color="#D55E00", linewidth=0.8)
    axis.set_title(title, loc="left", fontsize=8.0)
    axis.text(
        -0.07,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
    )
    axis.text(
        0.29,
        -0.24,
        "ROI × proposal-to-fit matrix",
        transform=axis.transAxes,
        ha="center",
        fontsize=6.0,
    )
    axis.text(
        0.81,
        -0.24,
        "one-factor-at-a-time perturbations",
        transform=axis.transAxes,
        ha="center",
        fontsize=6.0,
    )
    axis.tick_params(labelsize=5.5, length=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.024, pad=0.015)
    colorbar.set_label(colorbar_label, fontsize=6.0)
    colorbar.ax.tick_params(labelsize=5.5)


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
                fontsize=4.2,
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
        fontsize=7.2,
    )
    axis.text(
        -0.08,
        1.05,
        "e",
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
    )
    axis.tick_params(labelsize=5.2, length=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("normalized signed P05 (clipped)", fontsize=5.5)
    colorbar.ax.tick_params(labelsize=5.0)


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
        fontsize=8.0,
    )
    axis.text(
        -0.035,
        1.05,
        "f",
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
    )
    axis.text(
        0.99,
        0.93,
        "validated decision: width sensitivity not run",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#555555",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=6.0)
    axis.legend(frameon=False, ncol=1, fontsize=5.2, loc="upper left")


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
    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.72))
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]
    _render_heatmap(
        figure,
        axes[0],
        rows,
        panel_label="a",
        title="Branch-consistent coverage (numerator / 221)",
        value_field="branch_consistent_measurements",
        annotation=lambda row: str(row["branch_consistent_measurements"]),
        color_map="Blues",
        colorbar_label="branch-consistent measurements",
    )
    _render_heatmap(
        figure,
        axes[1],
        rows,
        panel_label="b",
        title="Post-gap branch-consistent recovery (numerator / 24)",
        value_field="branch_consistent_recovered_gap_count",
        annotation=lambda row: str(row["branch_consistent_recovered_gap_count"]),
        color_map="Greens",
        colorbar_label="recovered gaps",
    )
    _render_heatmap(
        figure,
        axes[2],
        rows,
        panel_label="c",
        title="Geometry-inconsistent continuation (events / accepted fits)",
        value_field="geometry_inconsistent_fraction",
        annotation=lambda row: (
            f"{row['geometry_inconsistent_accepted_fits']}/"
            f"{row['geometry_inconsistency_denominator']}"
        ),
        color_map="Oranges",
        colorbar_label="fraction of accepted fits",
    )
    _render_heatmap(
        figure,
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
    bottom_grid = grid[2, :].subgridspec(1, 2, width_ratios=(1.35, 1.0))
    _render_gate_margins(
        figure,
        figure.add_subplot(bottom_grid[0, 0]),
        gate_margins,
    )
    _render_gate_audit(figure.add_subplot(bottom_grid[0, 1]), audit)
    figure.suptitle(
        "Measurement-window and capture-contract sensitivity",
        fontsize=10.0,
        fontweight="bold",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{FIGURE_BASENAME}.png"
    pdf_path = output_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure S2 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Measurement-window and capture-contract sensitivity",
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
        "schema_version": 1,
        "figure": "Figure S2",
        "title": "Measurement-window and capture-contract sensitivity",
        "status": "completed",
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
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "png_dpi": PNG_DPI,
            "font_family": "DejaVu Sans",
            "central_variant_outline": "roi_48_proposal_15",
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
