"""Supplementary Figure S1 for all controlled-translation source trials."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from trackleed.publication.controlled_translation_evidence import (  # noqa: E402
    METHOD_ORDER,
    TRIAL_LABELS,
    TRIAL_ORDER,
    as_bool,
    sha256_file,
    validate_controlled_translation_source_data,
)


IMPLEMENTATION_PATHS = (
    Path("trackleed/publication/controlled_translation_evidence.py"),
    Path("trackleed/publication/controlled_translation_figure_s1.py"),
)
DIAGNOSTIC_ROWS = (
    "full_measurement_paths",
    "native_vs_refined_displacement",
    "pair_invariance",
)
OUTPUT_STEM = "figure_s1_controlled_translation_trials"
COLORS = {
    "locotrack_base_offline": "#0072B2",
    "ostrack_vitb_256_mae_ce": "#D55E00",
    "dam4sam_sam21pp_l": "#009E73",
    "lucas_kanade_skip_anchor": "#CC79A7",
    "easyleed_kalman": "#E69F00",
    "oracle_roi_control": "#6B7280",
}


@dataclass(frozen=True)
class FigureS1Report:
    trial_count: int
    panel_count: int
    spot_record_count: int
    pair_record_count: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display_names(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["method"]: row["display_method"] for row in rows}


def _trial_title(trial: int, rows: list[dict[str, str]]) -> str:
    trial_rows = [row for row in rows if int(row["source_frame_order"]) == trial]
    temperature = int(float(trial_rows[0]["temperature_c"]))
    return f"{TRIAL_LABELS[trial]}\nsource frame {trial}; {temperature} deg C"


def _plot_measurement_paths(
    axis: Any,
    rows: list[dict[str, str]],
    trial: int,
) -> None:
    trial_rows = [row for row in rows if int(row["source_frame_order"]) == trial]
    commanded = [
        row
        for row in trial_rows
        if row["method"] == "oracle_roi_control" and row["spot_id"] == "1"
    ]
    commanded.sort(key=lambda row: int(row["translation_step"]))
    axis.plot(
        [float(row["commanded_dx_original_px"]) for row in commanded],
        [float(row["commanded_dy_original_px"]) for row in commanded],
        color="#111827",
        linewidth=2.0,
        linestyle=(0, (2, 2)),
        marker="o",
        markersize=3.2,
        label="commanded path",
        zorder=5,
    )
    for method in METHOD_ORDER:
        for spot_id in ("1", "3", "5", "6"):
            path = [
                row
                for row in trial_rows
                if row["method"] == method and row["spot_id"] == spot_id
            ]
            path.sort(key=lambda row: int(row["translation_step"]))
            if not path or not as_bool(path[0]["fit_valid"]):
                continue
            origin_x = float(path[0]["gaussian_center_x_original_px"])
            origin_y = float(path[0]["gaussian_center_y_original_px"])
            x_values = [
                float(row["gaussian_center_x_original_px"]) - origin_x
                if as_bool(row["fit_valid"])
                else float("nan")
                for row in path
            ]
            y_values = [
                float(row["gaussian_center_y_original_px"]) - origin_y
                if as_bool(row["fit_valid"])
                else float("nan")
                for row in path
            ]
            axis.plot(
                x_values,
                y_values,
                color=COLORS[method],
                alpha=0.48 if method != "oracle_roi_control" else 0.32,
                linewidth=0.9,
                marker=".",
                markersize=2.5,
                zorder=2,
            )
    axis.axhline(0.0, color="#D1D5DB", linewidth=0.6, zorder=0)
    axis.axvline(0.0, color="#D1D5DB", linewidth=0.6, zorder=0)
    axis.set_xlim(-17.5, 16.5)
    axis.set_ylim(-13.5, 15.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("relative x displacement [px]")


def _plot_native_vs_refined(
    axis: Any,
    rows: list[dict[str, str]],
    trial: int,
) -> None:
    for method in METHOD_ORDER[:-1]:
        points = [
            row
            for row in rows
            if int(row["source_frame_order"]) == trial
            and row["method"] == method
            and as_bool(row["is_evaluation_step"])
            and row["native_displacement_error_px"]
            and row["measurement_displacement_error_px"]
        ]
        axis.scatter(
            [float(row["native_displacement_error_px"]) for row in points],
            [float(row["measurement_displacement_error_px"]) for row in points],
            s=11,
            alpha=0.68,
            linewidths=0,
            color=COLORS[method],
        )
    limit = 80.0
    axis.plot([0, limit], [0, limit], color="#6B7280", linewidth=0.8, linestyle="--")
    axis.set_xscale("symlog", linthresh=0.01)
    axis.set_yscale("symlog", linthresh=0.01)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("native displacement error [px]")


def _plot_pair_invariance(
    axis: Any,
    rows: list[dict[str, str]],
    trial: int,
) -> None:
    invalid: list[tuple[int, str]] = []
    for method in METHOD_ORDER:
        for pair in ("1-6", "3-5"):
            points = [
                row
                for row in rows
                if int(row["source_frame_order"]) == trial
                and row["method"] == method
                and row["pair"] == pair
                and as_bool(row["is_evaluation_step"])
            ]
            points.sort(key=lambda row: int(row["translation_step"]))
            valid = [row for row in points if as_bool(row["pair_valid"])]
            axis.plot(
                [int(row["translation_step"]) for row in valid],
                [abs(float(row["delta_d_over_d_percent"])) for row in valid],
                color=COLORS[method],
                alpha=0.58 if method != "oracle_roi_control" else 0.4,
                linewidth=0.8,
                marker="o" if pair == "1-6" else "s",
                markersize=2.6,
            )
            invalid.extend(
                (int(row["translation_step"]), method)
                for row in points
                if not as_bool(row["pair_valid"])
            )
    for step, method in invalid:
        axis.scatter(step, 30.0, marker="x", s=16, color=COLORS[method])
    axis.set_yscale("symlog", linthresh=0.001)
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(0, 45)
    axis.set_xticks((1, 2, 3, 4, 5))
    axis.set_xlabel("translation step")


def _render_figure(
    spot_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(3, 4, figsize=(10.5, 8.1), constrained_layout=False)
    for column, trial in enumerate(TRIAL_ORDER):
        _plot_measurement_paths(axes[0, column], spot_rows, trial)
        _plot_native_vs_refined(axes[1, column], spot_rows, trial)
        _plot_pair_invariance(axes[2, column], pair_rows, trial)
        axes[0, column].set_title(_trial_title(trial, spot_rows), pad=6)
    axes[0, 0].set_ylabel("relative y displacement [px]")
    axes[1, 0].set_ylabel("Gaussian-refined error [px]")
    axes[2, 0].set_ylabel("absolute Delta D/D [pp]")
    row_labels = (
        "full Gaussian-refined paths",
        "native vs Gaussian-refined error",
        "pair-invariance diagnostic",
    )
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].annotate(
            label,
            xy=(-0.36, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.13,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
        )
        axis.grid(color="#E5E7EB", linewidth=0.45, alpha=0.7)

    names = _display_names(spot_rows)
    handles = [
        Line2D([0], [0], color="#111827", linestyle=(0, (2, 2)), label="Commanded path")
    ]
    handles.extend(
        Line2D([0], [0], color=COLORS[method], linewidth=2, label=names[method])
        for method in METHOD_ORDER
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.007),
        fontsize=6.6,
    )
    figure.suptitle(
        "Figure S1 | Complete controlled-translation trial paths and diagnostics",
        x=0.06,
        y=0.987,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.036,
        (
            "Known translations assess measurement-chain equivariance; they do not establish "
            "absolute tracking accuracy or identity ground truth. x marks denote invalid pairs."
        ),
        ha="center",
        fontsize=6.5,
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12, hspace=0.48, wspace=0.28)
    figure.savefig(png_path, dpi=300, metadata={"Software": "trackleed deterministic builder"})
    figure.savefig(
        pdf_path,
        dpi=300,
        metadata={
            "Title": "Figure S1 controlled-translation trials",
            "Creator": "trackleed deterministic builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def build_figure_s1_artifact(
    *,
    workspace_root: Path,
    source_directory: Path,
    output_directory: Path,
    manifest_path: Path,
) -> FigureS1Report:
    """Build Figure S1 and bind all evidence and implementation hashes."""

    workspace_root = workspace_root.resolve()
    source_directory = source_directory.resolve()
    report = validate_controlled_translation_source_data(source_directory)
    spot_rows = _read_csv(source_directory / "equivariance_results.csv")
    pair_rows = _read_csv(source_directory / "pair_invariance.csv")
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{OUTPUT_STEM}.png"
    pdf_path = output_directory / f"{OUTPUT_STEM}.pdf"
    _render_figure(spot_rows, pair_rows, png_path, pdf_path)

    panel_order = [
        f"{diagnostic}:{trial}"
        for diagnostic in DIAGNOSTIC_ROWS
        for trial in TRIAL_ORDER
    ]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "figure": "Figure S1",
        "title": "Complete controlled-translation trial paths and diagnostics",
        "trial_count": report["trial_count"],
        "panel_count": len(panel_order),
        "panel_order": panel_order,
        "diagnostic_rows": list(DIAGNOSTIC_ROWS),
        "spot_record_count": report["spot_record_count"],
        "pair_record_count": report["pair_record_count"],
        "figure_size_inches": [10.5, 8.1],
        "png_dpi": 300,
        "source_data_path": source_directory.relative_to(workspace_root).as_posix(),
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
            path.name: sha256_file(path) for path in (pdf_path, png_path)
        },
        "caption": (
            "Complete per-source-trial Gaussian-refined paths, native-versus-refined "
            "displacement diagnostics, and pair invariance. Known translations assess "
            "measurement-chain equivariance, not absolute tracking accuracy and not "
            "identity ground truth."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FigureS1Report(
        trial_count=int(report["trial_count"]),
        panel_count=len(panel_order),
        spot_record_count=int(report["spot_record_count"]),
        pair_record_count=int(report["pair_record_count"]),
    )


__all__ = ["FigureS1Report", "build_figure_s1_artifact"]
