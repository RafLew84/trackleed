"""Build and validate the compact cross-contract main-figure package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trackleed.publication.cross_contract_claim_boundary import (
    CrossContractClaimBoundaryError,
    validate_cross_contract_claim_boundary,
)
from trackleed.publication.cross_contract_outcomes import (
    CrossContractValidationError,
    validate_cross_contract_outcomes,
)

DEFAULT_PACKAGE_PATH = Path("manuscript/cross_contract/figures")
EXPECTED_CONTRACT_SHA256 = (
    "06216eda68f1af563aca4b5ed48d07ced75896e7a1828e5e89a05fb185aa2409"
)
EXPECTED_MANIFEST_SHA256 = (
    "fd4972886cc2892d77ddea2f629459bc6255d3fc35ed98d54d035cf066fd563c"
)
STATIC_PATHS = ("README.md", "captions.json", "figure_contract.json")
GENERATED_PATHS = ("output", "source_data", "manifest.json", "validation_report.json")
OUTPUT_ROLES = {
    "README.md": "figure scope, build instructions, and publication constraints",
    "captions.json": "denominator-complete manuscript captions",
    "figure_contract.json": "frozen figure sizes, palette, panels, and source bindings",
    "output/figure01_cross_contract_framework.png": "300 dpi full-width conceptual figure",
    "output/figure02_measurability_and_gaps.png": "300 dpi single-column observability and gap figure",
    "output/figure03_out_of_branch_interpretation.png": "300 dpi single-column continuation-classification figure",
    "output/pdf/figure01_cross_contract_framework.pdf": "vector full-width conceptual figure",
    "output/pdf/figure02_measurability_and_gaps.pdf": "vector single-column observability and gap figure",
    "output/pdf/figure03_out_of_branch_interpretation.pdf": "vector single-column continuation-classification figure",
    "source_data/figure01_concept_flow.json": "machine-readable conceptual flow",
    "source_data/figure02_gap_components.csv": "machine-readable gap-component counts",
    "source_data/figure02_measurability.csv": "machine-readable four-state observability counts",
    "source_data/figure03_out_of_branch.csv": "machine-readable conditional continuation classifications",
}


class CrossContractFigureError(RuntimeError):
    """Raised when the cross-contract figure package fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CrossContractFigureError(f"figure source is unavailable: {path}") from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossContractFigureError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CrossContractFigureError(f"invalid {label}: {path}")
    return value


def _load_contract(workspace_root: Path, package: Path) -> dict[str, Any]:
    contract_path = package / "figure_contract.json"
    if not contract_path.is_file():
        raise CrossContractFigureError("cross-contract figure contract is unavailable")
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise CrossContractFigureError("cross-contract figure contract changed")
    contract = _load_json(contract_path, "cross-contract figure contract")
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("status") != "frozen-before-figure-rendering"
        or set(contract.get("figures", {})) != {"figure01", "figure02", "figure03"}
    ):
        raise CrossContractFigureError("cross-contract figure contract changed")
    bindings = contract.get("source_bindings")
    if not isinstance(bindings, dict) or len(bindings) != 7:
        raise CrossContractFigureError("cross-contract figure bindings changed")
    for binding_id, binding in bindings.items():
        if not isinstance(binding, dict):
            raise CrossContractFigureError(f"invalid figure binding: {binding_id}")
        relative_path = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise CrossContractFigureError(f"invalid figure binding: {binding_id}")
        if _sha256(workspace_root / relative_path) != expected_sha256:
            raise CrossContractFigureError(f"figure source drift: {binding_id}")
    try:
        validate_cross_contract_claim_boundary(workspace_root)
        validate_cross_contract_outcomes(workspace_root)
    except (CrossContractClaimBoundaryError, CrossContractValidationError) as exc:
        raise CrossContractFigureError(
            f"upstream cross-contract package failed validation: {exc}"
        ) from exc
    return contract


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


METHODS = (
    ("easyleed", "EasyLEED"),
    ("locotrack", "LocoTrack"),
    ("ostrack", "OSTrack"),
    ("dam4sam", "DAM4SAM"),
    ("lk", "Lucas-Kanade"),
)


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(
    figure: Any,
    package: Path,
    basename: str,
    dpi: int,
) -> tuple[Path, Path, list[int]]:
    from PIL import Image

    output = package / "output"
    pdf_output = output / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.mkdir(parents=True, exist_ok=True)
    png_path = output / f"{basename}.png"
    pdf_path = pdf_output / f"{basename}.pdf"
    figure.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "TrackLEED deterministic figure builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Creator": "TrackLEED deterministic figure builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    with Image.open(png_path) as image:
        dimensions = list(image.size)
    return png_path, pdf_path, dimensions


def _render_figure01(package: Path, contract: dict[str, Any]) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    from PIL import Image

    figure_spec = contract["figures"]["figure01"]
    palette = contract["palette"]
    dpi = int(contract["rendering"]["png_dpi"])
    width_mm = float(figure_spec["width_mm"])
    height_mm = float(figure_spec["height_mm"])
    figsize = (width_mm / 25.4, height_mm / 25.4)
    output = package / "output"
    pdf_output = output / "pdf"
    source_data = package / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    pdf_output.mkdir(parents=True, exist_ok=True)
    source_data.mkdir(parents=True, exist_ok=True)

    concept_source = {
        "concept_flow": contract["concept_flow"],
        "palette": palette,
        "schema_version": "1.0",
    }
    (source_data / "figure01_concept_flow.json").write_text(
        json.dumps(concept_source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    plt.rcParams.update(
        {
            "font.family": contract["rendering"]["font_family"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    figure.subplots_adjust(left=0.015, right=0.99, top=0.95, bottom=0.07)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        *,
        edge: str,
        face: str = "#FFFFFF",
        size: float = 7.0,
        weight: str = "normal",
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=face,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            linespacing=1.15,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=1.0,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    axis.text(0.01, 0.95, "a", fontsize=9.0, fontweight="bold", va="top")
    axis.text(
        0.045,
        0.95,
        "Two operational paths, one cross-contract interpretation layer",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    box(
        0.02,
        0.41,
        0.13,
        0.22,
        "Predeclared\nbranch-frame\nopportunity",
        edge="#4D4D4D",
        face="#F4F4F4",
        weight="bold",
    )

    gaussian = str(palette["gaussian"])
    native = str(palette["viperleed"])
    shared = str(palette["both_or_shared"])
    neutral = str(palette["neither_or_unresolved"])
    arrow(0.15, 0.57, 0.19, 0.70, gaussian)
    arrow(0.15, 0.47, 0.19, 0.28, native)
    box(
        0.19,
        0.62,
        0.27,
        0.20,
        "Local Gaussian path\nproposal -> fixed ROI -> 2D fit\n"
        "Gaussian-contract measurability",
        edge=gaussian,
        face="#FFF5DD",
        weight="bold",
    )
    box(
        0.19,
        0.18,
        0.27,
        0.24,
        "Native ViPErLEED path\nindexed whole-pattern\nregistration\n"
        "ViPEr-native direct\ndetectability",
        edge=native,
        face="#E8F7F1",
        weight="bold",
    )
    arrow(0.46, 0.72, 0.51, 0.62, gaussian)
    arrow(0.46, 0.30, 0.51, 0.42, native)

    axis.text(
        0.59,
        0.86,
        "Four-state observability",
        fontsize=7.6,
        fontweight="bold",
        ha="center",
    )
    outcomes = [
        (0.51, 0.61, "Both\nobserve", shared, "#E6F2F8"),
        (0.64, 0.61, "Gaussian\nonly", gaussian, "#FFF5DD"),
        (0.51, 0.40, "ViPErLEED\nonly", native, "#E8F7F1"),
        (0.64, 0.40, "Neither\nobserves", neutral, "#F0F0F0"),
    ]
    for x, y, label, edge, face in outcomes:
        box(x, y, 0.115, 0.15, label, edge=edge, face=face, weight="bold")

    arrow(0.755, 0.57, 0.79, 0.57, shared)
    box(
        0.79,
        0.66,
        0.19,
        0.15,
        "Bounded gaps\nand component exits",
        edge=shared,
        face="#E6F2F8",
        weight="bold",
    )
    box(
        0.79,
        0.32,
        0.19,
        0.30,
        "Continuation\ninterpretation\nBranch-model\ndisagreement\n"
        "Cross-contract target\ninconsistency\n"
        "Unresolved cross-contract\noutcome",
        edge=str(palette["cross_contract_target_inconsistency"]),
        face="#FFF0EA",
        weight="bold",
    )
    box(
        0.79,
        0.14,
        0.19,
        0.16,
        "Explicit denominators\nfixed opportunities /\nconditional events",
        edge="#4D4D4D",
        face="#F4F4F4",
        weight="bold",
    )
    axis.plot([0.885, 0.885], [0.30, 0.32], color="#8A8A8A", linewidth=0.8)
    axis.plot([0.885, 0.885], [0.62, 0.66], color="#8A8A8A", linewidth=0.8)

    axis.add_patch(
        Rectangle(
            (0.19, 0.05),
            0.79,
            0.075,
            facecolor="#F4F4F4",
            edgecolor="none",
        )
    )
    axis.text(
        0.585,
        0.087,
        "Operational outcomes, not ground truth  |  opportunities nested within scan, branch, and event",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#333333",
    )

    basename = str(figure_spec["basename"])
    png_path = output / f"{basename}.png"
    pdf_path = pdf_output / f"{basename}.pdf"
    figure.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "TrackLEED deterministic figure builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Creator": "TrackLEED deterministic figure builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    with Image.open(png_path) as image:
        dimensions = list(image.size)
    return {
        "png": png_path,
        "pdf": pdf_path,
        "png_dimensions_px": dimensions,
        "size_mm": [width_mm, height_mm],
        "minimum_font_pt": float(contract["rendering"]["minimum_final_size_font_pt"]),
    }


def _render_figure02(
    workspace_root: Path,
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_spec = contract["figures"]["figure02"]
    palette = contract["palette"]
    dpi = int(contract["rendering"]["png_dpi"])
    width_mm = float(figure_spec["width_mm"])
    height_mm = float(figure_spec["height_mm"])
    headline = _load_json(
        workspace_root / "manuscript/cross_contract/outcomes/headline_summary.json",
        "cross-contract outcome headline",
    )
    outcomes = headline["measurability_outcomes_by_method"]
    gap_comparisons = headline["gap_comparison_by_method"]
    denominator = int(headline["opportunity_count_per_method"])

    outcome_specs = (
        ("both_contracts_observe", "Both", str(palette["both_or_shared"])),
        ("gaussian_only", "Gaussian\nonly", str(palette["gaussian"])),
        ("viperleed_only", "ViPErLEED\nonly", str(palette["viperleed"])),
        (
            "neither_contract_observes",
            "Neither",
            str(palette["neither_or_unresolved"]),
        ),
    )
    measurability_rows: list[dict[str, object]] = []
    for method, display_name in METHODS:
        for outcome, _, _ in outcome_specs:
            count = int(outcomes[method][outcome])
            measurability_rows.append(
                {
                    "method": method,
                    "method_display_name": display_name,
                    "outcome": outcome,
                    "count": count,
                    "denominator": denominator,
                    "percent": f"{100.0 * count / denominator:.3f}",
                }
            )
    _write_csv(
        package / "source_data/figure02_measurability.csv",
        [
            "method",
            "method_display_name",
            "outcome",
            "count",
            "denominator",
            "percent",
        ],
        measurability_rows,
    )

    gap_specs = (
        ("gaussian_only", "Gaussian only", str(palette["gaussian"])),
        ("shared", "Shared", str(palette["both_or_shared"])),
        ("viperleed_only", "ViPErLEED only", str(palette["viperleed"])),
    )
    gap_rows: list[dict[str, object]] = []
    for method, display_name in METHODS:
        for category, _, _ in gap_specs:
            gap_rows.append(
                {
                    "method": method,
                    "method_display_name": display_name,
                    "category": category,
                    "component_count": int(gap_comparisons[method][category]),
                    "method_component_count": int(
                        gap_comparisons[method]["component_count"]
                    ),
                }
            )
    _write_csv(
        package / "source_data/figure02_gap_components.csv",
        [
            "method",
            "method_display_name",
            "category",
            "component_count",
            "method_component_count",
        ],
        gap_rows,
    )

    plt.rcParams.update(
        {
            "font.family": contract["rendering"]["font_family"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    matrix_axis = figure.add_axes((0.28, 0.58, 0.69, 0.29))
    gap_axis = figure.add_axes((0.28, 0.16, 0.69, 0.25))

    figure.text(0.03, 0.965, "a", fontsize=9.0, fontweight="bold", va="top")
    figure.text(
        0.105,
        0.962,
        "Four-state observability (n = 324 per method)",
        fontsize=7.4,
        fontweight="bold",
        va="top",
    )
    for row_index, (method, _) in enumerate(METHODS):
        for column_index, (outcome, _, color) in enumerate(outcome_specs):
            count = int(outcomes[method][outcome])
            fraction = count / denominator
            matrix_axis.scatter(
                column_index,
                row_index,
                s=38.0 + 245.0 * fraction,
                color=color,
                alpha=0.22 + 0.73 * fraction,
                edgecolors=color,
                linewidths=0.8,
                zorder=2,
            )
            matrix_axis.text(
                column_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#1F1F1F",
                zorder=3,
            )
    matrix_axis.set_xlim(-0.5, 3.5)
    matrix_axis.set_ylim(4.5, -0.5)
    matrix_axis.set_xticks(range(4), [spec[1] for spec in outcome_specs])
    matrix_axis.xaxis.tick_top()
    matrix_axis.set_yticks(range(5), [display for _, display in METHODS])
    matrix_axis.tick_params(axis="x", length=0, pad=3, labelsize=7.0)
    matrix_axis.tick_params(axis="y", length=0, pad=4, labelsize=7.0)
    for position in np.arange(-0.5, 5.0, 1.0):
        matrix_axis.axhline(float(position), color="#E5E5E5", linewidth=0.55, zorder=0)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    figure.text(0.03, 0.49, "b", fontsize=9.0, fontweight="bold", va="top")
    figure.text(
        0.105,
        0.487,
        "Bounded-gap comparison components",
        fontsize=7.4,
        fontweight="bold",
        va="top",
    )
    y_positions = np.arange(len(METHODS))
    left = np.zeros(len(METHODS))
    for category, label, color in gap_specs:
        values = np.array(
            [int(gap_comparisons[method][category]) for method, _ in METHODS]
        )
        gap_axis.barh(
            y_positions,
            values,
            left=left,
            height=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=label,
        )
        for index, value in enumerate(values):
            if value >= 4:
                gap_axis.text(
                    left[index] + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="#111111",
                    fontweight="bold",
                )
            elif value:
                gap_axis.text(
                    left[index] + value / 2,
                    index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="#FFFFFF",
                    fontweight="normal",
                )
        left += values
    gap_axis.set_yticks(y_positions, [display for _, display in METHODS])
    gap_axis.invert_yaxis()
    gap_axis.set_xlim(0, 28)
    gap_axis.set_xlabel("Components within method / scan / branch", labelpad=3)
    gap_axis.set_xticks([0, 10, 20])
    gap_axis.tick_params(axis="both", labelsize=7.0, length=2)
    gap_axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    gap_axis.set_axisbelow(True)
    gap_axis.spines["top"].set_visible(False)
    gap_axis.spines["right"].set_visible(False)
    gap_axis.spines["left"].set_visible(False)
    gap_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.2,
        columnspacing=1.0,
    )
    figure.text(
        0.03,
        0.025,
        "Nested descriptive events; not independent experimental replicates.",
        fontsize=7.0,
        color="#444444",
    )

    basename = str(figure_spec["basename"])
    png_path, pdf_path, dimensions = _save_figure(figure, package, basename, dpi)
    plt.close(figure)
    return {
        "png": png_path,
        "pdf": pdf_path,
        "png_dimensions_px": dimensions,
        "size_mm": [width_mm, height_mm],
        "minimum_font_pt": float(contract["rendering"]["minimum_final_size_font_pt"]),
    }


def _render_figure03(
    workspace_root: Path,
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_spec = contract["figures"]["figure03"]
    palette = contract["palette"]
    dpi = int(contract["rendering"]["png_dpi"])
    width_mm = float(figure_spec["width_mm"])
    height_mm = float(figure_spec["height_mm"])
    headline = _load_json(
        workspace_root / "manuscript/cross_contract/outcomes/headline_summary.json",
        "cross-contract outcome headline",
    )
    denominators = headline["method_denominators"]
    classification_specs = (
        (
            "branch_model_disagreement",
            "Branch-model disagreement",
            "out_of_branch_branch_model_disagreement",
            str(palette["branch_model_disagreement"]),
        ),
        (
            "cross_contract_target_inconsistency",
            "Cross-contract target inconsistency",
            "out_of_branch_cross_contract_target_inconsistency",
            str(palette["cross_contract_target_inconsistency"]),
        ),
        (
            "unresolved",
            "Unresolved cross-contract outcome",
            "out_of_branch_unresolved",
            str(palette["neither_or_unresolved"]),
        ),
    )
    rows: list[dict[str, object]] = []
    for method, display_name in METHODS:
        method_values = denominators[method]
        conditional_denominator = int(method_values["out_of_branch_all_frames"])
        accepted = int(method_values["accepted_gaussian_all_frames"])
        for classification, _, source_key, _ in classification_specs:
            event_count = int(method_values[source_key])
            rows.append(
                {
                    "method": method,
                    "method_display_name": display_name,
                    "classification": classification,
                    "event_count": event_count,
                    "conditional_denominator": conditional_denominator,
                    "percent_of_accepted_out_of_branch": (
                        f"{100.0 * event_count / conditional_denominator:.3f}"
                    ),
                    "accepted_gaussian_all_frames": accepted,
                }
            )
    _write_csv(
        package / "source_data/figure03_out_of_branch.csv",
        [
            "method",
            "method_display_name",
            "classification",
            "event_count",
            "conditional_denominator",
            "percent_of_accepted_out_of_branch",
            "accepted_gaussian_all_frames",
        ],
        rows,
    )

    plt.rcParams.update(
        {
            "font.family": contract["rendering"]["font_family"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    axis = figure.add_axes((0.29, 0.46, 0.52, 0.34))
    figure.text(0.03, 0.965, "a", fontsize=9.0, fontweight="bold", va="top")
    figure.text(
        0.105,
        0.962,
        "Accepted out-of-branch continuation",
        fontsize=7.6,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.105,
        0.902,
        "Classification conditional on each method's\naccepted out-of-branch events",
        fontsize=7.0,
        va="top",
        color="#444444",
    )

    y_positions = np.arange(len(METHODS))
    left = np.zeros(len(METHODS), dtype=float)
    for _, label, source_key, color in classification_specs:
        counts = np.array(
            [int(denominators[method][source_key]) for method, _ in METHODS]
        )
        conditional = np.array(
            [
                int(denominators[method]["out_of_branch_all_frames"])
                for method, _ in METHODS
            ]
        )
        percentages = 100.0 * counts / conditional
        axis.barh(
            y_positions,
            percentages,
            left=left,
            height=0.58,
            color=color,
            label=label,
        )
        for index, (count, width) in enumerate(zip(counts, percentages, strict=True)):
            if count:
                axis.text(
                    left[index] + width / 2,
                    index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="#111111",
                    fontweight="bold",
                )
        left += percentages

    axis.set_yticks(y_positions, [display for _, display in METHODS])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 50, 100], ["0", "50", "100%"])
    axis.set_xlabel("Share within accepted out-of-branch events", labelpad=3)
    axis.tick_params(axis="both", labelsize=7.0, length=2)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    for index, (method, _) in enumerate(METHODS):
        method_conditional = int(denominators[method]["out_of_branch_all_frames"])
        accepted = int(denominators[method]["accepted_gaussian_all_frames"])
        axis.text(
            103,
            index,
            f"{method_conditional} / {accepted}",
            ha="left",
            va="center",
            fontsize=7.0,
            clip_on=False,
        )
    figure.text(
        0.825,
        0.825,
        "OOB / fits",
        fontsize=7.0,
        ha="left",
        color="#444444",
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.355),
        frameon=False,
        fontsize=7.0,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.55,
    )
    figure.text(
        0.06,
        0.025,
        "Counts inside bars; right labels give out-of-branch / accepted\n"
        "Gaussian fits. Operational classes are not ground-truth\n"
        "identity assignments.",
        fontsize=7.0,
        color="#444444",
        linespacing=1.35,
    )

    basename = str(figure_spec["basename"])
    png_path, pdf_path, dimensions = _save_figure(figure, package, basename, dpi)
    plt.close(figure)
    return {
        "png": png_path,
        "pdf": pdf_path,
        "png_dimensions_px": dimensions,
        "size_mm": [width_mm, height_mm],
        "minimum_font_pt": float(contract["rendering"]["minimum_final_size_font_pt"]),
    }


def _render_package(
    workspace_root: Path,
    package: Path,
    contract: dict[str, Any],
) -> None:
    _render_figure01(package, contract)
    _render_figure02(workspace_root, package, contract)
    _render_figure03(workspace_root, package, contract)


def _build_manifest(package: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for relative, role in sorted(OUTPUT_ROLES.items()):
        path = package / relative
        if not path.is_file():
            raise CrossContractFigureError(
                f"cross-contract figure manifest input is unavailable: {relative}"
            )
        files.append(
            {
                "path": relative,
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "figure_package_id": "cross-contract-main-figures",
        "file_count": len(files),
        "files": files,
        "schema_version": "1.0",
        "status": "frozen-cross-contract-figure-manifest",
    }


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CrossContractFigureError(f"invalid {label}: {path}") from exc


def _inspect_figure_outputs(
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    from PIL import Image
    from pypdf import PdfReader

    dpi = int(contract["rendering"]["png_dpi"])
    inspected: dict[str, object] = {}
    for figure_id, specification in contract["figures"].items():
        basename = str(specification["basename"])
        width_mm = float(specification["width_mm"])
        height_mm = float(specification["height_mm"])
        png_path = package / "output" / f"{basename}.png"
        pdf_path = package / "output/pdf" / f"{basename}.pdf"
        try:
            with Image.open(png_path) as image:
                dimensions = list(image.size)
            reader = PdfReader(pdf_path)
        except (OSError, ValueError) as exc:
            raise CrossContractFigureError(
                f"cross-contract figure output is unreadable: {figure_id}"
            ) from exc
        expected_dimensions = [
            int(width_mm / 25.4 * dpi),
            int(height_mm / 25.4 * dpi),
        ]
        if dimensions != expected_dimensions or len(reader.pages) != 1:
            raise CrossContractFigureError(
                f"cross-contract figure publication size changed: {figure_id}"
            )
        page = reader.pages[0]
        pdf_size = [float(page.mediabox.width), float(page.mediabox.height)]
        expected_pdf_size = [width_mm / 25.4 * 72.0, height_mm / 25.4 * 72.0]
        if any(
            abs(actual - expected) > 0.1
            for actual, expected in zip(pdf_size, expected_pdf_size, strict=True)
        ):
            raise CrossContractFigureError(
                f"cross-contract figure PDF size changed: {figure_id}"
            )
        inspected[figure_id] = {
            "basename": basename,
            "layout": specification["layout"],
            "minimum_font_pt": float(
                contract["rendering"]["minimum_final_size_font_pt"]
            ),
            "pdf_page_count": 1,
            "pdf_size_pt": [round(value, 3) for value in pdf_size],
            "png_dimensions_px": dimensions,
            "size_mm": [width_mm, height_mm],
        }
    return inspected


def _validation_report(
    package: Path,
    contract: dict[str, Any],
) -> dict[str, object]:
    measurability = _read_csv(
        package / "source_data/figure02_measurability.csv",
        "figure 2 measurability source",
    )
    gap_components = _read_csv(
        package / "source_data/figure02_gap_components.csv",
        "figure 2 gap source",
    )
    out_of_branch = _read_csv(
        package / "source_data/figure03_out_of_branch.csv",
        "figure 3 continuation source",
    )
    gap_totals = {
        category: sum(
            int(row["component_count"])
            for row in gap_components
            if row["category"] == category
        )
        for category in ("gaussian_only", "shared", "viperleed_only")
    }
    classification_totals = {
        category: sum(
            int(row["event_count"])
            for row in out_of_branch
            if row["classification"] == category
        )
        for category in (
            "branch_model_disagreement",
            "cross_contract_target_inconsistency",
            "unresolved",
        )
    }
    if (
        len(measurability) != 20
        or len(gap_components) != 15
        or len(out_of_branch) != 15
        or gap_totals != {"gaussian_only": 111, "shared": 3, "viperleed_only": 7}
        or classification_totals
        != {
            "branch_model_disagreement": 83,
            "cross_contract_target_inconsistency": 16,
            "unresolved": 25,
        }
    ):
        raise CrossContractFigureError(
            "cross-contract figure source denominator or headline count changed"
        )
    captions = _load_json(package / "captions.json", "cross-contract captions")
    if set(captions) != {"figure01", "figure02", "figure03", "schema_version"}:
        raise CrossContractFigureError(
            "cross-contract figure caption inventory changed"
        )
    return {
        "caption_count": 3,
        "claim_boundary_decision_sha256": contract["source_bindings"][
            "claim_boundary_decision"
        ]["sha256"],
        "fail_closed": True,
        "figure_count": 3,
        "figure_package_id": "cross-contract-main-figures",
        "figures": _inspect_figure_outputs(package, contract),
        "gap_component_totals": gap_totals,
        "image_policy": contract["image_policy"],
        "manifest_sha256": _sha256(package / "manifest.json"),
        "out_of_branch_classification_totals": classification_totals,
        "outcomes_manifest_sha256": contract["source_bindings"]["outcomes_manifest"][
            "sha256"
        ],
        "schema_version": "1.0",
        "source_record_counts": {
            "figure02_gap_components": len(gap_components),
            "figure02_measurability": len(measurability),
            "figure03_out_of_branch": len(out_of_branch),
        },
        "status": "validated-cross-contract-main-figures",
    }


def _package_file(package: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossContractFigureError("manifested cross-contract figure path changed")
    path = (package / value).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise CrossContractFigureError(
            f"manifested cross-contract figure is unavailable: {value}"
        )
    return path


def _validate_frozen_package(
    workspace_root: Path,
    package: Path,
) -> dict[str, object]:
    contract = _load_contract(workspace_root, package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise CrossContractFigureError(
            f"cross-contract figure manifest is unavailable: {package}"
        )
    if EXPECTED_MANIFEST_SHA256 and _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CrossContractFigureError("cross-contract figure manifest changed")
    manifest = _load_json(manifest_path, "cross-contract figure manifest")
    manifested_files = manifest.get("files")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("figure_package_id") != "cross-contract-main-figures"
        or manifest.get("status") != "frozen-cross-contract-figure-manifest"
        or not isinstance(manifested_files, list)
        or manifest.get("file_count") != len(manifested_files)
    ):
        raise CrossContractFigureError(
            "cross-contract figure manifest identity changed"
        )
    manifested: set[str] = set()
    for item in manifested_files:
        if not isinstance(item, dict):
            raise CrossContractFigureError(
                "cross-contract figure manifest record changed"
            )
        relative = item.get("path")
        path = _package_file(package, relative)
        if not isinstance(relative, str) or relative in manifested:
            raise CrossContractFigureError(
                "duplicate cross-contract figure manifest path"
            )
        if (
            item.get("role") != OUTPUT_ROLES.get(relative)
            or item.get("sha256") != _sha256(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise CrossContractFigureError(
                f"manifested cross-contract figure drift: {relative}"
            )
        manifested.add(relative)
    if manifested != set(OUTPUT_ROLES):
        raise CrossContractFigureError(
            "cross-contract figure manifest coverage changed"
        )
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    }
    expected = set(OUTPUT_ROLES) | {"manifest.json", "validation_report.json"}
    if actual != expected:
        raise CrossContractFigureError(
            "cross-contract figure package file inventory changed"
        )

    with TemporaryDirectory(prefix="cross-contract-figure-validation-") as temporary:
        reproduction = Path(temporary) / "figures"
        reproduction.mkdir()
        for relative in STATIC_PATHS:
            shutil.copyfile(package / relative, reproduction / relative)
        _render_package(workspace_root, reproduction, contract)
        for relative in set(OUTPUT_ROLES) - set(STATIC_PATHS):
            if (package / relative).read_bytes() != (
                reproduction / relative
            ).read_bytes():
                raise CrossContractFigureError(
                    f"cross-contract figure no longer reproduces: {relative}"
                )
    if manifest_path.read_bytes() != _json_bytes(_build_manifest(package)):
        raise CrossContractFigureError(
            "cross-contract figure manifest no longer reproduces"
        )
    report = _validation_report(package, contract)
    report_path = package / "validation_report.json"
    if not report_path.is_file() or report_path.read_bytes() != _json_bytes(report):
        raise CrossContractFigureError(
            "cross-contract figure validation report changed"
        )
    return report


def build_cross_contract_figures(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Build once, validate, and publish the compact main-figure package."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    _load_contract(workspace_root, package)
    if any((package / relative).exists() for relative in GENERATED_PATHS):
        raise FileExistsError(
            f"Refusing to replace frozen cross-contract figure package: {package}"
        )
    for relative in STATIC_PATHS:
        if not (package / relative).is_file():
            raise CrossContractFigureError(
                f"cross-contract figure source is unavailable: {relative}"
            )
    with TemporaryDirectory(
        prefix="cross-contract-figures-",
        dir=package.parent,
    ) as temporary:
        stage = Path(temporary) / "figures"
        stage.mkdir()
        for relative in STATIC_PATHS:
            shutil.copyfile(package / relative, stage / relative)
        contract = _load_contract(workspace_root, stage)
        _render_package(workspace_root, stage, contract)
        (stage / "manifest.json").write_bytes(_json_bytes(_build_manifest(stage)))
        (stage / "validation_report.json").write_bytes(
            _json_bytes(_validation_report(stage, contract))
        )
        validated = _validate_frozen_package(workspace_root, stage)
        if any(
            _sha256(package / relative) != _sha256(stage / relative)
            for relative in STATIC_PATHS
        ):
            raise CrossContractFigureError(
                "cross-contract figure sources changed during staged validation"
            )
        for relative in GENERATED_PATHS:
            os.replace(stage / relative, package / relative)
    return validated


def validate_cross_contract_figures(
    workspace_root: Path,
    package_path: Path | None = None,
) -> dict[str, object]:
    """Validate the frozen three-figure main-text package."""

    workspace_root = workspace_root.resolve()
    package = (
        package_path.resolve()
        if package_path is not None
        else workspace_root / DEFAULT_PACKAGE_PATH
    )
    return _validate_frozen_package(workspace_root, package)


__all__ = [
    "CrossContractFigureError",
    "build_cross_contract_figures",
    "validate_cross_contract_figures",
]
