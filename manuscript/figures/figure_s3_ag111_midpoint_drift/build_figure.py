#!/usr/bin/env python3
"""Build and validate supplementary Figure S3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path

from freeze_source_data import validate_source_package


FIGURE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
FIGURE_BASENAME = "figure_s3_ag111_midpoint_drift"
FIGURE_SIZE_INCHES = (7.2, 8.4)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"

CASE_ORDER = (
    "scan_a_pair_1",
    "scan_a_pair_2",
    "scan_b_pair_1",
    "scan_b_pair_2",
)
STATUS_COLORS = {
    "visible_fit": "#DCEFD9",
    "iv_visibility_gap": "#F4BFC1",
    "beam_onset_or_termination": "#DDDDDD",
    "apparatus_absence": "#AFAFAF",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_case_contract() -> list[dict[str, object]]:
    return [
        {
            **row,
            "frame_count": int(row["frame_count"]),
            "valid_pair_frames": int(row["valid_pair_frames"]),
            "visibility_branch_frames": int(row["visibility_branch_frames"]),
        }
        for row in _read_csv(SOURCE_DATA_DIRECTORY / "case_contract.csv")
    ]


def load_pair_measurements() -> list[dict[str, object]]:
    numeric = (
        "energy_ev",
        "pair_distance_px",
        "predicted_distance_px",
        "distance_residual_px",
        "midpoint_x",
        "midpoint_y",
        "midpoint_dx_px",
        "midpoint_dy_px",
        "midpoint_drift_px",
    )
    return [
        {
            **row,
            "frame": int(row["frame"]),
            **{field: float(row[field]) for field in numeric},
        }
        for row in _read_csv(SOURCE_DATA_DIRECTORY / "pair_measurements.csv")
    ]


def load_visibility_regimes() -> list[dict[str, object]]:
    return [
        {
            **row,
            "frame": int(row["frame"]),
            "energy_ev": float(row["energy_ev"]),
            "fit_valid": row["fit_valid"] == "true",
        }
        for row in _read_csv(SOURCE_DATA_DIRECTORY / "visibility_regimes.csv")
    ]


def _draw_status_ribbons(axis: object, rows: list[dict[str, object]]) -> None:
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transData, axis.transAxes)
    for branch_index, branch in enumerate(("B1", "B2")):
        branch_rows = sorted(
            (row for row in rows if row["branch"] == branch),
            key=lambda row: row["frame"],
        )
        energies = [float(row["energy_ev"]) for row in branch_rows]
        for index, row in enumerate(branch_rows):
            left = (
                energies[index] - (energies[index + 1] - energies[index]) / 2
                if index == 0 and len(energies) > 1
                else (energies[index - 1] + energies[index]) / 2
                if index > 0
                else energies[index] - 0.5
            )
            right = (
                (energies[index] + energies[index + 1]) / 2
                if index + 1 < len(energies)
                else energies[index] + (energies[index] - energies[index - 1]) / 2
                if index > 0
                else energies[index] + 0.5
            )
            axis.add_patch(
                Rectangle(
                    (left, 0.015 + branch_index * 0.025),
                    max(right - left, 0.2),
                    0.02,
                    transform=transform,
                    facecolor=STATUS_COLORS[str(row["status"])],
                    edgecolor="none",
                    clip_on=False,
                    zorder=0,
                )
            )
        axis.text(
            0.005,
            0.025 + branch_index * 0.025,
            branch,
            transform=axis.transAxes,
            fontsize=5.5,
            fontweight="bold",
            va="center",
        )


def render(output_directory: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cases = {row["case_id"]: row for row in load_case_contract()}
    trajectories = load_pair_measurements()
    visibility = load_visibility_regimes()
    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES, facecolor="white")
    for panel_index, (axis, case_id) in enumerate(zip(axes.flat, CASE_ORDER)):
        case = cases[case_id]
        rows = [row for row in trajectories if row["case_id"] == case_id]
        status_rows = [row for row in visibility if row["case_id"] == case_id]
        energy = [row["energy_ev"] for row in rows]
        axis.axhline(0.0, color="#888888", linewidth=0.7, zorder=0)
        axis.scatter(
            energy,
            [row["distance_residual_px"] for row in rows],
            s=17,
            marker="o",
            facecolor="#4D4D4D",
            edgecolor="white",
            linewidth=0.35,
            label=r"$D_{measured}-D_{expected}$",
        )
        axis.scatter(
            energy,
            [row["midpoint_dx_px"] for row in rows],
            s=15,
            marker="^",
            facecolor="#0072B2",
            edgecolor="white",
            linewidth=0.35,
            label=r"midpoint $\Delta x$",
        )
        axis.scatter(
            energy,
            [row["midpoint_dy_px"] for row in rows],
            s=15,
            marker="s",
            facecolor="#E69F00",
            edgecolor="white",
            linewidth=0.35,
            label=r"midpoint $\Delta y$",
        )
        axis.scatter(
            energy,
            [row["midpoint_drift_px"] for row in rows],
            s=13,
            marker="x",
            color="#009E73",
            linewidth=0.75,
            label=r"midpoint $|\Delta m|$",
        )
        _draw_status_ribbons(axis, status_rows)
        axis.set_title(
            f"{case['series']}, {case['pair']}  "
            f"(valid pair frames {case['valid_pair_frames']}/{case['frame_count']})",
            loc="left",
        )
        axis.set_xlabel("Electron energy [eV]")
        axis.set_ylabel("Displacement or residual [px]")
        axis.grid(color="#E5E5E5", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.13,
            1.05,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        if panel_index == 0:
            axis.legend(loc="upper right", ncols=2, frameon=False)
    figure.suptitle(
        "Ag(111) pair-distance residual and pair-midpoint drift",
        fontsize=9.2,
        y=0.985,
    )
    figure.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=status.replace("_", " "))
            for status, color in STATUS_COLORS.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=4,
        frameon=False,
        fontsize=5.8,
    )
    figure.text(
        0.5,
        0.012,
        "Midpoint drift is an apparatus/projection diagnostic, not tracker error or absolute ground truth.",
        ha="center",
        fontsize=6.3,
        fontweight="bold",
        color="#4D4D4D",
    )
    figure.tight_layout(rect=(0.02, 0.035, 1.0, 0.925), h_pad=2.0, w_pad=1.5)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png = output_directory / f"{FIGURE_BASENAME}.png"
    pdf = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure S3 builder"},
    )
    figure.savefig(
        pdf,
        facecolor="white",
        metadata={
            "Title": "Ag(111) pair-midpoint drift diagnostics",
            "Author": "TrackLEED collaboration",
            "Subject": "Supplementary Figure S3",
            "Creator": "TrackLEED Figure S3 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png, "pdf": pdf}


def _source_hashes() -> dict[str, str]:
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(SOURCE_DATA_DIRECTORY.iterdir())
        if path.is_file()
    }


def _code_hashes() -> dict[str, str]:
    return {
        name: sha256_file(FIGURE_ROOT / name)
        for name in ("build_figure.py", "freeze_source_data.py", "test_build_figure.py")
    }


def build(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Path]:
    validate_source_package(check_upstream=False)
    outputs = render(output_directory)
    import matplotlib
    import numpy
    from PIL import __version__ as pillow_version

    payload = {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure S3",
        "title": "Pair-midpoint drift and apparatus/projection diagnostics",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "code_sha256": _code_hashes(),
        "source_data_sha256": _source_hashes(),
        "outputs": {
            name: {
                "path": path.relative_to(manifest_path.parent).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "case_count": len(load_case_contract()),
        "valid_pair_frame_count": len(load_pair_measurements()),
        "visibility_branch_frame_count": len(load_visibility_regimes()),
        "claims_absolute_position_accuracy": False,
        "rendering": {
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "png_dpi": PNG_DPI,
            "missing_pair_policy": "not interpolated or connected",
            "method_palette_used": False,
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return outputs


def validate() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise ValueError("Figure S3 contract hash mismatch")
    if manifest["code_sha256"] != _code_hashes():
        raise ValueError("Figure S3 code hash mismatch")
    if manifest["source_data_sha256"] != _source_hashes():
        raise ValueError("Figure S3 source-data hash mismatch")
    validate_source_package(check_upstream=False)
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s3-render-") as tmp:
        rebuilt = render(Path(tmp))
        for name, path in rebuilt.items():
            if sha256_file(path) != manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Non-deterministic Figure S3 render: {name}")
    print("Figure S3 validation passed: four cases with explicit midpoint diagnostics.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-source-data", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        validate()
        return
    if arguments.refresh_source_data:
        from freeze_source_data import build as freeze_sources

        freeze_sources()
    build()
    validate()


if __name__ == "__main__":
    main()
