#!/usr/bin/env python3
"""Build and validate supplementary Figure S4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import tempfile
from pathlib import Path

from freeze_source_data import validate_source_package


FIGURE_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
FIGURE_BASENAME = "figure_s4_easyleed_viperleed_compatibility"
FIGURE_SIZE_INCHES = (7.2, 7.2)
PNG_DPI = 400
COLORS = {"EasyLEED": "#D55E00", "ViPErLEED": "#4C78A8"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_common_measurements() -> list[dict[str, object]]:
    optional_float = (
        "center_x",
        "center_y",
        "proposal_offset_px",
        "r_squared",
        "amplitude",
        "volume",
        "sigma_x",
        "sigma_y",
    )
    rows = []
    for source in _read_csv(SOURCE_DATA_DIRECTORY / "common_measurements.csv"):
        rows.append(
            {
                **source,
                "segment_index": int(source["segment_index"]),
                "frame": int(source["frame"]),
                "energy_ev": float(source["energy_ev"]),
                "proposal_x": float(source["proposal_x"]),
                "proposal_y": float(source["proposal_y"]),
                "fit_valid": source["fit_valid"] == "True",
                **{
                    field: float(source[field]) if source[field] else None
                    for field in optional_float
                },
            }
        )
    return rows


def load_pair_measurements() -> list[dict[str, object]]:
    rows = []
    for source in _read_csv(SOURCE_DATA_DIRECTORY / "common_pair_measurements.csv"):
        rows.append(
            {
                **source,
                "frame": int(source["frame"]),
                "energy_ev": float(source["energy_ev"]),
                "pair_valid": source["pair_valid"] == "True",
                "pair_distance_px": float(source["pair_distance_px"])
                if source["pair_distance_px"]
                else None,
                "predicted_distance_px": float(source["predicted_distance_px"]),
                "distance_residual_px": float(source["distance_residual_px"])
                if source["distance_residual_px"]
                else None,
            }
        )
    return rows


def load_gap_recovery() -> list[dict[str, object]]:
    return [
        {
            **row,
            "gap_index": int(row["gap_index"]),
            "recovery_fit_valid": row["recovery_fit_valid"] == "True",
        }
        for row in _read_csv(SOURCE_DATA_DIRECTORY / "gap_recovery.csv")
    ]


def load_summary() -> dict[str, object]:
    return json.loads(
        (SOURCE_DATA_DIRECTORY / "compatibility_contract.json").read_text(
            encoding="utf-8"
        )
    )


def _matched_separations() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows = load_common_measurements()
    by_key: dict[tuple[int, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        by_key.setdefault((int(row["frame"]), str(row["branch"])), {})[
            str(row["method"])
        ] = row
    native: list[dict[str, float]] = []
    common: list[dict[str, float]] = []
    for (frame, branch), methods in sorted(by_key.items()):
        easy = methods["EasyLEED"]
        viper = methods["ViPErLEED"]
        energy = float(easy["energy_ev"])
        native.append(
            {
                "frame": float(frame),
                "energy_ev": energy,
                "distance_px": math.hypot(
                    float(easy["proposal_x"]) - float(viper["proposal_x"]),
                    float(easy["proposal_y"]) - float(viper["proposal_y"]),
                ),
            }
        )
        if easy["fit_valid"] and viper["fit_valid"]:
            common.append(
                {
                    "frame": float(frame),
                    "energy_ev": energy,
                    "distance_px": math.hypot(
                        float(easy["center_x"]) - float(viper["center_x"]),
                        float(easy["center_y"]) - float(viper["center_y"]),
                    ),
                }
            )
    return native, common


def render(output_directory: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    summary = load_summary()
    measurements = load_common_measurements()
    pair_rows = load_pair_measurements()
    native, common = _matched_separations()
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
    axis_a, axis_b, axis_c, axis_d = axes.flat

    axis_a.scatter(
        [row["energy_ev"] for row in native],
        [row["distance_px"] for row in native],
        s=16,
        color="#4D4D4D",
        label="native proposals (n=60)",
    )
    axis_a.scatter(
        [row["energy_ev"] for row in common],
        [row["distance_px"] for row in common],
        s=18,
        marker="x",
        color="#009E73",
        label="Gaussian centers (n=38)",
    )
    axis_a.set_yscale("symlog", linthresh=1e-6, linscale=0.4, base=10)
    axis_a.set_xlabel("Electron energy [eV]")
    axis_a.set_ylabel("EasyLEED–ViPErLEED separation [px]")
    axis_a.set_title("Native and common-center agreement", loc="left")
    axis_a.legend(frameon=False, loc="upper right")

    branches = ("P1_B1", "P1_B2", "P2_B1", "P2_B2")
    x = list(range(len(branches)))
    for offset, method in ((-0.17, "EasyLEED"), (0.17, "ViPErLEED")):
        values = [
            sum(
                row["fit_valid"]
                for row in measurements
                if row["method"] == method and row["branch"] == branch
            )
            for branch in branches
        ]
        axis_b.bar(
            [position + offset for position in x],
            values,
            width=0.32,
            color=COLORS[method],
            label=method,
        )
    shared_branch_counts = [
        sum(
            row["fit_valid"]
            for row in measurements
            if row["method"] == "EasyLEED" and row["branch"] == branch
        )
        for branch in branches
    ]
    for position, value in zip(x, shared_branch_counts):
        axis_b.text(
            position, value + 0.25, f"{value}/15 both", ha="center", fontsize=5.8
        )
    axis_b.set_xticks(x, branches)
    axis_b.set_ylim(0, 15.8)
    axis_b.set_ylabel("Accepted common fits [count]")
    axis_b.set_title("Branch-wise common measurement", loc="left")
    axis_b.legend(frameon=False)

    positions = []
    labels = []
    values = []
    colors = []
    annotations = []
    for pair_index, pair in enumerate(("P1", "P2")):
        for method_index, method in enumerate(("EasyLEED", "ViPErLEED")):
            valid = [
                row
                for row in pair_rows
                if row["pair"] == pair and row["method"] == method and row["pair_valid"]
            ]
            positions.append(pair_index * 3 + method_index)
            labels.append(method)
            values.append(
                statistics.median(
                    abs(float(row["distance_residual_px"])) for row in valid
                )
            )
            colors.append(COLORS[method])
            annotations.append(f"n={len(valid)}")
    axis_c.bar(positions, values, color=colors, width=0.75)
    for position, value, annotation in zip(positions, values, annotations):
        axis_c.text(position, value + 0.02, annotation, ha="center", fontsize=6.0)
    axis_c.set_xticks([0.5, 3.5], ["P1", "P2"])
    axis_c.set_ylabel("Median |pair-distance residual| [px]")
    axis_c.set_title("Conditional pair agreement", loc="left")
    axis_c.set_ylim(0, 0.9)

    axis_d.set_axis_off()
    axis_d.add_patch(
        FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.90,
            transform=axis_d.transAxes,
            boxstyle="round,pad=0.012",
            facecolor="#F4F8FC",
            edgecolor="#4C78A8",
            linestyle=(0, (4, 2)),
        )
    )
    agreement = summary["cross_method_contract_agreement_px"]
    axis_d.text(
        0.06,
        0.86,
        "Separate external-reference contract",
        transform=axis_d.transAxes,
        fontweight="bold",
    )
    axis_d.text(
        0.06,
        0.72,
        "EasyLEED: one Kalman tracker per spot\n"
        "ViPErLEED: registered 18-beam whole pattern",
        transform=axis_d.transAxes,
        linespacing=1.4,
    )
    axis_d.text(
        0.06,
        0.50,
        "60 branch opportunities per method\n38/60 accepted common fits per method\n4/4 frozen gap checks per method",
        transform=axis_d.transAxes,
        linespacing=1.45,
    )
    axis_d.text(
        0.06,
        0.26,
        f"Median separation: {agreement['native_proposal_distance_median']:.3f} px native\n"
        f"→ {agreement['common_center_distance_median']:.6f} px after common fit",
        transform=axis_d.transAxes,
        fontweight="bold",
        linespacing=1.45,
    )
    axis_d.text(
        0.06,
        0.09,
        "Not a sixth main-matrix method.\nNot position ground truth.",
        transform=axis_d.transAxes,
        color="#4D4D4D",
        fontweight="bold",
        linespacing=1.25,
    )
    axis_d.set_title("Compatibility scope", loc="left")

    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.13,
            1.05,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        if index < 3:
            axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "EasyLEED–ViPErLEED compatibility under the common Gaussian estimator",
        fontsize=9.2,
        y=0.985,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.96), h_pad=2.2, w_pad=1.7)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png = output_directory / f"{FIGURE_BASENAME}.png"
    pdf = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure S4 builder"},
    )
    figure.savefig(
        pdf,
        facecolor="white",
        metadata={
            "Title": "EasyLEED-ViPErLEED compatibility",
            "Author": "TrackLEED collaboration",
            "Subject": "Supplementary Figure S4",
            "Creator": "TrackLEED Figure S4 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png, "pdf": pdf}


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _code_hashes() -> dict[str, str]:
    return {
        name: sha256_file(FIGURE_ROOT / name)
        for name in ("build_figure.py", "freeze_source_data.py", "test_build_figure.py")
    }


def build(
    output_directory: Path = OUTPUT_DIRECTORY, manifest_path: Path = MANIFEST_PATH
) -> dict[str, Path]:
    validate_source_package(check_upstream=False)
    outputs = render(output_directory)
    summary = load_summary()
    import matplotlib
    import numpy
    from PIL import __version__ as pillow_version

    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure S4",
        "title": "EasyLEED-ViPErLEED compatibility under the common Gaussian estimator",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "code_sha256": _code_hashes(),
        "source_data_sha256": _hashes(SOURCE_DATA_DIRECTORY),
        "outputs": {
            name: {
                "path": path.relative_to(manifest_path.parent).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
        "branch_opportunities_per_method": summary["branch_opportunities_per_method"],
        "valid_common_fits_per_method": summary["valid_common_fits_per_method"],
        "conditional_pair_frames_per_method": summary[
            "conditional_pair_frames_per_method"
        ],
        "part_of_main_five_method_matrix": False,
        "claims_position_ground_truth": False,
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return outputs


def validate() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise ValueError("Figure S4 contract hash mismatch")
    if manifest["code_sha256"] != _code_hashes():
        raise ValueError("Figure S4 code hash mismatch")
    if manifest["source_data_sha256"] != _hashes(SOURCE_DATA_DIRECTORY):
        raise ValueError("Figure S4 source-data hash mismatch")
    validate_source_package(check_upstream=False)
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s4-render-") as tmp:
        rebuilt = render(Path(tmp))
        for name, path in rebuilt.items():
            if sha256_file(path) != manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Non-deterministic Figure S4 render: {name}")
    print("Figure S4 validation passed: separate 60-opportunity external reference.")


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
