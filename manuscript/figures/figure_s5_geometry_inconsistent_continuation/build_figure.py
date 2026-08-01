#!/usr/bin/env python3
"""Build and validate supplementary Figure S5."""

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
FIGURE_BASENAME = "figure_s5_geometry_inconsistent_continuation"
FIGURE_SIZE_INCHES = (7.2, 7.0)
PNG_DPI = 400
METHOD_COLORS = {
    "locotrack": "#0072B2",
    "ostrack": "#E69F00",
    "dam4sam": "#009E73",
    "lk": "#CC79A7",
    "easyleed": "#D55E00",
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


def load_examples() -> list[dict[str, object]]:
    float_fields = (
        "energy_ev",
        "native_x",
        "native_y",
        "predicted_x",
        "predicted_y",
        "center_x",
        "center_y",
        "branch_error_px",
        "r_squared",
        "amplitude",
        "sigma_x",
        "sigma_y",
    )
    int_fields = ("frame", "crop_x0", "crop_y0", "crop_x1", "crop_y1")
    return [
        {
            **row,
            "fit_valid": row["fit_valid"] == "true",
            **{field: float(row[field]) for field in float_fields},
            **{field: int(row[field]) for field in int_fields},
        }
        for row in _read_csv(SOURCE_DATA_DIRECTORY / "example_records.csv")
    ]


def load_selection_contract() -> dict[str, object]:
    return json.loads(
        (SOURCE_DATA_DIRECTORY / "selection_contract.json").read_text(encoding="utf-8")
    )


def render(output_directory: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    examples = load_examples()
    selection = load_selection_contract()
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_INCHES, facecolor="white")
    for index, (axis, row) in enumerate(zip(axes.flat, examples)):
        crop_path = SOURCE_DATA_DIRECTORY / str(row["crop_path"])
        with Image.open(crop_path) as image:
            pixels = np.asarray(image, dtype=float)
        finite = pixels[np.isfinite(pixels)]
        vmin, vmax = np.percentile(finite, [1.0, 99.7])
        axis.imshow(pixels, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        x0 = int(row["crop_x0"])
        y0 = int(row["crop_y0"])
        expected = (float(row["predicted_x"]) - x0, float(row["predicted_y"]) - y0)
        native = (float(row["native_x"]) - x0, float(row["native_y"]) - y0)
        center = (float(row["center_x"]) - x0, float(row["center_y"]) - y0)
        color = METHOD_COLORS[str(row["method"])]
        axis.scatter(
            *expected, marker="x", s=52, linewidth=1.5, color="#111111", zorder=4
        )
        axis.scatter(
            *native,
            marker="^",
            s=38,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axis.scatter(
            *center,
            marker="o",
            s=42,
            facecolor="none",
            edgecolor="#FFFFFF",
            linewidth=1.3,
            zorder=6,
        )
        axis.plot(
            [expected[0], center[0]],
            [expected[1], center[1]],
            color="#FFFFFF",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
        axis.set_title(
            str(row["display_method"]), loc="left", color=color, fontweight="bold"
        )
        axis.text(
            0.02,
            0.02,
            f"{row['series']}, {row['pair']}, {row['energy_ev']:.1f} eV\n"
            f"{row['physics_baseline_status'].replace('_', ' ')}\n"
            f"branch error {row['branch_error_px']:.1f} px (>18 px)",
            transform=axis.transAxes,
            color="white",
            fontsize=5.8,
            va="bottom",
            bbox={
                "facecolor": "#000000",
                "alpha": 0.58,
                "edgecolor": "none",
                "pad": 2.0,
            },
        )
        axis.text(
            -0.08,
            1.05,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        axis.set_xticks([])
        axis.set_yticks([])

    contract_axis = axes.flat[-1]
    contract_axis.set_axis_off()
    contract_axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.91,
            transform=contract_axis.transAxes,
            boxstyle="round,pad=0.012",
            facecolor="#F7F7F7",
            edgecolor="#777777",
            linestyle=(0, (4, 2)),
        )
    )
    contract_axis.set_title(
        "Selection and interpretation", loc="left", fontweight="bold"
    )
    contract_axis.text(
        0.07,
        0.84,
        "First qualifying source row per method\n"
        "in frozen case/frame/branch order.\n"
        "No selection by error magnitude.",
        transform=contract_axis.transAxes,
        linespacing=1.35,
    )
    counts = selection["geometry_inconsistent_count_by_method"]
    denominators = selection["accepted_fit_denominator_by_method"]
    contract_axis.text(
        0.07,
        0.58,
        "Full outcome counts (not ranks):\n"
        + "\n".join(
            f"{row['display_method']}: {counts[row['method']]}/{denominators[row['method']]} accepted fits"
            for row in examples
        ),
        transform=contract_axis.transAxes,
        fontsize=5.8,
        linespacing=1.3,
    )
    contract_axis.text(
        0.07,
        0.13,
        "Possible false continuity;\nnot proven identity switches\nand not tracker ranking.",
        transform=contract_axis.transAxes,
        fontweight="bold",
        color="#A23B3B",
        linespacing=1.3,
    )
    contract_axis.text(
        -0.08,
        1.05,
        "f",
        transform=contract_axis.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="#111111",
            linestyle="none",
            label="expected branch",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#777777",
            markerfacecolor="#777777",
            linestyle="none",
            label="native proposal",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#777777",
            markerfacecolor="none",
            linestyle="none",
            label="Gaussian center",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=3,
        frameon=False,
    )
    figure.suptitle(
        "Source-linked geometry-inconsistent continuation examples",
        fontsize=9.2,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Every crop is hash-linked to its frozen source row and raw image; examples do not supply identity ground truth.",
        ha="center",
        fontsize=6.2,
        fontweight="bold",
        color="#4D4D4D",
    )
    figure.tight_layout(rect=(0.02, 0.035, 1.0, 0.91), h_pad=1.5, w_pad=1.2)
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory = output_directory / "pdf"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    png = output_directory / f"{FIGURE_BASENAME}.png"
    pdf = pdf_directory / f"{FIGURE_BASENAME}.pdf"
    figure.savefig(
        png,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "TrackLEED Figure S5 builder"},
    )
    figure.savefig(
        pdf,
        facecolor="white",
        metadata={
            "Title": "Geometry-inconsistent continuation examples",
            "Author": "TrackLEED collaboration",
            "Subject": "Supplementary Figure S5",
            "Creator": "TrackLEED Figure S5 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png, "pdf": pdf}


def _source_hashes() -> dict[str, str]:
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(SOURCE_DATA_DIRECTORY.rglob("*"))
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
    import matplotlib
    import numpy
    from PIL import __version__ as pillow_version

    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure S5",
        "title": "Source-linked examples of geometry-inconsistent continuation",
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
        "example_count": len(load_examples()),
        "selection_rule": load_selection_contract()["selection_rule"],
        "claims_proven_identity_switches": False,
        "supports_tracker_ranking": False,
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
        raise ValueError("Figure S5 contract hash mismatch")
    if manifest["code_sha256"] != _code_hashes():
        raise ValueError("Figure S5 code hash mismatch")
    if manifest["source_data_sha256"] != _source_hashes():
        raise ValueError("Figure S5 source-data hash mismatch")
    validate_source_package(check_upstream=False)
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s5-render-") as tmp:
        rebuilt = render(Path(tmp))
        for name, path in rebuilt.items():
            if sha256_file(path) != manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Non-deterministic Figure S5 render: {name}")
    print("Figure S5 validation passed: five source-linked explanatory examples.")


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
