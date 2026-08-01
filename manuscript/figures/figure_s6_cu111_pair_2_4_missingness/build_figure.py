#!/usr/bin/env python3
"""Build and validate supplementary Figure S6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tempfile
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = FIGURE_ROOT / "contract.json"
SOURCE_DATA_DIRECTORY = FIGURE_ROOT / "source_data"
SOURCE_DATA_PATH = SOURCE_DATA_DIRECTORY / "pair_2_4_missingness_diagnostic.csv"
PROVENANCE_PATH = SOURCE_DATA_DIRECTORY / "source_provenance.json"
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"
FIGURE_BASENAME = "figure_s6_cu111_pair_2_4_missingness"
FIGURE_SIZE_INCHES = (7.0, 4.5)
PNG_DPI = 400
FONT_FAMILY = "DejaVu Sans"
CODE_FILES = ("build_figure.py", "freeze_source_data.py", "test_build_figure.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected explicit boolean, got {value!r}")
    return normalized == "true"


def _load_contract() -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "source_contract_frozen":
        raise ValueError("Figure S6 source contract is not frozen")
    if contract.get("figure") != "Figure S6":
        raise ValueError("Figure S6 identity changed")
    if contract.get("inside_five_method_run") is not False:
        raise ValueError("Figure S6 must remain outside the five-method run")
    return contract


def load_and_validate_rows(
    path: Path = SOURCE_DATA_PATH,
) -> list[dict[str, object]]:
    """Load the diagnostic through its publication-facing source seam."""

    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    boolean_fields = (
        "inside_five_method_run",
        "tracking_proposal_present",
        "exported_fit_present",
        "refit_returned_parameters",
        "r_squared_gate",
        "amplitude_gate",
        "center_gate",
        "sigma_gate",
        "refit_accepted",
    )
    rows = [
        {
            **row,
            "frame_id": int(row["frame_id"]),
            "temperature_c": float(row["temperature_c"]),
            "energy_ev": float(row["energy_ev"]),
            **{field: _as_bool(row[field]) for field in boolean_fields},
            "r_squared": float(row["r_squared"]),
            "amplitude": float(row["amplitude"]),
            "amplitude_threshold": float(row["amplitude_threshold"]),
        }
        for row in raw_rows
    ]
    contract = _load_contract()
    expected_frames = list(
        range(
            int(contract["context_frame_start"]),
            int(contract["context_frame_end"]) + 1,
        )
    )
    if [row["frame_id"] for row in rows] != expected_frames:
        raise ValueError("Figure S6 diagnostic frame set changed")
    if {row["pair"] for row in rows} != {contract["pair"]}:
        raise ValueError("Figure S6 diagnostic pair changed")
    if {row["tracking_source"] for row in rows} != {contract["tracking_source"]}:
        raise ValueError("Figure S6 tracking source changed")
    if any(row["inside_five_method_run"] for row in rows):
        raise ValueError("Figure S6 must stay outside the five-method run")
    if sum(row["tracking_proposal_present"] for row in rows) != int(
        contract["tracking_proposal_present_count"]
    ):
        raise ValueError("Figure S6 proposal-availability count changed")
    rejected = [row for row in rows if not row["refit_accepted"]]
    if len(rejected) != int(contract["rejected_fit_count"]):
        raise ValueError("Figure S6 rejected-fit count changed")
    if any(
        not row["tracking_proposal_present"]
        or not row["r_squared_gate"]
        or row["amplitude_gate"]
        or not row["center_gate"]
        or not row["sigma_gate"]
        or row["rejection_reason"] != "amplitude_below_or_outside_threshold"
        or row["amplitude_threshold"] != float(contract["amplitude_threshold"])
        for row in rejected
    ):
        raise ValueError("Figure S6 rejections are no longer amplitude-only")
    return rows


def validate_provenance() -> dict[str, object]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    source = provenance.get("source", {})
    if provenance.get("figure") != "Figure S6":
        raise ValueError("Figure S6 provenance identity changed")
    if source.get("evidence_role") != "separate saved-proposal diagnostic":
        raise ValueError("Figure S6 provenance role changed")
    if source.get("frozen_source_sha256") != sha256_file(SOURCE_DATA_PATH):
        raise ValueError("Figure S6 frozen source hash mismatch")
    origin_path = Path(str(source.get("origin_path", "")))
    if origin_path.is_absolute() or ".." in origin_path.parts:
        raise ValueError("Figure S6 provenance path escapes the repository")
    if len(str(source.get("origin_sha256", ""))) != 64:
        raise ValueError("Figure S6 provenance origin hash is malformed")
    return provenance


def render_figure(
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = load_and_validate_rows()
    ordered = sorted(rows, key=lambda row: int(row["frame_id"]))
    x = [100.0 * index / (len(ordered) - 1) for index in range(len(ordered))]
    amplitudes = [float(row["amplitude"]) for row in ordered]
    accepted = [bool(row["refit_accepted"]) for row in ordered]
    threshold = float(ordered[0]["amplitude_threshold"])

    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    figure.subplots_adjust(left=0.12, right=0.89, bottom=0.16, top=0.87)
    axis.plot(x, amplitudes, color="#737373", linewidth=1.0, zorder=1)
    axis.scatter(
        [value for value, state in zip(x, accepted) if not state],
        [value for value, state in zip(amplitudes, accepted) if not state],
        s=35,
        marker="o",
        color="#A23B3B",
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    axis.scatter(
        [value for value, state in zip(x, accepted) if state],
        [value for value, state in zip(amplitudes, accepted) if state],
        s=48,
        marker="D",
        color="#009E73",
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    axis.axhline(
        threshold,
        color="#222222",
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        zorder=2,
    )
    plateau_indices = [
        index
        for index, row in enumerate(ordered)
        if float(row["temperature_c"]) == 800.0
    ]
    axis.axvspan(
        x[min(plateau_indices)] - 1.5,
        x[max(plateau_indices)] + 1.5,
        color="#E69F00",
        alpha=0.13,
        linewidth=0,
        zorder=0,
    )
    axis.set_xlim(-1.0, 101.0)
    axis.set_ylim(0.039, 0.054)
    axis.set_xticks((0.0, 72.0, 100.0))
    axis.set_xticklabels(("before", "800 °C plateau", "after"))
    axis.set_xlabel("Local sequence around the intensity minimum")
    axis.set_ylabel("Gaussian amplitude")
    axis.set_title("Cu(111) pair 2-4 saved-proposal missingness diagnostic", loc="left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.55)

    temperature_axis = axis.twinx()
    temperature_axis.plot(
        x,
        [float(row["temperature_c"]) for row in ordered],
        color="#929292",
        linestyle=(0, (1.5, 2)),
        linewidth=0.85,
        alpha=0.75,
        zorder=0,
    )
    temperature_axis.set_ylim(775.0, 805.0)
    temperature_axis.set_ylabel("Temperature [°C]", color="#6F6F6F")
    temperature_axis.tick_params(axis="y", colors="#6F6F6F")
    temperature_axis.spines["top"].set_visible(False)
    temperature_axis.spines["right"].set_color("#B5B5B5")
    axis.text(
        0.98,
        0.96,
        "Earlier saved YOLO trajectory - outside the five-method run",
        transform=axis.transAxes,
        fontsize=7.0,
        color="#4D4D4D",
        ha="right",
        va="top",
    )
    axis.text(
        0.52,
        0.84,
        "proposal present 26/26\n24/24 rejected fits fail only amplitude gate",
        transform=axis.transAxes,
        fontsize=7.3,
        fontweight="bold",
        color="#A23B3B",
        ha="center",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#009E73",
                markeredgecolor="white",
                label="accepted fit",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#A23B3B",
                markeredgecolor="white",
                label="amplitude-gate rejection",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle=(0, (4, 2)),
                label="amplitude threshold 0.05",
            ),
        ],
        frameon=False,
        loc="lower left",
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.4,
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
        metadata={"Software": "TrackLEED Figure S6 builder"},
    )
    figure.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Cu pair 2–4 missingness diagnostic",
            "Author": "TrackLEED collaboration",
            "Subject": "Publication Figure S6",
            "Creator": "TrackLEED Figure S6 builder",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _code_hashes() -> dict[str, str]:
    return {name: sha256_file(FIGURE_ROOT / name) for name in CODE_FILES}


def build(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Path]:
    rows = load_and_validate_rows()
    provenance = validate_provenance()
    outputs = render_figure(output_directory)
    rejected = [row for row in rows if not row["refit_accepted"]]
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure S6",
        "title": "Cu pair 2–4 missingness diagnostic",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "caption_sha256": sha256_file(FIGURE_ROOT / "README.md"),
        "code_sha256": _code_hashes(),
        "source_data_sha256": {
            SOURCE_DATA_PATH.relative_to(FIGURE_ROOT).as_posix(): (
                sha256_file(SOURCE_DATA_PATH)
            ),
            PROVENANCE_PATH.relative_to(FIGURE_ROOT).as_posix(): (
                sha256_file(PROVENANCE_PATH)
            ),
        },
        "source_provenance": provenance["source"],
        "output_files": {
            name: path.relative_to(manifest_path.parent).as_posix()
            for name, path in outputs.items()
        },
        "outputs_sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "tracking_proposal_present": (
            f"{sum(row['tracking_proposal_present'] for row in rows)}/{len(rows)}"
        ),
        "amplitude_only_rejections": f"{len(rejected)}/{len(rejected)}",
        "inside_five_method_run": False,
        "rendering": {
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "png_dpi": PNG_DPI,
            "font_family": FONT_FAMILY,
            "missing_value_rendering": (
                "explicit rejected-fit markers; no accepted endpoint is imputed"
            ),
        },
        "environment": {
            "python": platform.python_version(),
        },
        "caption_limits": _load_contract()["limitations"],
    }
    _write_json(manifest_path, manifest)
    return outputs


def validate(manifest_path: Path = MANIFEST_PATH) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Figure S6 manifest is not completed")
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("Figure S6 contract hash mismatch")
    if manifest.get("caption_sha256") != sha256_file(FIGURE_ROOT / "README.md"):
        raise ValueError("Figure S6 caption hash mismatch")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Figure S6 code hash mismatch")
    expected_source_hashes = {
        SOURCE_DATA_PATH.relative_to(FIGURE_ROOT).as_posix(): (
            sha256_file(SOURCE_DATA_PATH)
        ),
        PROVENANCE_PATH.relative_to(FIGURE_ROOT).as_posix(): (
            sha256_file(PROVENANCE_PATH)
        ),
    }
    if manifest.get("source_data_sha256") != expected_source_hashes:
        raise ValueError("Figure S6 source-data hash mismatch")
    load_and_validate_rows()
    validate_provenance()
    for name, relative_path in manifest["output_files"].items():
        path = manifest_path.parent / relative_path
        if not path.is_file() or sha256_file(path) != manifest["outputs_sha256"][name]:
            raise ValueError(f"Figure S6 output hash mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="trackleed-figure-s6-render-") as temporary:
        rebuilt = render_figure(Path(temporary))
        for name, path in rebuilt.items():
            if sha256_file(path) != manifest["outputs_sha256"][name]:
                raise ValueError(f"Non-deterministic Figure S6 render: {name}")
    print(
        "Figure S6 validation passed: 26/26 proposals present and 24/24 "
        "rejected fits failing only the amplitude gate outside the five-method run."
    )


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
