#!/usr/bin/env python3
"""Build and validate publication Figure 5."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

from figure_contract import (
    CONTRACT_PATH,
    FIGURE_ROOT,
    FIGURE_SIZE_INCHES,
    FONT_FAMILY,
    FONT_SIZES_PT,
    MANIFEST_PATH,
    METHOD_COLORS,
    OUTPUT_DIRECTORY,
    PANEL_ORDER,
    PNG_DPI,
    SOURCE_DATA_DIRECTORY,
    load_representative_gap_source_data,
    sha256_file,
    shared_crop_intensity_limits,
    validate_source_contract,
)
from figure_render import render_figure


CODE_FILES = (
    "build_figure.py",
    "figure_contract.py",
    "figure_render.py",
    "freeze_source_data.py",
    "test_build_figure.py",
    "test_freeze_source_data.py",
    "test_source_contract.py",
)


def _code_hashes() -> dict[str, str]:
    return {
        filename: sha256_file(FIGURE_ROOT / filename)
        for filename in CODE_FILES
    }


def _source_data_hashes() -> dict[str, str]:
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(SOURCE_DATA_DIRECTORY.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative_output_paths(
    outputs: dict[str, Path],
    manifest_path: Path,
) -> dict[str, str]:
    return {
        name: path.relative_to(manifest_path.parent).as_posix()
        for name, path in outputs.items()
    }


def build_manifest(
    outputs: dict[str, Path],
    manifest_path: Path,
) -> dict[str, object]:
    import matplotlib
    import numpy
    from matplotlib import font_manager
    from PIL import Image, __version__ as pillow_version

    with Image.open(outputs["png"]) as image:
        dimensions = list(image.size)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    intensity_limits = shared_crop_intensity_limits()
    representative_rows = load_representative_gap_source_data()
    native_proposals = sum(
        bool(row["native_visible"]) for row in representative_rows
    )
    accepted_centers = sum(
        bool(row["measurement_valid"]) for row in representative_rows
    )
    rejected_centers = sum(
        not bool(row["measurement_valid"])
        and row["gaussian_center_x"] is not None
        for row in representative_rows
    )
    return {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure 5",
        "title": "Controlled SAM2-family continuity–consistency trade-off",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "code_sha256": _code_hashes(),
        "source_artifacts": provenance["sources"],
        "source_data_sha256": _source_data_hashes(),
        "output_files": _relative_output_paths(outputs, manifest_path),
        "outputs_sha256": {
            name: sha256_file(path) for name, path in outputs.items()
        },
        "rendering": {
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "png_dpi": PNG_DPI,
            "png_dimensions_px": dimensions,
            "font_family": FONT_FAMILY,
            "font_file": Path(font_manager.findfont(FONT_FAMILY)).name,
            "font_sizes_pt": FONT_SIZES_PT,
            "method_palette": METHOD_COLORS,
            "panel_order": list(PANEL_ORDER),
            "representative_crop_normalization": {
                "mode": "shared_linear",
                "vmin": intensity_limits[0],
                "vmax": intensity_limits[1],
            },
            "representative_panel_states": {
                "native_proposals": native_proposals,
                "accepted_gaussian_centers": accepted_centers,
                "rejected_gaussian_centers": rejected_centers,
                "no_proposal_states": len(representative_rows)
                - native_proposals,
                "outcome_selected_for_explanation": contract[
                    "representative_gap"
                ]["selection_uses_method_outcomes"],
            },
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
        "caption_limits": contract["caption_limits"],
    }


def build(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Path]:
    validate_source_contract()
    outputs = render_figure(output_directory)
    _write_json(manifest_path, build_manifest(outputs, manifest_path))
    return outputs


def validate(
    output_directory: Path = OUTPUT_DIRECTORY,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError("Figure 5 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Figure 5 manifest is not completed")
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("Figure 5 contract hash mismatch")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Figure 5 code hash mismatch")
    if manifest["rendering"]["panel_order"] != list(PANEL_ORDER):
        raise ValueError("Figure 5 panel-order manifest mismatch")
    if manifest["rendering"]["font_sizes_pt"] != FONT_SIZES_PT:
        raise ValueError("Figure 5 font-size manifest mismatch")
    for relative_path, expected_hash in manifest["source_data_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Figure 5 source hash mismatch: {relative_path}")
    for name, relative_path in manifest["output_files"].items():
        path = manifest_path.parent / relative_path
        if (
            not path.is_file()
            or sha256_file(path) != manifest["outputs_sha256"][name]
        ):
            raise ValueError(f"Figure 5 output hash mismatch: {name}")

    validate_source_contract()
    with tempfile.TemporaryDirectory(
        prefix="trackleed-figure05-render-"
    ) as temporary:
        rebuilt = render_figure(Path(temporary))
        for name, path in rebuilt.items():
            if sha256_file(path) != manifest["outputs_sha256"][name]:
                raise ValueError(
                    f"Non-deterministic Figure 5 render: {name}"
                )
    print(
        "Figure 5 validation passed: matched SAM2-family memory comparison "
        "with shared-scale visibility-gap images, pairwise coverage and "
        "recovery, false-continuity cost, and conditional D(E) residuals."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-source-data",
        action="store_true",
        help="Rebuild committed source data from frozen upstream artifacts.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate hashes and a deterministic render without rewriting outputs.",
    )
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
