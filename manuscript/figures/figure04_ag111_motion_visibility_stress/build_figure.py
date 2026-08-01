#!/usr/bin/env python3
"""Build and validate publication Figure 4."""

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
    PNG_DPI,
    SOURCE_DATA_DIRECTORY,
    load_representative_pair_source_data,
    sha256_file,
    validate_source_contract,
)
from figure_render import render_figure, summarize_representative_trajectory


CODE_FILES = (
    "build_figure.py",
    "figure_contract.py",
    "figure_render.py",
    "freeze_source_data.py",
    "figure_test_support.py",
    "test_build_figure.py",
    "test_freeze_source_data.py",
)


def _code_hashes() -> dict[str, str]:
    return {filename: sha256_file(FIGURE_ROOT / filename) for filename in CODE_FILES}


def _source_data_hashes() -> dict[str, str]:
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(SOURCE_DATA_DIRECTORY.iterdir())
        if path.is_file()
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_manifest(outputs: dict[str, Path]) -> dict[str, object]:
    import matplotlib
    import numpy
    from matplotlib import font_manager
    from PIL import Image, __version__ as pillow_version

    with Image.open(outputs["png"]) as image:
        dimensions = list(image.size)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(encoding="utf-8")
    )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    trajectory = summarize_representative_trajectory(
        load_representative_pair_source_data()
    )
    return {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure 4",
        "title": "Ag(111) motion and visibility-gap stress test",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "code_sha256": _code_hashes(),
        "source_artifacts": provenance["sources"],
        "source_data_sha256": _source_data_hashes(),
        "outputs_sha256": {
            path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
            for path in outputs.values()
        },
        "rendering": {
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "png_dpi": PNG_DPI,
            "png_dimensions_px": dimensions,
            "font_family": FONT_FAMILY,
            "font_file": Path(font_manager.findfont(FONT_FAMILY)).name,
            "font_sizes_pt": FONT_SIZES_PT,
            "method_palette": METHOD_COLORS,
            "panel_order": contract["panel_order"],
            "representative_trajectory_encoding": {
                "point": (
                    "cross-method median of available valid Gaussian pair measurements"
                ),
                "whisker": "cross-method minimum-to-maximum",
                "grouping": "image frame; repeated energies retained",
                "method_specific_markers": False,
                "maximum_cross_method_range_px": (trajectory.maximum_range_px),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
        "claim_limits": contract["claim_limits"],
    }


def build() -> None:
    validate_source_contract()
    outputs = render_figure(OUTPUT_DIRECTORY)
    _write_json(MANIFEST_PATH, build_manifest(outputs))


def validate() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Figure 4 manifest is missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Figure 4 manifest is not completed")
    if manifest.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("Figure 4 contract hash mismatch")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Figure 4 code hash mismatch")
    if manifest.get("rendering", {}).get("font_sizes_pt") != FONT_SIZES_PT:
        raise ValueError("Figure 4 font-size contract mismatch")
    for relative_path, expected_hash in manifest["source_data_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Figure 4 source hash mismatch: {relative_path}")
    for relative_path, expected_hash in manifest["outputs_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Figure 4 output hash mismatch: {relative_path}")

    validate_source_contract()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure04-render-") as temporary:
        rebuilt = render_figure(Path(temporary))
        expected_by_suffix = {
            Path(path).suffix: expected
            for path, expected in manifest["outputs_sha256"].items()
        }
        for path in rebuilt.values():
            if sha256_file(path) != expected_by_suffix[path.suffix]:
                raise ValueError(f"Non-deterministic Figure 4 render: {path.suffix}")
    print(
        "Figure 4 validation passed: 5-method Ag(111) stress test with "
        "coverage and recovery before inconsistency and conditional residual."
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
