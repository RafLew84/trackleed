#!/usr/bin/env python3
"""Build and validate publication Figure 2."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

from figure_contract import (
    FIGURE_ROOT,
    FONT_SIZES_PT,
    METHOD_COLORS,
    OUTPUT_DIRECTORY,
    SOURCE_DATA_DIRECTORY,
    as_int,
    load_method_metrics,
    refresh_source_data,
    sha256_file,
    write_json,
)
from figure_render import render_figure


CODE_FILES = (
    "build_figure.py",
    "figure_contract.py",
    "figure_render.py",
    "test_build_figure.py",
)
PANEL_ORDER = (
    "translation_path",
    "native_vs_refined_displacement",
    "measurement_coverage",
    "pair_invariance",
)


def _source_data_hashes() -> dict[str, str]:
    files = sorted(
        path for path in SOURCE_DATA_DIRECTORY.iterdir() if path.is_file()
    )
    return {
        path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
        for path in files
    }


def _code_hashes() -> dict[str, str]:
    return {
        filename: sha256_file(FIGURE_ROOT / filename)
        for filename in CODE_FILES
    }


def build_manifest(outputs: dict[str, Path]) -> dict[str, object]:
    import matplotlib
    import numpy
    from matplotlib import font_manager
    from PIL import Image, __version__ as pillow_version

    with Image.open(outputs["png"]) as image:
        png_dimensions = list(image.size)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": "1.1",
        "status": "completed",
        "figure": "Figure 2",
        "title": "Controlled estimator behavior",
        "panel_order": list(PANEL_ORDER),
        "code_sha256": _code_hashes(),
        "source_artifacts": provenance["sources"],
        "source_data_sha256": _source_data_hashes(),
        "outputs_sha256": {
            path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
            for path in outputs.values()
        },
        "rendering": {
            "figure_size_inches": [7.2, 5.3],
            "png_dpi": 400,
            "png_dimensions_px": png_dimensions,
            "font_family": "DejaVu Sans",
            "font_file": Path(font_manager.findfont("DejaVu Sans")).name,
            "font_sizes_pt": FONT_SIZES_PT,
            "method_palette": METHOD_COLORS,
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
        "claim_limit": (
            "Known translations test measurement-chain equivariance, not "
            "absolute tracking accuracy or identity ground truth."
        ),
    }


def build() -> None:
    outputs = render_figure(OUTPUT_DIRECTORY)
    write_json(FIGURE_ROOT / "manifest.json", build_manifest(outputs))


def _validate_contract() -> None:
    methods = load_method_metrics(SOURCE_DATA_DIRECTORY / "method_metrics.csv")
    if [as_int(row["valid_gaussian_measurements"]) for row in methods] != [
        80,
        80,
        80,
        76,
        74,
    ]:
        raise ValueError("Measurement-coverage denominators changed")
    if [as_int(row["valid_pair_measurements"]) for row in methods] != [
        40,
        40,
        40,
        36,
        34,
    ]:
        raise ValueError("Pair-invariance denominators changed")
    if any(as_int(row["pair_opportunities"]) != 40 for row in methods):
        raise ValueError("Pair-opportunity denominator changed")



def validate() -> None:
    manifest_path = FIGURE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Figure manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Figure manifest is not completed")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Figure code hash mismatch")
    if manifest.get("rendering", {}).get("font_sizes_pt") != FONT_SIZES_PT:
        raise ValueError("Figure font-size contract mismatch")

    for relative_path, expected_hash in manifest["source_data_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Source-data hash mismatch: {relative_path}")
    for relative_path, expected_hash in manifest["outputs_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Output hash mismatch: {relative_path}")

    _validate_contract()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure02-") as temporary:
        rebuilt = render_figure(Path(temporary))
        expected_by_suffix = {
            Path(path).suffix: expected
            for path, expected in manifest["outputs_sha256"].items()
        }
        for path in rebuilt.values():
            if sha256_file(path) != expected_by_suffix[path.suffix]:
                raise ValueError(f"Non-deterministic rebuild: {path.suffix}")

    print(
        "Figure 2 validation passed: 5 methods, 80 spot and 40 pair "
        "opportunities/method."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-source-data",
        action="store_true",
        help="Regenerate committed source data from frozen local artifacts.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate hashes and a deterministic rebuild without rewriting outputs.",
    )
    arguments = parser.parse_args()

    if arguments.validate_only:
        validate()
        return
    if arguments.refresh_source_data:
        refresh_source_data()
    build()
    validate()


if __name__ == "__main__":
    main()
