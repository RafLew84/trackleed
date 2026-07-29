#!/usr/bin/env python3
"""Build and validate publication Figure 1."""

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
    RENDER_MANIFEST,
    SOURCE_CASE_MANIFEST,
    SOURCE_DATA_DIRECTORY,
    load_contract,
    sha256_file,
    validate_render_contract,
)
from figure_render import render_figure


CODE_FILES = (
    "build_figure.py",
    "figure_contract.py",
    "figure_render.py",
    "test_build_figure.py",
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
        png_dimensions = list(image.size)
    provenance = json.loads(
        (SOURCE_DATA_DIRECTORY / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": "1.0",
        "status": "completed",
        "figure": "Figure 1",
        "title": "Measurement chain and evidence ladder",
        "code_sha256": _code_hashes(),
        "frozen_case_manifest_sha256": sha256_file(SOURCE_CASE_MANIFEST),
        "source_artifacts": provenance["sources"],
        "source_data_sha256": _source_data_hashes(),
        "outputs_sha256": {
            path.relative_to(FIGURE_ROOT).as_posix(): sha256_file(path)
            for path in outputs.values()
        },
        "rendering": {
            "figure_size_inches": [7.2, 6.8],
            "png_dpi": 400,
            "png_dimensions_px": png_dimensions,
            "font_family": "DejaVu Sans",
            "font_file": Path(font_manager.findfont("DejaVu Sans")).name,
            "font_sizes_pt": FONT_SIZES_PT,
            "method_palette": METHOD_COLORS,
            "native_geometry_policy": (
                "Empirical OSTrack bbox coordinates and unmodified DAM4SAM "
                "binary mask pixels."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pillow": pillow_version,
        },
        "claim_limit": load_contract()["claim_limit"],
    }


def build() -> None:
    validate_render_contract()
    outputs = render_figure(OUTPUT_DIRECTORY)
    _write_json(RENDER_MANIFEST, build_manifest(outputs))


def validate() -> None:
    if not RENDER_MANIFEST.is_file():
        raise FileNotFoundError("Figure 1 render manifest is missing")
    manifest = json.loads(RENDER_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Figure 1 render manifest is not completed")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("Figure 1 rendering code hash mismatch")
    if manifest.get("frozen_case_manifest_sha256") != sha256_file(
        SOURCE_CASE_MANIFEST
    ):
        raise ValueError("Figure 1 frozen-case manifest hash mismatch")
    if manifest.get("rendering", {}).get("font_sizes_pt") != FONT_SIZES_PT:
        raise ValueError("Figure 1 font-size contract mismatch")

    for relative_path, expected_hash in manifest["source_data_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Source-data hash mismatch: {relative_path}")
    for relative_path, expected_hash in manifest["outputs_sha256"].items():
        path = FIGURE_ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Output hash mismatch: {relative_path}")

    validate_render_contract()
    with tempfile.TemporaryDirectory(prefix="trackleed-figure01-") as temporary:
        rebuilt = render_figure(Path(temporary))
        expected_by_suffix = {
            Path(path).suffix: expected
            for path, expected in manifest["outputs_sha256"].items()
        }
        for path in rebuilt.values():
            if sha256_file(path) != expected_by_suffix[path.suffix]:
                raise ValueError(f"Non-deterministic rebuild: {path.suffix}")

    print(
        "Figure 1 validation passed: 5 empirical native outputs, common "
        "48 x 48 ROI conversion, Gaussian endpoint, pairwise geometry and "
        "4 evidence roles."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate hashes and a deterministic rebuild without rewriting outputs.",
    )
    arguments = parser.parse_args()
    if arguments.validate_only:
        validate()
        return
    build()
    validate()


if __name__ == "__main__":
    main()
