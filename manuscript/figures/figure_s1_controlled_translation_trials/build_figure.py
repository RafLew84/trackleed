#!/usr/bin/env python3
"""Build supplementary Figure S1 from authenticated controlled translations."""

from __future__ import annotations

from pathlib import Path

from trackleed.publication.controlled_translation_figure_s1 import (
    build_figure_s1_artifact,
)


FIGURE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DIRECTORY = (
    WORKSPACE_ROOT / "manuscript/source_data/controlled_translations"
)
OUTPUT_DIRECTORY = FIGURE_ROOT / "output"
MANIFEST_PATH = FIGURE_ROOT / "manifest.json"


def main() -> None:
    build_figure_s1_artifact(
        workspace_root=WORKSPACE_ROOT,
        source_directory=SOURCE_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        manifest_path=MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
