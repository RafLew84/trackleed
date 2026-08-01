#!/usr/bin/env python3
"""Build and synchronize supplementary Table S5."""

from __future__ import annotations

from pathlib import Path

from trackleed.publication.controlled_translation_table import (
    build_table_s5_artifact,
    synchronize_table_s5,
)


TABLE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = TABLE_ROOT.parents[2]
SOURCE_DIRECTORY = (
    WORKSPACE_ROOT / "manuscript/source_data/controlled_translations"
)
OUTPUT_DIRECTORY = TABLE_ROOT / "output"
MANIFEST_PATH = TABLE_ROOT / "manifest.json"
SUPPLEMENT_PATH = WORKSPACE_ROOT / "manuscript/supplementary-material.md"


def main() -> None:
    build_table_s5_artifact(
        workspace_root=WORKSPACE_ROOT,
        source_directory=SOURCE_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        manifest_path=MANIFEST_PATH,
    )
    synchronize_table_s5(
        OUTPUT_DIRECTORY / "table_s5_controlled_translation.md",
        SUPPLEMENT_PATH,
    )


if __name__ == "__main__":
    main()
