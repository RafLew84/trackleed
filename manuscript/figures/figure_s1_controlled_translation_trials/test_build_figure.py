"""Artifact-level determinism checks for Figure S1."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trackleed.publication.controlled_translation_figure_s1 import (
    build_figure_s1_artifact,
)


FIGURE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = FIGURE_ROOT.parents[2]
SOURCE_DIRECTORY = (
    WORKSPACE_ROOT / "manuscript/source_data/controlled_translations"
)


class FigureS1ArtifactTest(unittest.TestCase):
    def test_committed_outputs_and_manifest_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_figure_s1_artifact(
                workspace_root=WORKSPACE_ROOT,
                source_directory=SOURCE_DIRECTORY,
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in (root / "output").iterdir()
                },
                {
                    path.name: path.read_bytes()
                    for path in (FIGURE_ROOT / "output").iterdir()
                },
            )
            self.assertEqual(
                json.loads((root / "manifest.json").read_text(encoding="utf-8")),
                json.loads(
                    (FIGURE_ROOT / "manifest.json").read_text(encoding="utf-8")
                ),
            )


if __name__ == "__main__":
    unittest.main()
