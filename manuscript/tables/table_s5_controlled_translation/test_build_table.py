"""Artifact-level tests for supplementary Table S5."""

from __future__ import annotations

import json
import tempfile
import unittest
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


class TableS5ArtifactTest(unittest.TestCase):
    def test_committed_artifact_is_deterministic_and_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            manifest = root / "manifest.json"
            build_table_s5_artifact(
                workspace_root=WORKSPACE_ROOT,
                source_directory=SOURCE_DIRECTORY,
                output_directory=output,
                manifest_path=manifest,
            )

            expected_output = {
                path.name: path.read_bytes()
                for path in (TABLE_ROOT / "output").iterdir()
            }
            actual_output = {
                path.name: path.read_bytes() for path in output.iterdir()
            }
            self.assertEqual(actual_output, expected_output)
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8")),
                json.loads(
                    (TABLE_ROOT / "manifest.json").read_text(encoding="utf-8")
                ),
            )

        markdown = (
            TABLE_ROOT / "output/table_s5_controlled_translation.md"
        ).read_text(encoding="utf-8")
        supplement = (
            WORKSPACE_ROOT / "manuscript/supplementary-material.md"
        ).read_text(encoding="utf-8")
        self.assertIn(f"<!-- BEGIN TABLE S5 -->\n{markdown.rstrip()}\n", supplement)

    def test_synchronization_replaces_exactly_one_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            supplement = Path(temporary_directory) / "supplement.md"
            supplement.write_text("# Supplement\n", encoding="utf-8")
            table = Path(temporary_directory) / "table.md"
            table.write_text("first\n", encoding="utf-8")
            synchronize_table_s5(table, supplement)
            table.write_text("second\n", encoding="utf-8")
            synchronize_table_s5(table, supplement)
            rendered = supplement.read_text(encoding="utf-8")
            self.assertEqual(rendered.count("<!-- BEGIN TABLE S5 -->"), 1)
            self.assertEqual(rendered.count("<!-- END TABLE S5 -->"), 1)
            self.assertIn("second", rendered)
            self.assertNotIn("first", rendered)


if __name__ == "__main__":
    unittest.main()
