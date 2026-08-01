#!/usr/bin/env python3
"""Contract tests for supplementary Figure S5."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import build_figure


class FigureS5ContractTest(unittest.TestCase):
    def test_builder_validates_the_frozen_package_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            committed_outputs = {
                "png": output / f"{build_figure.FIGURE_BASENAME}.png",
                "pdf": output / "pdf" / f"{build_figure.FIGURE_BASENAME}.pdf",
            }
            for name, target in committed_outputs.items():
                source = (
                    build_figure.OUTPUT_DIRECTORY
                    / ("pdf" if name == "pdf" else "")
                    / target.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            with (
                mock.patch.object(build_figure, "validate_source_package") as validator,
                mock.patch.object(
                    build_figure,
                    "render",
                    return_value=committed_outputs,
                ),
            ):
                build_figure.build(output, root / "manifest.json")

        validator.assert_called_once_with(check_upstream=False)

    def test_examples_are_source_linked_and_selected_without_ranking(self) -> None:
        examples = build_figure.load_examples()
        selection = build_figure.load_selection_contract()

        self.assertEqual(
            [row["method"] for row in examples],
            ["locotrack", "ostrack", "dam4sam", "lk", "easyleed"],
        )
        self.assertTrue(all(row["fit_valid"] for row in examples))
        self.assertTrue(all(row["branch_error_px"] > 18.0 for row in examples))
        self.assertEqual(
            selection["geometry_inconsistent_count_by_method"],
            {"locotrack": 36, "ostrack": 34, "dam4sam": 36, "lk": 14, "easyleed": 4},
        )
        self.assertFalse(selection["selection_uses_error_magnitude"])
        self.assertFalse(selection["examples_are_ground_truth_identity_switches"])
        for row in examples:
            crop = build_figure.SOURCE_DATA_DIRECTORY / row["crop_path"]
            self.assertTrue(crop.is_file())
            self.assertEqual(build_figure.sha256_file(crop), row["crop_sha256"])

    def test_build_is_deterministic_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_figure.build(root / "output", root / "manifest.json")
            first_bytes = {name: path.read_bytes() for name, path in first.items()}
            second = build_figure.build(root / "output", root / "manifest.json")
            second_bytes = {name: path.read_bytes() for name, path in second.items()}

            self.assertEqual(first_bytes, second_bytes)
            with Image.open(first["png"]) as image:
                self.assertEqual(image.mode, "RGBA")
            self.assertTrue(first["pdf"].read_bytes().startswith(b"%PDF"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure"], "Figure S5")
            self.assertEqual(manifest["example_count"], 5)
            self.assertFalse(manifest["claims_proven_identity_switches"])
            self.assertFalse(manifest["supports_tracker_ranking"])


if __name__ == "__main__":
    unittest.main()
