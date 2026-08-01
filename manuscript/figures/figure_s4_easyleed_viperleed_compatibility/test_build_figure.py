#!/usr/bin/env python3
"""Contract tests for supplementary Figure S4."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import build_figure


class FigureS4ContractTest(unittest.TestCase):
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

    def test_separate_external_reference_denominators_are_preserved(self) -> None:
        measurements = build_figure.load_common_measurements()
        pairs = build_figure.load_pair_measurements()
        gaps = build_figure.load_gap_recovery()
        summary = build_figure.load_summary()

        self.assertEqual(len(measurements), 120)
        self.assertEqual(len(pairs), 60)
        self.assertEqual(len(gaps), 8)
        for method in ("EasyLEED", "ViPErLEED"):
            method_rows = [row for row in measurements if row["method"] == method]
            self.assertEqual(len(method_rows), 60)
            self.assertEqual(sum(row["fit_valid"] for row in method_rows), 38)
            self.assertEqual(
                sum(
                    row["recovery_fit_valid"] for row in gaps if row["method"] == method
                ),
                4,
            )
        self.assertEqual(summary["branch_opportunities_per_method"], 60)
        self.assertEqual(
            summary["conditional_pair_frames_per_method"], {"P1": 5, "P2": 3}
        )
        self.assertFalse(summary["part_of_main_five_method_matrix"])

    def test_native_and_common_center_agreement_are_not_conflated(self) -> None:
        summary = build_figure.load_summary()
        agreement = summary["cross_method_contract_agreement_px"]

        self.assertAlmostEqual(
            agreement["native_proposal_distance_median"], 0.6256103065878225
        )
        self.assertAlmostEqual(
            agreement["common_center_distance_median"], 0.0009857897447900941
        )
        self.assertGreater(
            agreement["native_proposal_distance_median"],
            agreement["common_center_distance_median"],
        )

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
            self.assertEqual(manifest["figure"], "Figure S4")
            self.assertEqual(manifest["branch_opportunities_per_method"], 60)
            self.assertEqual(manifest["valid_common_fits_per_method"], 38)
            self.assertFalse(manifest["part_of_main_five_method_matrix"])
            self.assertFalse(manifest["claims_position_ground_truth"])


if __name__ == "__main__":
    unittest.main()
