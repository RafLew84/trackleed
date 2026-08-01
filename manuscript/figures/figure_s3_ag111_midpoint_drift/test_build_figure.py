#!/usr/bin/env python3
"""Contract tests for supplementary Figure S3."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import build_figure


class FigureS3ContractTest(unittest.TestCase):
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

    def test_source_package_covers_all_four_frozen_cases(self) -> None:
        cases = build_figure.load_case_contract()
        trajectories = build_figure.load_pair_measurements()
        visibility = build_figure.load_visibility_regimes()

        self.assertEqual(
            [(row["series"], row["pair"]) for row in cases],
            [
                ("Scan A", "pair 1"),
                ("Scan A", "pair 2"),
                ("Scan B", "pair 1"),
                ("Scan B", "pair 2"),
            ],
        )
        self.assertEqual(
            [row["valid_pair_frames"] for row in cases],
            [17, 20, 25, 27],
        )
        self.assertEqual(len(trajectories), 89)
        self.assertEqual(len(visibility), 324)
        self.assertEqual(
            {row["status"] for row in visibility},
            {
                "visible_fit",
                "iv_visibility_gap",
                "beam_onset_or_termination",
                "apparatus_absence",
            },
        )

    def test_midpoint_drift_is_separate_from_pair_distance_residual(self) -> None:
        rows = build_figure.load_pair_measurements()

        self.assertTrue(all(row["midpoint_drift_px"] >= 0 for row in rows))
        self.assertTrue(
            any(
                abs(row["distance_residual_px"] - row["midpoint_drift_px"]) > 1.0
                for row in rows
            )
        )

    def test_build_is_deterministic_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            first_bytes = {name: path.read_bytes() for name, path in first.items()}
            second = build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            second_bytes = {name: path.read_bytes() for name, path in second.items()}

            self.assertEqual(first_bytes, second_bytes)
            with Image.open(first["png"]) as image:
                self.assertEqual(image.mode, "RGBA")
            self.assertTrue(first["pdf"].read_bytes().startswith(b"%PDF"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure"], "Figure S3")
            self.assertEqual(manifest["case_count"], 4)
            self.assertEqual(manifest["valid_pair_frame_count"], 89)
            self.assertEqual(manifest["visibility_branch_frame_count"], 324)
            self.assertFalse(manifest["claims_absolute_position_accuracy"])


if __name__ == "__main__":
    unittest.main()
