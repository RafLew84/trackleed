#!/usr/bin/env python3
"""Rendering-contract tests for publication Figure 5."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import build_figure
from figure_contract import load_cycle_landmarks
from figure_render import _reader_label


class Figure5RenderContractTest(unittest.TestCase):
    def test_renderer_uses_frozen_phase_and_temperature_landmark_labels(
        self,
    ) -> None:
        landmarks = load_cycle_landmarks()

        self.assertEqual(
            _reader_label(landmarks, "reference"),
            "Reference (10-16 °C)",
        )
        self.assertEqual(
            _reader_label(landmarks, "plateau"),
            "800 °C plateau",
        )
        self.assertEqual(
            _reader_label(landmarks, "post_plateau_minimum"),
            "Post-plateau minimum (797 °C)",
        )
        self.assertEqual(
            _reader_label(landmarks, "final"),
            "Final cooled state (92 °C)",
        )

    def test_build_creates_publication_png_pdf_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-render-") as tmp:
            root = Path(tmp)
            outputs = build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )

            self.assertEqual(set(outputs), {"png", "pdf"})
            with Image.open(outputs["png"]) as image:
                self.assertEqual(image.size, (2880, 3600))
                self.assertEqual(image.mode, "RGBA")
            self.assertTrue(outputs["pdf"].read_bytes().startswith(b"%PDF"))

            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["figure"], "Figure 5")
            self.assertEqual(
                manifest["rendering"]["panel_order"],
                [
                    "thermal_cycle_landmarks",
                    "pair_1_6_endpoint",
                    "pair_3_5_endpoint",
                    "between_method_endpoint_spread",
                    "separate_pair_2_4_fit_missingness",
                ],
            )

    def test_manifest_audits_primary_run_spread_and_separate_diagnostic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-audit-") as tmp:
            root = Path(tmp)
            build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            rendering = manifest["rendering"]

            self.assertEqual(
                rendering["reader_facing_x_coordinate"],
                "thermal-cycle progress percent",
            )
            self.assertTrue(
                all(
                    "frame" not in label.lower()
                    and "25w28" not in label.lower()
                    for label in rendering["reader_facing_landmarks"]
                )
            )
            self.assertEqual(
                rendering["primary_run"]["method_count"], 5
            )
            self.assertEqual(
                rendering["primary_run"]["frame_count"], 701
            )
            self.assertEqual(
                set(
                    rendering["primary_run"][
                        "native_proposals_per_method"
                    ].values()
                ),
                {2804},
            )
            self.assertEqual(
                set(
                    rendering["primary_run"][
                        "valid_gaussian_fits_per_method"
                    ].values()
                ),
                {2804},
            )
            self.assertEqual(
                rendering["between_method_spread"][
                    "complete_sequence_maximum_percentage_points"
                ],
                {
                    "1-6": 0.0017096162773233559,
                    "3-5": 0.002236605543060044,
                },
            )
            diagnostic = rendering["separate_pair_2_4_diagnostic"]
            self.assertFalse(diagnostic["inside_five_method_run"])
            self.assertEqual(diagnostic["context_count"], 26)
            self.assertEqual(
                diagnostic["tracking_roi_present_count"], 26
            )
            self.assertEqual(diagnostic["rejected_fit_count"], 24)
            self.assertEqual(
                diagnostic["amplitude_only_rejection_count"], 24
            )


if __name__ == "__main__":
    unittest.main()
