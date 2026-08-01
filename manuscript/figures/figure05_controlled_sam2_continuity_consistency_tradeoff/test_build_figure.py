#!/usr/bin/env python3
"""Rendering-contract tests for publication Figure 5."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import build_figure


class Figure5RenderContractTest(unittest.TestCase):
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
                outputs["png"].name,
                "figure05_controlled_sam2_continuity_consistency_tradeoff.png",
            )
            self.assertEqual(
                manifest["rendering"]["panel_order"],
                [
                    "representative_visibility_gap",
                    "coverage_by_pair",
                    "gap_recovery_by_pair",
                    "failure_cost_and_conditional_residual_by_pair",
                    "mechanistic_tradeoff_summary",
                ],
            )

    def test_manifest_audits_representative_panel_states(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-panel-a-") as tmp:
            root = Path(tmp)
            build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                manifest["rendering"]["representative_crop_normalization"],
                {"mode": "shared_linear", "vmin": 33.0, "vmax": 48.0},
            )
            self.assertEqual(
                manifest["rendering"]["representative_panel_states"],
                {
                    "native_proposals": 7,
                    "accepted_gaussian_centers": 5,
                    "rejected_gaussian_centers": 2,
                    "no_proposal_states": 5,
                    "outcome_selected_for_explanation": True,
                },
            )

    def test_reader_facing_language_names_branch_consistent_recovery(self) -> None:
        artifact_root = Path(__file__).resolve().parent
        readme = (artifact_root / "README.md").read_text(encoding="utf-8")
        renderer = (artifact_root / "figure_render.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("post-gap branch-consistent recovery", readme.lower())
        self.assertIn("Post-gap branch-consistent recovery", renderer)
        self.assertNotIn("Post-gap recovery [", renderer)


if __name__ == "__main__":
    unittest.main()
