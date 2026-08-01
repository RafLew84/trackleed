#!/usr/bin/env python3
"""Acceptance tests for the final Cu(111) main Figure 3."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import build_figure
from figure_contract import PANEL_ORDER, load_primary_records


class Figure3Cu111PositiveControlTest(unittest.TestCase):
    def test_primary_source_contract_preserves_complete_five_method_run(
        self,
    ) -> None:
        records = load_primary_records()

        self.assertEqual(len(records.endpoints), 5 * 2 * 701)
        self.assertEqual(records.run["frame_count"], 701)
        self.assertEqual(
            set(records.run["native_proposals_per_method"].values()),
            {2804},
        )
        self.assertEqual(
            set(records.run["valid_gaussian_fits_per_method"].values()),
            {2804},
        )
        self.assertTrue(
            all(summary["valid_frame_count"] == 701 for summary in records.summaries)
        )

    def test_main_figure_excludes_the_pair_2_4_diagnostic(self) -> None:
        self.assertEqual(
            PANEL_ORDER,
            (
                "thermal_cycle_landmarks",
                "pair_1_6_endpoint",
                "pair_3_5_endpoint",
                "between_method_endpoint_spread",
            ),
        )
        figure_root = Path(__file__).resolve().parent
        self.assertFalse(
            (figure_root / "source_data/pair_2_4_missingness_diagnostic.csv").exists()
        )

    def test_build_is_deterministic_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure03-cu111-"
        ) as temporary:
            root = Path(temporary)
            outputs = build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            first_hashes = {
                name: build_figure.sha256_file(path) for name, path in outputs.items()
            }
            outputs = build_figure.build(
                output_directory=root / "output",
                manifest_path=root / "manifest.json",
            )
            second_hashes = {
                name: build_figure.sha256_file(path) for name, path in outputs.items()
            }

            self.assertEqual(first_hashes, second_hashes)
            with Image.open(outputs["png"]) as image:
                self.assertEqual(image.size, (2880, 3200))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure"], "Figure 3")
            self.assertEqual(
                manifest["caption_sha256"],
                build_figure.sha256_file(Path(__file__).resolve().parent / "README.md"),
            )
            self.assertNotIn(
                "pair_2_4_missingness_diagnostic.csv",
                "\n".join(manifest["source_data_sha256"]),
            )
            self.assertEqual(manifest["rendering"]["panel_order"], list(PANEL_ORDER))
            self.assertEqual(
                manifest["rendering"]["anchor_annotation"],
                "2804/2804 measurements per method; 701/701 endpoints per pair",
            )
            self.assertEqual(
                manifest["anchors"],
                {
                    "valid_measurements_per_method": "2804/2804",
                    "valid_endpoints_per_method_pair": "701/701",
                },
            )
            caption = (
                (Path(__file__).resolve().parent / "README.md")
                .read_text(encoding="utf-8")
                .split("## Caption", 1)[1]
            )
            self.assertIn("2804/2804", caption)
            self.assertIn("701/701", caption)


if __name__ == "__main__":
    unittest.main()
