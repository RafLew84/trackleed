#!/usr/bin/env python3
"""Acceptance tests for the separate Cu(111) pair 2-4 diagnostic."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import build_figure


class FigureS6Cu111MissingnessTest(unittest.TestCase):
    def test_contract_keeps_the_saved_proposal_diagnostic_separate(self) -> None:
        rows = build_figure.load_and_validate_rows()
        rejected = [row for row in rows if not row["refit_accepted"]]

        self.assertEqual(len(rows), 26)
        self.assertEqual(len(rejected), 24)
        self.assertTrue(all(row["tracking_proposal_present"] for row in rows))
        self.assertTrue(all(not row["inside_five_method_run"] for row in rows))
        self.assertTrue(
            all(
                row["r_squared_gate"]
                and not row["amplitude_gate"]
                and row["center_gate"]
                and row["sigma_gate"]
                and row["rejection_reason"] == "amplitude_below_or_outside_threshold"
                for row in rejected
            )
        )

    def test_contract_fails_closed_if_diagnostic_is_mixed_into_main_run(
        self,
    ) -> None:
        source_path = build_figure.SOURCE_DATA_PATH
        with source_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["inside_five_method_run"] = "true"

        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure-s6-mutation-"
        ) as temporary:
            changed = Path(temporary) / source_path.name
            with changed.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "outside the five-method"):
                build_figure.load_and_validate_rows(changed)

    def test_build_is_deterministic_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure-s6-") as temporary:
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
                self.assertEqual(image.size, (2800, 1800))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure"], "Figure S6")
            self.assertEqual(manifest["tracking_proposal_present"], "26/26")
            self.assertEqual(manifest["amplitude_only_rejections"], "24/24")
            self.assertFalse(manifest["inside_five_method_run"])


if __name__ == "__main__":
    unittest.main()
