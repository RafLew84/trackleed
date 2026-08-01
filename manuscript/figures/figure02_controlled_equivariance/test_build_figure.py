#!/usr/bin/env python3
"""Contract tests for publication Figure 2."""

from __future__ import annotations

import json
import unittest

import build_figure
import figure_contract


class Figure2SourceContractTest(unittest.TestCase):
    def test_main_figure_contains_only_controlled_translation_evidence(self) -> None:
        manifest = json.loads(
            (build_figure.FIGURE_ROOT / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            manifest["panel_order"],
            [
                "translation_path",
                "native_vs_refined_displacement",
                "measurement_coverage",
                "pair_invariance",
            ],
        )
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("energy-scan", serialized)
        self.assertNotIn("sensitivity", serialized)

    def test_translation_path_is_closed_and_has_six_steps(self) -> None:
        points = figure_contract.load_translation_path(
            figure_contract.CONTROLLED_TRANSLATION_CONTRACT
        )

        self.assertEqual(len(points), 6)
        self.assertEqual(points[0]["dx_px"], 0.0)
        self.assertEqual(points[0]["dy_px"], 0.0)
        self.assertEqual(points[-1]["dx_px"], 0.0)
        self.assertEqual(points[-1]["dy_px"], 0.0)

    def test_method_metrics_preserve_all_five_denominators(self) -> None:
        rows = figure_contract.load_method_metrics(
            figure_contract.CONTROLLED_TRANSLATION_SUMMARY
        )

        self.assertEqual(
            [row["method"] for row in rows],
            list(figure_contract.METHOD_ORDER),
        )
        self.assertEqual(
            [row["valid_gaussian_measurements"] for row in rows],
            [80, 80, 80, 76, 74],
        )
        self.assertTrue(
            all(row["evaluation_spot_opportunities"] == 80 for row in rows)
        )
        self.assertEqual(
            [row["valid_pair_measurements"] for row in rows],
            [40, 40, 40, 36, 34],
        )
        self.assertTrue(all(row["pair_opportunities"] == 40 for row in rows))

if __name__ == "__main__":
    unittest.main()
