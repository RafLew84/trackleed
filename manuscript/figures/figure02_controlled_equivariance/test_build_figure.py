#!/usr/bin/env python3
"""Contract tests for publication Figure 2."""

from __future__ import annotations

import unittest

import build_figure


class Figure2SourceContractTest(unittest.TestCase):
    def test_translation_path_is_closed_and_has_six_steps(self) -> None:
        points = build_figure.load_translation_path(
            build_figure.CONTROLLED_TRANSLATION_CONTRACT
        )

        self.assertEqual(len(points), 6)
        self.assertEqual(points[0]["dx_px"], 0.0)
        self.assertEqual(points[0]["dy_px"], 0.0)
        self.assertEqual(points[-1]["dx_px"], 0.0)
        self.assertEqual(points[-1]["dy_px"], 0.0)

    def test_method_metrics_preserve_all_five_denominators(self) -> None:
        rows = build_figure.load_method_metrics(
            build_figure.CONTROLLED_TRANSLATION_SUMMARY
        )

        self.assertEqual(
            [row["method"] for row in rows],
            list(build_figure.METHOD_ORDER),
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

    def test_energy_geometry_retains_missing_pair_measurements(self) -> None:
        rows = build_figure.build_energy_geometry_rows(
            build_figure.ENERGY_SCAN_BRANCH_MEASUREMENTS,
            build_figure.ENERGY_SCAN_PAIR_MEASUREMENTS,
        )

        self.assertEqual(len(rows), 69)
        self.assertEqual(sum(row["pair_valid"] for row in rows), 36)
        self.assertEqual(sum(not row["pair_valid"] for row in rows), 33)
        reference = [row for row in rows if row["energy_ev"] == 187.9]
        self.assertEqual(len(reference), 1)
        self.assertTrue(reference[0]["pair_valid"])
        self.assertAlmostEqual(reference[0]["midpoint_drift_px"], 0.0021737785)


if __name__ == "__main__":
    unittest.main()
