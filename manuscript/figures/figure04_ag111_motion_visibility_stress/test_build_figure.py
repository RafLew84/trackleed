#!/usr/bin/env python3
"""Frozen-data and rendering-contract tests for publication Figure 4."""

from __future__ import annotations

import json
import statistics
import unittest

import figure_contract
from figure_test_support import assert_case_rows_reproduce_aggregates
from figure_render import summarize_representative_trajectory


class Figure4RenderContractTest(unittest.TestCase):
    def test_manifest_records_non_overplotted_panel_a_encoding(self) -> None:
        manifest = json.loads(figure_contract.MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["rendering"]["representative_trajectory_encoding"],
            {
                "point": (
                    "cross-method median of available valid Gaussian pair measurements"
                ),
                "whisker": "cross-method minimum-to-maximum",
                "grouping": "image frame; repeated energies retained",
                "method_specific_markers": False,
                "maximum_cross_method_range_px": 0.06604708906252199,
            },
        )

    def test_representative_trajectory_uses_framewise_median_and_min_max(
        self,
    ) -> None:
        rows = figure_contract.load_representative_pair_source_data()

        summary = summarize_representative_trajectory(rows)

        self.assertEqual(len(summary.frames), 21)
        self.assertEqual(summary.frames[0], 5)
        self.assertEqual(summary.valid_method_counts[0], 2)
        frame = 13
        index = summary.frames.index(frame)
        values = [
            float(row["pair_distance_px"]) for row in rows if row["frame"] == frame
        ]
        self.assertEqual(summary.valid_method_counts[index], 5)
        self.assertAlmostEqual(
            summary.median_distances_px[index],
            statistics.median(values),
        )
        self.assertAlmostEqual(
            summary.minimum_distances_px[index],
            min(values),
        )
        self.assertAlmostEqual(
            summary.maximum_distances_px[index],
            max(values),
        )
        self.assertAlmostEqual(
            summary.maximum_range_px,
            0.06604708906252199,
        )

    def test_headline_metrics_preserve_counts_before_conditional_error(self) -> None:
        rows = figure_contract.load_aggregate_source_data()

        self.assertEqual(
            [row["method"] for row in rows],
            list(figure_contract.METHOD_ORDER),
        )
        self.assertEqual(
            [row["branch_consistent_measurements"] for row in rows],
            [217, 216, 216, 188, 214],
        )
        self.assertEqual(
            [row["recovered_gaps"] for row in rows],
            [23, 21, 23, 17, 22],
        )
        self.assertEqual(
            [row["jointly_consistent_pair_frames"] for row in rows],
            [91, 90, 91, 68, 84],
        )
        self.assertTrue(
            all(row["median_abs_distance_residual_px"] is not None for row in rows)
        )

    def test_geometry_inconsistency_uses_method_specific_accepted_denominator(
        self,
    ) -> None:
        rows = figure_contract.load_aggregate_source_data()

        self.assertEqual(
            [
                (
                    row["geometry_inconsistent_accepted_fits"],
                    row["accepted_measurements_all_frames"],
                )
                for row in rows
            ],
            [(36, 272), (34, 269), (36, 272), (14, 206), (4, 222)],
        )

    def test_frozen_case_rows_reproduce_headline_counts(self) -> None:
        aggregate = figure_contract.load_aggregate_source_data()
        cases = figure_contract.load_case_source_data()

        assert_case_rows_reproduce_aggregates(self, aggregate, cases)

    def test_representative_visibility_data_contains_natural_gaps(self) -> None:
        rows = figure_contract.load_visibility_source_data()

        self.assertEqual(len(rows), 70)
        self.assertEqual({row["branch"] for row in rows}, {"B1", "B2"})
        self.assertIn("iv_visibility_gap", {row["status"] for row in rows})
        self.assertIn("apparatus_absence", {row["status"] for row in rows})
        self.assertIn("beam_onset_or_termination", {row["status"] for row in rows})

    def test_external_reference_is_absent_from_main_figure_contract(self) -> None:
        contract = figure_contract.load_contract()

        self.assertNotIn("external_reference", contract)
        self.assertNotIn("external_domain_reference", figure_contract.PANEL_ORDER)
        self.assertFalse(
            (
                figure_contract.SOURCE_DATA_DIRECTORY
                / "external_reference_summary.json"
            ).exists()
        )
        self.assertNotIn("viperleed", figure_contract.METHOD_ORDER)

    def test_panel_order_places_coverage_and_recovery_before_residual(self) -> None:
        self.assertEqual(
            figure_contract.PANEL_ORDER,
            (
                "representative_trajectory",
                "measurement_coverage",
                "gap_recovery",
                "geometry_inconsistency",
                "conditional_pair_residual",
            ),
        )


if __name__ == "__main__":
    unittest.main()
