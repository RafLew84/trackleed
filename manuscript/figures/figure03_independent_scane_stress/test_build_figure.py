#!/usr/bin/env python3
"""Frozen-data and rendering-contract tests for publication Figure 3."""

from __future__ import annotations

import unittest

import figure_contract
from figure_test_support import assert_case_rows_reproduce_aggregates


class Figure3RenderContractTest(unittest.TestCase):
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

    def test_external_reference_is_not_a_sixth_method(self) -> None:
        external = figure_contract.load_external_reference_source_data()

        self.assertFalse(external["part_of_main_five_method_matrix"])
        self.assertEqual(set(external["methods"]), {"EasyLEED", "ViPErLEED"})
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
                "external_domain_reference",
            ),
        )


if __name__ == "__main__":
    unittest.main()
