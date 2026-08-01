#!/usr/bin/env python3
"""Source-contract tests for publication Figure 4."""

from __future__ import annotations

import json
import unittest

import freeze_source_data
from figure_test_support import assert_case_rows_reproduce_aggregates


class Figure4SourceFreezeTest(unittest.TestCase):
    def test_aggregate_keeps_common_denominators_and_method_order(self) -> None:
        rows = freeze_source_data.load_aggregate_results()

        self.assertEqual(
            [row["method"] for row in rows],
            list(freeze_source_data.METHOD_ORDER),
        )
        self.assertTrue(all(row["visible_opportunities"] == 221 for row in rows))
        self.assertTrue(all(row["evaluated_gaps"] == 24 for row in rows))
        self.assertEqual(
            [row["branch_consistent_measurements"] for row in rows],
            [217, 216, 216, 188, 214],
        )
        self.assertEqual(
            [row["recovered_gaps"] for row in rows],
            [23, 21, 23, 17, 22],
        )

    def test_case_rows_reproduce_every_aggregate_numerator(self) -> None:
        aggregate = freeze_source_data.load_aggregate_results()
        cases = freeze_source_data.load_case_results()

        assert_case_rows_reproduce_aggregates(self, aggregate, cases)

    def test_representative_case_uses_predeclared_physics_rule(self) -> None:
        summary = json.loads(
            freeze_source_data.INDEPENDENT_SUMMARY.read_text(encoding="utf-8")
        )
        selected = freeze_source_data.select_representative_case(summary)

        self.assertEqual(selected["scientific_series_label"], "Scan A")
        self.assertEqual(selected["scientific_pair_label"], "pair 2")
        self.assertEqual(selected["archive_series_id"], "26a21")
        self.assertEqual(selected["archive_pair_id"], "pair2")
        self.assertEqual(selected["frame_count"], 35)
        self.assertEqual(selected["reference_energy_ev"], 188.5)
        self.assertEqual(selected["iv_gap_event_count"], 7)
        self.assertTrue(selected["decision_checks"]["coherent_pairwise_energy_law"])
        self.assertTrue(selected["decision_checks"]["natural_iv_gap_available"])

    def test_representative_rows_include_all_five_methods(self) -> None:
        summary = json.loads(
            freeze_source_data.INDEPENDENT_SUMMARY.read_text(encoding="utf-8")
        )
        selected = freeze_source_data.select_representative_case(summary)
        rows = freeze_source_data.build_representative_pair_rows(selected)
        counts = {
            method: sum(row["method"] == method for row in rows)
            for method in freeze_source_data.METHOD_ORDER
        }

        self.assertEqual(
            counts,
            {
                "locotrack": 19,
                "ostrack": 20,
                "dam4sam": 20,
                "lk": 14,
                "easyleed": 18,
            },
        )
        self.assertTrue(all(row["scientific_series_label"] == "Scan A" for row in rows))


if __name__ == "__main__":
    unittest.main()
