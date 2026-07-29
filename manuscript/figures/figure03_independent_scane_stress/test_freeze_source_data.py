#!/usr/bin/env python3
"""Source-contract tests for publication Figure 3."""

from __future__ import annotations

import json
import unittest

import freeze_source_data


class Figure3SourceFreezeTest(unittest.TestCase):
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

        for aggregate_row in aggregate:
            method_rows = [
                row
                for row in cases
                if row["method"] == aggregate_row["method"]
            ]
            self.assertEqual(len(method_rows), 4)
            for field in (
                "visible_opportunities",
                "branch_consistent_measurements",
                "evaluated_gaps",
                "recovered_gaps",
                "accepted_measurements_all_frames",
                "geometry_inconsistent_accepted_fits",
                "jointly_consistent_pair_frames",
            ):
                self.assertEqual(
                    sum(row[field] for row in method_rows),
                    aggregate_row[field],
                )

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
        self.assertTrue(
            selected["decision_checks"]["coherent_pairwise_energy_law"]
        )
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
        self.assertTrue(
            all(row["scientific_series_label"] == "Scan A" for row in rows)
        )

    def test_external_reference_retains_its_separate_contract(self) -> None:
        external = freeze_source_data.build_external_reference_summary()

        self.assertEqual(external["frame_count"], 15)
        self.assertEqual(external["branch_opportunities_per_method"], 60)
        self.assertEqual(external["valid_common_fits_per_method"], 38)
        self.assertEqual(external["branch_gap_checks"], "4/4")
        self.assertEqual(
            external["initialization_contracts"],
            {
                "EasyLEED": "one Kalman tracker per frozen spot",
                "ViPErLEED": "registered 18-beam whole-pattern trajectory",
            },
        )
        self.assertFalse(external["part_of_main_five_method_matrix"])


if __name__ == "__main__":
    unittest.main()
