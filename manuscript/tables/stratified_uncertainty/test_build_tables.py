#!/usr/bin/env python3
"""Behavior tests for the stratified uncertainty-reporting tables."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_tables


class StratifiedReportingTableTest(unittest.TestCase):
    def test_count_outcome_derives_fraction_and_rate_from_same_counts(
        self,
    ) -> None:
        outcome = build_tables.CountOutcome(3, 4)

        self.assertEqual(outcome.fraction, "3/4")
        self.assertEqual(outcome.rate, 0.75)
        with self.assertRaises(ValueError):
            build_tables.CountOutcome(5, 4)

    def test_energy_scan_table_preserves_each_frozen_series_pair_case(self) -> None:
        rows = build_tables.build_energy_scan_case_table()

        self.assertEqual(len(rows), 20)
        self.assertEqual({row["scan"] for row in rows}, {"Scan A", "Scan B"})
        row = next(
            row
            for row in rows
            if row["scan"] == "Scan A"
            and row["pair"] == "pair 1"
            and row["method"] == "EasyLEED"
        )
        self.assertEqual(row["measurement_coverage"], "42/44")
        self.assertEqual(row["gap_recovery"], "5/5")
        self.assertEqual(row["geometry_inconsistency"], "0/42")
        self.assertEqual(row["conditional_pair_frames"], 15)
        self.assertAlmostEqual(
            row["conditional_median_abs_distance_residual_px"],
            1.5260859052693831,
        )

    def test_leave_one_case_out_recomputes_only_count_based_outcomes(self) -> None:
        rows = build_tables.build_leave_one_case_out_table()

        self.assertEqual(len(rows), 20)
        row = next(
            row
            for row in rows
            if row["omitted_scan"] == "Scan A"
            and row["omitted_pair"] == "pair 1"
            and row["method"] == "EasyLEED"
        )
        self.assertEqual(row["included_case_count"], 3)
        self.assertEqual(row["measurement_coverage"], "172/177")
        self.assertEqual(row["gap_recovery"], "17/19")
        self.assertEqual(row["geometry_inconsistency"], "4/180")
        self.assertNotIn("conditional_residual", row)

    def test_translation_table_uses_trials_as_the_reporting_unit(self) -> None:
        rows = build_tables.build_controlled_translation_trial_table()

        self.assertEqual(len(rows), 24)
        row = next(
            row
            for row in rows
            if row["trial"] == "initial low-temperature state"
            and row["method"] == "LocoTrack"
        )
        self.assertEqual(row["temperature_c"], 10)
        self.assertEqual(row["proposal_coverage"], "20/20")
        self.assertEqual(row["measurement_coverage"], "20/20")
        self.assertAlmostEqual(
            row["measurement_error_median_px"],
            0.0031384264221719837,
        )
        self.assertAlmostEqual(
            row["measurement_error_p95_px"],
            0.00437150211062956,
        )
        self.assertAlmostEqual(
            row["measurement_error_max_px"],
            0.005410235401619575,
        )
        self.assertEqual(row["pair_coverage"], "10/10")
        self.assertAlmostEqual(
            row["maximum_abs_delta_d_over_d_pp"],
            0.00018074382868849865,
        )

    def test_artifact_records_descriptive_policy_and_deterministic_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "output"
            manifest_path = root / "manifest.json"

            build_tables.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )

            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {
                    "controlled_translation_by_trial.csv",
                    "energy_scan_by_case.csv",
                    "energy_scan_leave_one_case_out.csv",
                    "table_s3_energy_scan_by_case.md",
                    "table_s5_controlled_translation_by_trial.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["uncertainty_policy"]["inference"],
                "descriptive; no confidence intervals or p-values",
            )
            self.assertEqual(
                manifest["uncertainty_policy"]["sensitivity_analysis"],
                (
                    "leave one frozen scan-pair case out; recompute count-based "
                    "outcomes only"
                ),
            )
            self.assertEqual(
                manifest["uncertainty_policy"]["prohibited"],
                [
                    "image-level bootstrap",
                    "binomial confidence intervals over correlated frames or gaps",
                    "reaggregation of case medians as pooled residuals",
                ],
            )
            self.assertEqual(len(manifest["outputs"]), 5)
            for markdown_name in (
                "table_s3_energy_scan_by_case.md",
                "table_s5_controlled_translation_by_trial.md",
            ):
                groups: list[list[str]] = []
                for line in (
                    output_directory / markdown_name
                ).read_text(encoding="utf-8").splitlines():
                    if line.startswith("|"):
                        if not groups or not groups[-1]:
                            groups.append([])
                        groups[-1].append(line)
                    elif groups and groups[-1]:
                        groups.append([])
                groups = [group for group in groups if group]
                self.assertTrue(groups)
                for group in groups:
                    self.assertEqual(
                        {line.count("|") for line in group},
                        {group[0].count("|")},
                    )


if __name__ == "__main__":
    unittest.main()
