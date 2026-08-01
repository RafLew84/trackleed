#!/usr/bin/env python3
"""Frozen-data contract tests for publication Figure 5."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import figure_contract
import freeze_source_data


class Figure5SourceContractTest(unittest.TestCase):
    def test_contract_defines_control_scope_and_validates_built_sources(
        self,
    ) -> None:
        contract = figure_contract.load_contract()

        self.assertEqual(contract["status"], "source_contract_frozen")
        self.assertEqual(
            contract["controlled_family"]["methods"],
            ["sam2", "samurai", "dam4sam"],
        )
        self.assertEqual(contract["controlled_family"]["series_count"], 1)
        self.assertEqual(contract["controlled_family"]["pair_count"], 2)
        self.assertEqual(
            contract["controlled_family"]["visible_opportunities_per_method"],
            182,
        )
        self.assertEqual(
            contract["controlled_family"]["evaluated_gaps_per_method"],
            13,
        )
        self.assertTrue(
            contract["controlled_family"]["identical_checkpoint_weights"]
        )
        self.assertEqual(
            contract["panel_order"],
            [
                "representative_visibility_gap",
                "coverage_by_pair",
                "gap_recovery_by_pair",
                "failure_cost_and_conditional_residual_by_pair",
                "mechanistic_tradeoff_summary",
            ],
        )
        panel_metrics = contract["panel_metrics"]
        self.assertIn(
            "zero_delay_recoveries",
            panel_metrics["gap_recovery_by_pair"],
        )
        self.assertIn(
            "geometry_inconsistent_accepted_fits",
            panel_metrics[
                "failure_cost_and_conditional_residual_by_pair"
            ],
        )
        self.assertIn(
            "conditional_p95_abs_distance_residual_px",
            panel_metrics[
                "failure_cost_and_conditional_residual_by_pair"
            ],
        )
        self.assertEqual(
            contract["pair_definition"]["source_file"],
            "source_data/pair_seeds.csv",
        )
        self.assertEqual(
            contract["pair_definition"][
                "minimum_cross_pair_seed_separation_px"
            ],
            48,
        )
        self.assertTrue(
            contract["representative_gap"]["selection_uses_method_outcomes"]
        )

        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-contract-") as tmp:
            root = Path(tmp)
            freeze_source_data.build(root)
            figure_contract.validate_source_contract(root / "source_data")


if __name__ == "__main__":
    unittest.main()
