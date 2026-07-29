#!/usr/bin/env python3
"""Frozen-data contract tests for publication Figure 4."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import figure_contract
import freeze_source_data


class Figure4SourceContractTest(unittest.TestCase):
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
                "geometry_inconsistency_by_pair",
                "mechanistic_tradeoff_summary",
            ],
        )
        self.assertTrue(
            contract["representative_gap"]["selection_uses_method_outcomes"]
        )

        with tempfile.TemporaryDirectory(prefix="trackleed-figure04-contract-") as tmp:
            root = Path(tmp)
            freeze_source_data.build(root)
            figure_contract.validate_source_contract(root / "source_data")


if __name__ == "__main__":
    unittest.main()
