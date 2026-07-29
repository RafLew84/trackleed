#!/usr/bin/env python3
"""Frozen source-contract tests for publication Figure 5."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import figure_contract
import freeze_source_data


class Figure5SourceContractTest(unittest.TestCase):
    def test_contract_defines_the_positive_control_and_separate_diagnostic(
        self,
    ) -> None:
        contract = figure_contract.load_contract()

        self.assertEqual(contract["status"], "source_contract_frozen")
        self.assertEqual(
            contract["primary_scope"]["scientific_series_label"],
            "Cu(111) thermal cycle",
        )
        self.assertEqual(contract["primary_scope"]["frame_count"], 701)
        self.assertEqual(contract["primary_scope"]["energy_ev"], 56.0)
        self.assertEqual(
            contract["primary_scope"]["methods"],
            [
                "locotrack_base_offline",
                "ostrack_vitb_256_mae_ce",
                "dam4sam_sam21pp_l",
                "lucas_kanade_skip_anchor",
                "easyleed_kalman",
            ],
        )
        self.assertEqual(
            contract["primary_scope"]["pairs"], ["1-6", "3-5"]
        )
        self.assertEqual(
            contract["run_contract_source"],
            "source_data/five_method_run_contract.json",
        )
        self.assertEqual(
            contract["endpoint_definition"]["reference"]["frame_start"], 0
        )
        self.assertEqual(
            contract["endpoint_definition"]["reference"]["frame_end"], 9
        )
        self.assertEqual(
            contract["endpoint_definition"]["reference"][
                "minimum_valid_distances"
            ],
            8,
        )
        self.assertFalse(
            contract["endpoint_definition"]["pattern_center_required"]
        )
        self.assertEqual(
            contract["panel_order"],
            [
                "thermal_cycle_landmarks",
                "pair_1_6_endpoint",
                "pair_3_5_endpoint",
                "between_method_endpoint_spread",
                "separate_pair_2_4_fit_missingness",
            ],
        )
        diagnostic = contract["separate_diagnostic"]
        self.assertEqual(diagnostic["pair"], "2-4")
        self.assertFalse(diagnostic["inside_five_method_run"])
        self.assertEqual(
            diagnostic["tracking_source"],
            "earlier saved YOLO proposal trajectory",
        )
        self.assertEqual(diagnostic["rejected_fit_count"], 24)
        self.assertEqual(diagnostic["failed_gate"], "amplitude")

        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure05-contract-"
        ) as tmp:
            root = Path(tmp)
            freeze_source_data.build(root)
            figure_contract.validate_source_contract(root / "source_data")

    def test_contract_rejects_mixing_the_saved_yolo_diagnostic_into_the_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="trackleed-figure05-separation-"
        ) as tmp:
            root = Path(tmp)
            freeze_source_data.build(root)
            path = (
                root
                / "source_data"
                / "pair_2_4_missingness_diagnostic.csv"
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[0]["inside_five_method_run"] = "true"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                ValueError, "mixed into five-method run"
            ):
                figure_contract.validate_source_contract(
                    root / "source_data"
                )


if __name__ == "__main__":
    unittest.main()
