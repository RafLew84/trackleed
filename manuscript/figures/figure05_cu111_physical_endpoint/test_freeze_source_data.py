#!/usr/bin/env python3
"""Source-package tests for publication Figure 5."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import freeze_source_data


class Figure5SourceFreezeTest(unittest.TestCase):
    def test_run_contract_establishes_initialization_and_denominators(
        self,
    ) -> None:
        contract = freeze_source_data.load_five_method_run_contract()

        self.assertEqual(contract["status"], "completed")
        self.assertEqual(contract["frame_count"], 701)
        self.assertEqual(contract["spot_ids"], [1, 3, 5, 6])
        self.assertEqual(contract["pairs"], ["1-6", "3-5"])
        self.assertEqual(contract["seed_frame_order"], 0)
        self.assertTrue(contract["single_forward_pass"])
        self.assertTrue(contract["correction_and_reinitialization_forbidden"])
        self.assertEqual(contract["manual_annotations"], 0)
        self.assertEqual(
            contract["native_proposals_per_method"],
            {
                method: 2804
                for method in freeze_source_data.METHOD_ORDER
            },
        )
        self.assertEqual(
            contract["valid_gaussian_fits_per_method"],
            {
                method: 2804
                for method in freeze_source_data.METHOD_ORDER
            },
        )
        self.assertTrue(contract["manifest_hash_chain_valid"])

    def test_primary_timeseries_contains_only_the_frozen_five_method_run(
        self,
    ) -> None:
        rows = freeze_source_data.load_pair_endpoint_timeseries()

        self.assertEqual(len(rows), 5 * 2 * 701)
        self.assertEqual(
            {row["method"] for row in rows},
            {
                "locotrack_base_offline",
                "ostrack_vitb_256_mae_ce",
                "dam4sam_sam21pp_l",
                "lucas_kanade_skip_anchor",
                "easyleed_kalman",
            },
        )
        self.assertEqual({row["pair"] for row in rows}, {"1-6", "3-5"})
        self.assertTrue(all(row["pair_valid"] for row in rows))
        self.assertTrue(
            all(row["reference_valid_frame_count"] == 10 for row in rows)
        )
        self.assertEqual(
            {
                (row["method"], row["pair"], row["frame_order"])
                for row in rows
            },
            {
                (method, pair, frame)
                for method in {
                    "locotrack_base_offline",
                    "ostrack_vitb_256_mae_ce",
                    "dam4sam_sam21pp_l",
                    "lucas_kanade_skip_anchor",
                    "easyleed_kalman",
                }
                for pair in {"1-6", "3-5"}
                for frame in range(701)
            },
        )

    def test_endpoint_summaries_use_the_frozen_thermal_landmarks(self) -> None:
        rows = freeze_source_data.build_endpoint_summaries(
            freeze_source_data.load_pair_endpoint_timeseries()
        )

        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["valid_frame_count"] == 701 for row in rows))
        self.assertTrue(
            all(row["reference_valid_frame_count"] == 10 for row in rows)
        )
        self.assertTrue(
            all(row["minimum_frame_order"] == 378 for row in rows)
        )
        by_pair = {
            pair: [row for row in rows if row["pair"] == pair]
            for pair in ("1-6", "3-5")
        }
        self.assertAlmostEqual(
            min(row["plateau_median_percent"] for row in by_pair["1-6"]),
            -0.6866755868635943,
        )
        self.assertAlmostEqual(
            max(row["plateau_median_percent"] for row in by_pair["1-6"]),
            -0.6862444332467299,
        )
        self.assertAlmostEqual(
            min(row["final_percent"] for row in by_pair["3-5"]),
            -0.15247760709317104,
        )
        self.assertAlmostEqual(
            max(row["final_percent"] for row in by_pair["3-5"]),
            -0.15233848352844692,
        )

    def test_between_method_spread_preserves_displayed_and_sequence_scopes(
        self,
    ) -> None:
        timeseries = freeze_source_data.load_pair_endpoint_timeseries()
        rows = freeze_source_data.build_between_method_spread(timeseries)
        by_key = {(row["pair"], row["summary_point"]): row for row in rows}

        self.assertEqual(
            set(by_key),
            {
                (pair, point)
                for pair in ("1-6", "3-5")
                for point in (
                    "plateau",
                    "post_plateau_minimum",
                    "final",
                    "entire_sequence_maximum",
                )
            },
        )
        self.assertTrue(
            all(
                by_key[(pair, point)]["display_in_panel_d"]
                for pair in ("1-6", "3-5")
                for point in ("plateau", "post_plateau_minimum", "final")
            )
        )
        self.assertFalse(
            by_key[("1-6", "entire_sequence_maximum")][
                "display_in_panel_d"
            ]
        )
        self.assertAlmostEqual(
            by_key[("1-6", "entire_sequence_maximum")][
                "spread_percentage_points"
            ],
            0.0017096162773233559,
        )
        self.assertEqual(
            by_key[("1-6", "entire_sequence_maximum")]["frame_order"],
            482,
        )
        self.assertAlmostEqual(
            by_key[("3-5", "entire_sequence_maximum")][
                "spread_percentage_points"
            ],
            0.002236605543060044,
        )
        self.assertEqual(
            by_key[("3-5", "entire_sequence_maximum")]["frame_order"],
            297,
        )

    def test_cycle_landmarks_separate_reader_labels_from_frame_metadata(
        self,
    ) -> None:
        profile = freeze_source_data.load_temperature_profile()
        summaries = freeze_source_data.build_endpoint_summaries(
            freeze_source_data.load_pair_endpoint_timeseries()
        )
        landmarks = freeze_source_data.build_cycle_landmarks(
            profile, summaries
        )
        by_name = {row["landmark"]: row for row in landmarks}

        self.assertEqual(len(profile), 701)
        self.assertEqual({row["energy_ev"] for row in profile}, {56.0})
        self.assertEqual(
            [row["landmark"] for row in landmarks],
            [
                "reference",
                "plateau",
                "post_plateau_minimum",
                "final",
            ],
        )
        self.assertEqual(
            (by_name["reference"]["frame_start"], by_name["reference"]["frame_end"]),
            (0, 9),
        )
        self.assertEqual(
            (
                by_name["plateau"]["frame_start"],
                by_name["plateau"]["frame_end"],
                by_name["plateau"]["temperature_min_c"],
            ),
            (372, 376, 800.0),
        )
        self.assertEqual(
            (
                by_name["post_plateau_minimum"]["frame_start"],
                by_name["post_plateau_minimum"]["temperature_min_c"],
            ),
            (378, 797.0),
        )
        self.assertTrue(
            by_name["post_plateau_minimum"]["outcome_selected"]
        )
        self.assertFalse(by_name["plateau"]["outcome_selected"])
        self.assertTrue(
            all("frame" not in row["reader_label"].lower() for row in landmarks)
        )
        self.assertTrue(
            all(
                "25w28" not in row["reader_label"].lower()
                for row in landmarks
            )
        )

    def test_pair_2_4_is_an_amplitude_only_saved_yolo_diagnostic(
        self,
    ) -> None:
        rows = freeze_source_data.load_pair_2_4_missingness_diagnostic()
        rejected = [row for row in rows if not row["refit_accepted"]]

        self.assertEqual(len(rows), 26)
        self.assertEqual(len(rejected), 24)
        self.assertEqual({row["pair"] for row in rows}, {"2-4"})
        self.assertEqual(
            {row["tracking_source"] for row in rows},
            {"earlier saved YOLO proposal trajectory"},
        )
        self.assertTrue(
            all(not row["inside_five_method_run"] for row in rows)
        )
        self.assertTrue(all(row["tracking_roi_present"] for row in rows))
        self.assertTrue(all(row["r_squared_gate"] for row in rejected))
        self.assertTrue(all(not row["amplitude_gate"] for row in rejected))
        self.assertTrue(all(row["center_gate"] for row in rejected))
        self.assertTrue(all(row["sigma_gate"] for row in rejected))
        self.assertTrue(
            all(
                row["rejection_reason"]
                == "amplitude_below_or_outside_threshold"
                for row in rejected
            )
        )
        self.assertEqual(
            {row["amplitude_threshold"] for row in rows}, {0.05}
        )
        self.assertEqual(
            [row["frame_id"] for row in rejected],
            list(range(357, 381)),
        )

    def test_build_creates_a_self_contained_reader_neutral_package(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-test-") as tmp:
            root = Path(tmp)
            freeze_source_data.build(root)
            source = root / "source_data"

            self.assertEqual(
                {
                    path.relative_to(source).as_posix()
                    for path in source.rglob("*")
                    if path.is_file()
                },
                {
                    "five_method_run_contract.json",
                    "temperature_profile.csv",
                    "cycle_landmarks.csv",
                    "pair_endpoint_timeseries.csv",
                    "endpoint_summaries.csv",
                    "between_method_spread.csv",
                    "pair_2_4_missingness_diagnostic.csv",
                    "source_provenance.json",
                },
            )
            with (
                source / "pair_endpoint_timeseries.csv"
            ).open(newline="", encoding="utf-8") as handle:
                endpoint_rows = list(csv.DictReader(handle))
            self.assertEqual(len(endpoint_rows), 7010)
            self.assertEqual(
                {row["scientific_series_label"] for row in endpoint_rows},
                {"Cu(111) thermal cycle"},
            )
            self.assertFalse(
                any(
                    "25w28" in value
                    for row in endpoint_rows
                    for value in row.values()
                )
            )
            provenance = json.loads(
                (source / "source_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                provenance["sources"]["primary_pair_endpoints"][
                    "evidence_role"
                ],
                "five-method positive control",
            )
            self.assertEqual(
                provenance["sources"]["pair_2_4_fit_diagnostics"][
                    "evidence_role"
                ],
                "separate saved-YOLO missingness diagnostic",
            )
            self.assertEqual(
                provenance["sources"]["frozen_subset_manifest"][
                    "evidence_role"
                ],
                "frozen five-method experimental design",
            )
            self.assertEqual(
                provenance["sources"]["completed_run_manifest"][
                    "evidence_role"
                ],
                "completed-run hash chain",
            )
            self.assertEqual(
                provenance["sources"]["completed_run_summary"][
                    "evidence_role"
                ],
                "proposal and common-fit denominators",
            )

    def test_source_package_rebuild_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trackleed-figure05-a-") as a:
            with tempfile.TemporaryDirectory(
                prefix="trackleed-figure05-b-"
            ) as b:
                root_a = Path(a)
                root_b = Path(b)
                freeze_source_data.build(root_a)
                freeze_source_data.build(root_b)
                files_a = sorted(
                    path.relative_to(root_a)
                    for path in root_a.rglob("*")
                    if path.is_file()
                )
                files_b = sorted(
                    path.relative_to(root_b)
                    for path in root_b.rglob("*")
                    if path.is_file()
                )

                self.assertEqual(files_a, files_b)
                self.assertEqual(
                    {
                        path.as_posix(): freeze_source_data.sha256_file(
                            root_a / path
                        )
                        for path in files_a
                    },
                    {
                        path.as_posix(): freeze_source_data.sha256_file(
                            root_b / path
                        )
                        for path in files_b
                    },
                )


if __name__ == "__main__":
    unittest.main()
