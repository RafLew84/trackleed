#!/usr/bin/env python3
"""Source-contract tests for publication Figure 5."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import freeze_source_data
import tifffile


class Figure5SourceFreezeTest(unittest.TestCase):
    def test_family_aggregate_preserves_matched_control(self) -> None:
        rows = freeze_source_data.load_family_aggregate()

        self.assertEqual(
            [row["method"] for row in rows],
            ["sam2", "samurai", "dam4sam"],
        )
        self.assertEqual(
            {row["checkpoint_sha256"] for row in rows},
            {
                "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb"
                "746c004c5"
            },
        )
        self.assertTrue(all(row["identical_checkpoint_weights"] for row in rows))
        self.assertTrue(all(row["visible_opportunities"] == 182 for row in rows))
        self.assertTrue(all(row["evaluated_gaps"] == 13 for row in rows))
        self.assertEqual(
            [row["branch_consistent_measurements"] for row in rows],
            [126, 181, 179],
        )
        self.assertEqual([row["recovered_gaps"] for row in rows], [9, 13, 13])
        self.assertEqual(
            [row["geometry_inconsistent_accepted_fits"] for row in rows],
            [8, 36, 36],
        )

    def test_two_disjoint_pairs_reproduce_aggregate_counts(self) -> None:
        aggregate = freeze_source_data.load_family_aggregate()
        pair_rows = freeze_source_data.load_pair_results()
        count_fields = (
            "visible_opportunities",
            "branch_consistent_measurements",
            "evaluated_gaps",
            "recovered_gaps",
            "accepted_measurements_all_frames",
            "geometry_inconsistent_accepted_fits",
            "jointly_consistent_pair_frames",
        )

        self.assertEqual(
            {(row["pair"], row["visible_opportunities"]) for row in pair_rows},
            {("pair 1", 91), ("pair 2", 91)},
        )
        for aggregate_row in aggregate:
            method_rows = [
                row
                for row in pair_rows
                if row["method"] == aggregate_row["method"]
            ]
            self.assertEqual(len(method_rows), 2)
            for field in count_fields:
                self.assertEqual(
                    sum(row[field] for row in method_rows),
                    aggregate_row[field],
                )

    def test_pair_seeds_are_frozen_and_spatially_disjoint(self) -> None:
        rows = freeze_source_data.load_pair_seeds()

        self.assertEqual(
            {(row["pair"], row["branch"]) for row in rows},
            {
                ("pair 1", "B1"),
                ("pair 1", "B2"),
                ("pair 2", "B1"),
                ("pair 2", "B2"),
            },
        )
        self.assertTrue(
            all(row["reference_energy_ev"] == 187.9 for row in rows)
        )
        pair_1 = [row for row in rows if row["pair"] == "pair 1"]
        pair_2 = [row for row in rows if row["pair"] == "pair 2"]
        for first in pair_1:
            for second in pair_2:
                separation = (
                    (first["center_x"] - second["center_x"]) ** 2
                    + (first["center_y"] - second["center_y"]) ** 2
                ) ** 0.5
                self.assertGreaterEqual(separation, 48.0)

    def test_representative_gap_exposes_distinct_memory_behaviors(self) -> None:
        selected = freeze_source_data.select_representative_gap()
        rows = freeze_source_data.build_representative_gap_rows(selected)

        self.assertEqual(selected["scientific_series_label"], "Controlled scan")
        self.assertEqual(selected["scientific_pair_label"], "pair 1")
        self.assertEqual(selected["branch"], "B1")
        self.assertEqual(selected["direction"], "decreasing energy")
        self.assertEqual(
            selected["display_energies_ev"],
            [169.2, 166.5, 163.9, 160.9],
        )
        self.assertTrue(selected["selection_uses_method_outcomes"])
        self.assertEqual(len(rows), 12)

        post_gap = {
            row["method"]: row
            for row in rows
            if row["energy_ev"] == 160.9
        }
        self.assertFalse(post_gap["sam2"]["native_visible"])
        self.assertFalse(post_gap["sam2"]["measurement_valid"])
        for method in ("samurai", "dam4sam"):
            self.assertTrue(post_gap[method]["native_visible"])
            self.assertTrue(post_gap[method]["measurement_valid"])
            self.assertEqual(
                post_gap[method]["measurement_status"],
                "branch_consistent",
            )

        gap_rows = [
            row
            for row in rows
            if row["energy_ev"] in {166.5, 163.9}
        ]
        self.assertTrue(
            all(row["physics_baseline_status"] == "iv_visibility_gap"
                for row in gap_rows)
        )
        self.assertTrue(
            all(
                not row["measurement_valid"]
                for row in gap_rows
                if row["method"] == "samurai"
            )
        )

    def test_build_creates_self_contained_source_package(self) -> None:
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
                    "family_aggregate.csv",
                    "family_by_pair.csv",
                    "pair_seeds.csv",
                    "representative_gap_measurements.csv",
                    "representative_gap.json",
                    "source_provenance.json",
                    "crops/b1_169p2ev.tiff",
                    "crops/b1_166p5ev.tiff",
                    "crops/b1_163p9ev.tiff",
                    "crops/b1_160p9ev.tiff",
                },
            )
            representative = json.loads(
                (source / "representative_gap.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                representative["crop_box_xyxy"],
                [200, 456, 328, 584],
            )
            crop = tifffile.imread(source / "crops/b1_169p2ev.tiff")
            upstream = tifffile.imread(
                freeze_source_data.SCAN_FRAMES[169.2]
            )
            self.assertEqual(crop.shape, (128, 128))
            self.assertTrue(
                (crop == upstream[456:584, 200:328]).all()
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
