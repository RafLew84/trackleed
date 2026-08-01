#!/usr/bin/env python3
"""Contract tests for supplementary Table S4."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_table


TABLE_ROOT = Path(__file__).resolve().parent
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"


class Cu111EndpointTableTest(unittest.TestCase):
    def test_table_preserves_five_methods_and_two_primary_pairs(self) -> None:
        rows = build_table.load_and_validate_endpoints()

        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["pair"] for row in rows},
            {"1–6", "3–5"},
        )
        self.assertEqual(
            {row["method"] for row in rows},
            {
                "LocoTrack",
                "OSTrack",
                "DAM4SAM",
                "Lucas–Kanade",
                "EasyLEED",
            },
        )
        self.assertTrue(
            all(row["valid_endpoints"] == "701/701" for row in rows)
        )

    def test_exact_endpoint_values_are_retained_before_display_rounding(
        self,
    ) -> None:
        rows = build_table.load_and_validate_endpoints()
        easy_pair_3_5 = next(
            row
            for row in rows
            if row["pair"] == "3–5" and row["method"] == "EasyLEED"
        )

        self.assertEqual(
            easy_pair_3_5["plateau_median_percent"],
            "-1.5707547878278549",
        )
        self.assertEqual(
            easy_pair_3_5["post_plateau_minimum_percent"],
            "-1.5801230095593755",
        )
        self.assertEqual(
            easy_pair_3_5["final_cooled_state_percent"],
            "-0.15234264342884885",
        )

    def test_reference_denominator_and_landmark_roles_are_explicit(self) -> None:
        rows = build_table.load_and_validate_endpoints()
        landmarks = build_table.load_and_validate_landmarks()

        self.assertTrue(
            all(row["reference_valid_images"] == "10/10" for row in rows)
        )
        self.assertTrue(
            all(float(row["reference_distance_px"]) > 0 for row in rows)
        )
        by_name = {row["landmark"]: row for row in landmarks}
        self.assertEqual(by_name["plateau"]["temperature"], "800 °C")
        self.assertEqual(by_name["plateau"]["selection"], "predefined")
        self.assertEqual(
            by_name["post_plateau_minimum"]["temperature"],
            "797 °C",
        )
        self.assertEqual(
            by_name["post_plateau_minimum"]["selection"],
            "outcome-selected",
        )
        self.assertEqual(by_name["final"]["temperature"], "92 °C")
        self.assertEqual(by_name["final"]["selection"], "predefined")

    def test_spread_contract_matches_method_level_endpoint_extrema(self) -> None:
        rows = build_table.load_and_validate_endpoints()

        build_table.validate_between_method_spread(rows)

    def test_render_is_reader_facing_and_states_scientific_limits(self) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_endpoints(),
        )

        self.assertTrue(rendered.startswith("**Table S4."))
        self.assertIn("701/701", rendered)
        self.assertIn("outcome-selected", rendered)
        self.assertIn("first ten images", rendered)
        self.assertIn("not external absolute accuracy", rendered)
        self.assertIn("do not establish crystallographic anisotropy", rendered)
        self.assertNotIn("frame 378", rendered)
        self.assertNotIn("frame 700", rendered)
        self.assertNotIn("locotrack_base_offline", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("C:\\Users", rendered)

    def test_render_rounds_endpoint_values_to_six_decimal_places(self) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_endpoints(),
        )

        self.assertIn("| 1–6 | OSTrack | 701/701 | −0.686676", rendered)
        self.assertIn("| 3–5 | EasyLEED | 701/701 | −1.570755", rendered)
        self.assertIn("| 1–6 | DAM4SAM | 701/701 | −0.686244", rendered)

    def test_provenance_is_confined_and_hash_checked(self) -> None:
        provenance = build_table.validate_provenance()

        self.assertEqual(len(provenance["sources"]), 3)
        self.assertIn("Figure 3", provenance["freeze_policy"])
        for source in provenance["sources"]:
            self.assertTrue(
                source["origin_path"].startswith(
                    "manuscript/figures/figure03_cu111_physical_endpoint/"
                )
            )
            frozen = TABLE_ROOT / source["frozen_path"]
            origin = build_table.REPOSITORY_ROOT / source["origin_path"]
            self.assertTrue(frozen.is_file())
            self.assertTrue(origin.is_file())
            self.assertEqual(
                build_table.sha256_file(frozen),
                source["frozen_sha256"],
            )
            self.assertEqual(
                build_table.sha256_file(origin),
                source["origin_sha256"],
            )

    def test_provenance_rejects_malformed_hashes_and_escaping_paths(self) -> None:
        provenance = json.loads(
            build_table.PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        for field, value in (
            ("frozen_sha256", "not-a-hash"),
            ("frozen_path", "../../../../etc/passwd"),
            ("origin_path", "../../../../etc/passwd"),
        ):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(provenance))
                changed["sources"][0][field] = value
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "provenance.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with mock.patch.object(
                        build_table,
                        "PROVENANCE_PATH",
                        path,
                    ):
                        with self.assertRaises(ValueError):
                            build_table.validate_provenance()

    def test_artifact_is_deterministic_and_manifest_covers_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "output"
            manifest_path = root / "manifest.json"

            first = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            first_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }
            second = build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=manifest_path,
            )
            second_bytes = {
                path.name: path.read_bytes()
                for path in output_directory.iterdir()
            }

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first, second)
            self.assertEqual(
                set(first_bytes),
                {
                    "table_s4_cu111_endpoints.csv",
                    "table_s4_cu111_endpoints.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["outputs"]), set(first_bytes))
            self.assertEqual(manifest["row_count"], 10)
            self.assertEqual(
                manifest["reporting_unit"],
                "method–pair within one Cu(111) thermal-cycle series",
            )

    def test_supplement_markers_must_be_one_ordered_pair(self) -> None:
        valid = (
            f"{build_table.SUPPLEMENT_BEGIN_MARKER}\ncontent\n"
            f"{build_table.SUPPLEMENT_END_MARKER}"
        )
        self.assertEqual(build_table.extract_supplement_block(valid), "content\n")

        invalid_records = (
            build_table.SUPPLEMENT_BEGIN_MARKER,
            build_table.SUPPLEMENT_END_MARKER,
            (
                f"{valid}\n"
                f"{build_table.SUPPLEMENT_BEGIN_MARKER}\nduplicate\n"
                f"{build_table.SUPPLEMENT_END_MARKER}"
            ),
            (
                f"{build_table.SUPPLEMENT_END_MARKER}\n"
                f"{build_table.SUPPLEMENT_BEGIN_MARKER}"
            ),
        )
        for invalid in invalid_records:
            with self.subTest(record=invalid):
                with self.assertRaises(ValueError):
                    build_table.extract_supplement_block(invalid)

    def test_committed_outputs_and_supplement_are_synchronized(self) -> None:
        output_directory = TABLE_ROOT / "output"
        expected = {
            path.name: path.read_bytes()
            for path in output_directory.iterdir()
            if path.is_file()
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated_output = root / "output"
            generated_manifest = root / "manifest.json"
            build_table.build_artifact(
                output_directory=generated_output,
                manifest_path=generated_manifest,
            )
            actual = {
                path.name: path.read_bytes()
                for path in generated_output.iterdir()
            }
            generated_manifest_bytes = generated_manifest.read_bytes()

        self.assertEqual(expected, actual)
        self.assertEqual(
            (TABLE_ROOT / "manifest.json").read_bytes(),
            generated_manifest_bytes,
        )
        markdown = (
            output_directory / "table_s4_cu111_endpoints.md"
        ).read_text(encoding="utf-8")
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            build_table.extract_supplement_block(supplement),
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
