#!/usr/bin/env python3
"""Contract tests for Table S7 and its supplementary-material copy."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_table


TABLE_ROOT = Path(__file__).resolve().parent
SUPPLEMENT_PATH = TABLE_ROOT.parents[1] / "supplementary-material.md"


class CrossRegimeSummaryTableTest(unittest.TestCase):
    def test_source_contains_the_common_five_methods_in_frozen_order(
        self,
    ) -> None:
        rows = build_table.load_and_validate_source()

        self.assertEqual(
            [row["display_method"] for row in rows],
            [
                "LocoTrack",
                "OSTrack",
                "DAM4SAM",
                "Lucas–Kanade",
                "EasyLEED",
            ],
        )

    def test_blocks_preserve_counts_and_conditional_denominators(self) -> None:
        blocks = build_table.build_blocks(
            build_table.load_and_validate_source()
        )

        translation = blocks["controlled_translation"][0]
        self.assertEqual(translation["method"], "LocoTrack")
        self.assertEqual(translation["proposal_coverage"], "80/80 (100.00%)")
        self.assertEqual(
            translation["measurement_coverage"],
            "80/80 (100.00%)",
        )
        self.assertEqual(
            translation["median_p95_displacement_error_px"],
            "0.0043 / 0.0130",
        )
        self.assertEqual(translation["pair_coverage"], "40/40 (100.00%)")
        self.assertEqual(
            translation["maximum_abs_delta_d_over_d_pp"],
            "0.001229",
        )

        cu = blocks["cu111_positive_control"][-1]
        self.assertEqual(cu["method"], "EasyLEED")
        self.assertEqual(
            cu["gaussian_measurement_coverage"],
            "2804/2804 (100.00%)",
        )
        self.assertEqual(cu["valid_pair_endpoints"], "701/701; 701/701")
        self.assertEqual(
            cu["median_max_abs_endpoint_deviation_pp"],
            "0.000256 / 0.000949",
        )

        energy_scan = blocks["energy_scan_stress_test"][3]
        self.assertEqual(energy_scan["method"], "Lucas–Kanade")
        self.assertEqual(
            energy_scan["measurement_coverage"],
            "188/221 (85.07%)",
        )
        self.assertEqual(
            energy_scan["branch_consistent_recovery"],
            "17/24 (70.83%)",
        )
        self.assertEqual(
            energy_scan["geometry_inconsistent_continuation"],
            "14/206 (6.80%)",
        )
        self.assertEqual(
            energy_scan["conditional_median_pair_distance_residual_px"],
            "1.764 (n=68)",
        )

    def test_reader_facing_table_has_three_blocks_without_archive_names(
        self,
    ) -> None:
        rendered = build_table.render_markdown(
            build_table.build_blocks(
                build_table.load_and_validate_source()
            )
        )

        self.assertIn("**Controlled translation**", rendered)
        self.assertIn("**Cu(111) positive control**", rendered)
        self.assertIn(
            "**Independent Ag(111) energy-scan stress test**",
            rendered,
        )
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("| Score |", rendered)
        self.assertNotIn("| Rank |", rendered)
        self.assertIn(
            "No composite score or rank is calculated.",
            rendered,
        )
        self.assertIn(
            "The energy-scan residual is conditional",
            rendered,
        )
        self.assertIn(
            "Translation errors are conditional",
            rendered,
        )
        self.assertIn(
            "Branch-consistent measurement coverage",
            rendered,
        )
        self.assertIn(
            "DAM4SAM used sam21pp-L for Cu(111) and controlled translations",
            rendered,
        )
        self.assertIn(
            "sam21pp-B for the energy scans",
            rendered,
        )

    def test_provenance_hash_checks_the_committed_matrix(self) -> None:
        provenance = build_table.validate_provenance()
        source = provenance["sources"][0]
        frozen_path = build_table.TABLE_ROOT / source["frozen_path"]

        self.assertTrue(frozen_path.is_file())
        self.assertNotIn(".scratch", source["frozen_path"])
        self.assertEqual(
            build_table.sha256_file(frozen_path),
            source["frozen_sha256"],
        )

    def test_artifact_is_deterministic_and_records_no_ranking_policy(
        self,
    ) -> None:
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

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                set(first_bytes),
                {
                    "table_s7_cross_regime_summary.csv",
                    "table_s7_cross_regime_summary.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["table_contract"]["method_count"], 5)
            self.assertEqual(manifest["table_contract"]["block_count"], 3)
            self.assertEqual(
                manifest["table_contract"]["placement"],
                "supplementary",
            )
            self.assertEqual(
                manifest["table_contract"][
                    "translation_displaced_spot_opportunities_per_method"
                ],
                80,
            )
            self.assertEqual(
                manifest["table_contract"][
                    "translation_displaced_pair_opportunities_per_method"
                ],
                40,
            )
            self.assertEqual(
                manifest["table_contract"]["ranking_policy"],
                "no composite score, rank, or rank-like color scale",
            )
            self.assertTrue(
                manifest["table_contract"][
                    "conditional_residual_keeps_pair_frame_n"
                ]
            )
            self.assertEqual(len(manifest["source_data"]), 2)
            self.assertEqual(len(manifest["outputs"]), 2)

            with (
                output_directory / "table_s7_cross_regime_summary.csv"
            ).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertNotIn("score", {key.lower() for key in rows[0]})
            self.assertNotIn("rank", {key.lower() for key in rows[0]})

    def test_supplement_copy_matches_generated_markdown(self) -> None:
        supplement = SUPPLEMENT_PATH.read_text(encoding="utf-8")
        table_block = build_table.extract_supplement_table(supplement)
        rendered = build_table.render_markdown(
            build_table.build_blocks(
                build_table.load_and_validate_source()
            )
        )

        self.assertEqual(table_block, rendered)
        self.assertGreater(
            supplement.index(build_table.SUPPLEMENT_BEGIN_MARKER),
            supplement.index("<!-- END TABLE S6 -->"),
        )
        self.assertEqual(supplement.count("<!-- BEGIN TABLE S7 -->"), 1)
        self.assertEqual(supplement.count("<!-- END TABLE S7 -->"), 1)


if __name__ == "__main__":
    unittest.main()
