#!/usr/bin/env python3
"""Contract tests for main Table 1 and its manuscript copy."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_table


TABLE_ROOT = Path(__file__).resolve().parent
MANUSCRIPT_PATH = TABLE_ROOT.parents[1] / "manuscript.md"


class DataEvidenceRolesTableTest(unittest.TestCase):
    def test_source_rows_encode_four_regimes_and_five_evidence_roles(
        self,
    ) -> None:
        rows = build_table.load_and_validate_rows()

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [row["row_type"] for row in rows],
            [
                "primary_regime",
                "derived_regime",
                "primary_regime",
                "primary_regime",
                "subset",
            ],
        )
        self.assertEqual(
            rows[1]["scientific_series_label"],
            "Controlled translations",
        )
        self.assertIn(
            "not four acquisitions",
            rows[1]["independence_and_reporting_unit"],
        )
        self.assertIn(
            "not an independent replicate",
            rows[4]["independence_and_reporting_unit"],
        )

    def test_reader_facing_table_excludes_internal_archive_names(self) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_rows()
        )

        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("26a24", rendered)

    def test_markdown_states_nonindependence_and_has_seven_columns(self) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_rows()
        )

        self.assertIn(
            "Rows describe evidence roles, not five independent datasets.",
            rendered,
        )
        self.assertIn(
            "Controlled translations are derived from the Cu(111) sequence",
            rendered,
        )
        self.assertIn(
            "external-reference row is a subset of the memory-analysis scan",
            rendered,
        )
        self.assertIn(
            "outside the common five-method matrix",
            rendered,
        )
        table_lines = [
            line for line in rendered.splitlines() if line.startswith("|")
        ]
        self.assertEqual(len(table_lines), 7)
        self.assertEqual(
            {line.count("|") for line in table_lines},
            {8},
        )
        self.assertEqual(
            table_lines[0],
            (
                "| Scientific series label | Control variable/range | "
                "Image count | Frozen spots/pairs | Evidence role | "
                "Primary endpoint | Independence/replication unit |"
            ),
        )

    def test_provenance_uses_hash_checked_local_frozen_contracts(self) -> None:
        provenance = build_table.validate_provenance()

        self.assertEqual(len(provenance["sources"]), 5)
        for source in provenance["sources"]:
            frozen_path = build_table.TABLE_ROOT / source["frozen_path"]
            self.assertTrue(frozen_path.is_file())
            self.assertNotIn(".scratch", source["frozen_path"])
            self.assertEqual(
                build_table.sha256_file(frozen_path),
                source["frozen_sha256"],
            )

    def test_artifact_is_deterministic_and_records_table_contract(self) -> None:
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
                    "table01_data_evidence_roles.csv",
                    "table01_data_evidence_roles.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["table_contract"]["row_count"], 5)
            self.assertEqual(
                manifest["table_contract"]["data_regime_count"],
                4,
            )
            self.assertTrue(
                manifest["table_contract"][
                    "controlled_translation_is_derived"
                ]
            )
            self.assertTrue(
                manifest["table_contract"]["external_reference_is_subset"]
            )
            self.assertEqual(
                manifest["table_contract"]["ranking_policy"],
                "no composite tracker ranking",
            )
            self.assertIn("builder_sha256", manifest)
            self.assertEqual(len(manifest["outputs"]), 2)
            self.assertEqual(len(manifest["source_data"]), 7)

            with (
                output_directory / "table01_data_evidence_roles.csv"
            ).open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 5)
            self.assertNotIn("row_type", csv_rows[0])
            self.assertNotIn("order", csv_rows[0])

    def test_manuscript_copy_matches_generated_markdown(self) -> None:
        manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
        table_block = build_table.extract_manuscript_table(manuscript)

        self.assertEqual(
            table_block,
            build_table.render_markdown(build_table.load_and_validate_rows()),
        )
        self.assertLess(
            manuscript.index(build_table.MANUSCRIPT_BEGIN_MARKER),
            manuscript.index("### 2.2 Tracking proposal and measurement ROI"),
        )


if __name__ == "__main__":
    unittest.main()
