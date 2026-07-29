#!/usr/bin/env python3
"""Contract tests for main Table 2 and its manuscript copy."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import build_table


TABLE_ROOT = Path(__file__).resolve().parent
MANUSCRIPT_PATH = TABLE_ROOT.parents[1] / "manuscript.md"
SHARED_SAM2_CHECKPOINT = (
    "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)


class TrackingMethodsContractTableTest(unittest.TestCase):
    def test_rows_keep_the_frozen_panels_and_method_order(self) -> None:
        rows = build_table.load_and_validate_rows()

        self.assertEqual(len(rows), 9)
        self.assertEqual(
            [row["panel"] for row in rows],
            ["Main"] * 5
            + ["Controlled SAM2 family"] * 3
            + ["External reference"],
        )
        self.assertEqual(
            [row["method"] for row in rows],
            [
                "LocoTrack",
                "OSTrack",
                "DAM4SAM",
                "Lucas–Kanade",
                "EasyLEED",
                "SAM2.1",
                "SAMURAI",
                "DAM4SAM",
                "ViPErLEED",
            ],
        )

    def test_audit_fields_preserve_checkpoint_and_configuration_hashes(
        self,
    ) -> None:
        rows = build_table.load_and_validate_rows()
        main = rows[:5]
        family = rows[5:8]

        self.assertEqual(
            main[0]["checkpoint_sha256"],
            "8a5adbaeb610d1f06adfbc7c9076b66f727d674c0fd1d668890201cf3339736c",
        )
        self.assertEqual(
            main[1]["executed_configuration_sha256"],
            "dc9bd17199ffc0aaea78fc00d0f68ba0b2b3047ef15fac30ec1c4997ed973736",
        )
        self.assertEqual(
            {row["checkpoint_sha256"] for row in family},
            {SHARED_SAM2_CHECKPOINT},
        )
        self.assertEqual(
            {
                row["upstream_revision"]
                for row in family
            },
            {
                "2b90b9f5ceec907a1c18123530e92e794ad901a4",
                "76ba195984892b0d1e3db5d9c9f90bb62175680a",
                "9c954504b39ebca4c412f207be0787c26bfac85a",
            },
        )
        self.assertEqual(main[3]["checkpoint_sha256"], "not applicable")
        self.assertEqual(main[4]["software_version"], "2.5.2")
        self.assertEqual(
            {
                row["executed_configuration_sha256"] for row in family
            },
            {"not recorded separately"},
        )
        self.assertNotIn(
            "ee579ab76263a3ee20c227f79f83bc1c1d6f26793718209a0b517f5e5131a60c",
            {
                row["executed_configuration_sha256"] for row in rows
            },
        )

    def test_reader_table_has_nine_columns_and_separates_viperleed(
        self,
    ) -> None:
        rendered = build_table.render_markdown(
            build_table.load_and_validate_rows()
        )

        self.assertIn(
            "The controlled SAM2-family rows used byte-identical",
            rendered,
        )
        self.assertIn(SHARED_SAM2_CHECKPOINT, rendered)
        self.assertIn(
            "ViPErLEED is a separate whole-pattern external reference",
            rendered,
        )
        self.assertIn(
            "not a sixth row in the common five-method matrix",
            rendered,
        )
        self.assertIn(
            "same 48 × 48 raw-image measurement ROI",
            rendered,
        )
        self.assertIn(
            "TrackLEED two-dimensional elliptical-Gaussian estimator",
            rendered,
        )
        self.assertIn(
            (
                "energy scans additionally require combined native "
                "visibility probability > 0.5"
            ),
            rendered,
        )
        self.assertIn("Finite exported Kalman x,y", rendered)
        self.assertNotIn("ScanE", rendered)
        self.assertNotIn("26a21", rendered)
        self.assertNotIn("26a24", rendered)
        lines = [
            line for line in rendered.splitlines() if line.startswith("|")
        ]
        self.assertEqual(len(lines), 11)
        self.assertEqual({line.count("|") for line in lines}, {10})

    def test_publication_csv_excludes_internal_audit_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "output"
            build_table.build_artifact(
                output_directory=output_directory,
                manifest_path=Path(temporary_directory) / "manifest.json",
            )
            with (
                output_directory / "table02_tracking_methods_contract.csv"
            ).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 9)
        self.assertEqual(
            tuple(rows[0]),
            tuple(label for _, label in build_table.DISPLAY_COLUMNS),
        )
        self.assertNotIn("checkpoint_sha256", rows[0])
        self.assertNotIn("upstream_revision", rows[0])

    def test_provenance_uses_hash_checked_local_snapshots(self) -> None:
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

    def test_artifact_is_deterministic_and_records_contract(self) -> None:
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
                    "table02_tracking_methods_contract.csv",
                    "table02_tracking_methods_contract.md",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = manifest["table_contract"]
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(contract["row_count"], 9)
            self.assertEqual(contract["main_method_count"], 5)
            self.assertEqual(contract["controlled_family_count"], 3)
            self.assertEqual(contract["external_reference_count"], 1)
            self.assertTrue(contract["external_reference_is_separate"])
            self.assertTrue(contract["controlled_family_weights_identical"])
            self.assertEqual(
                contract["controlled_family_checkpoint_sha256"],
                SHARED_SAM2_CHECKPOINT,
            )
            self.assertEqual(len(manifest["outputs"]), 2)
            self.assertEqual(len(manifest["source_data"]), 7)

    def test_manuscript_copy_matches_generated_markdown(self) -> None:
        manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
        table_block = build_table.extract_manuscript_table(manuscript)

        self.assertEqual(
            table_block,
            build_table.render_markdown(build_table.load_and_validate_rows()),
        )
        self.assertGreater(
            manuscript.index(build_table.MANUSCRIPT_BEGIN_MARKER),
            manuscript.index("### 3.3 Domain-specific references"),
        )
        self.assertLess(
            manuscript.index(build_table.MANUSCRIPT_END_MARKER),
            manuscript.index("## 4. Validation protocol"),
        )


if __name__ == "__main__":
    unittest.main()
