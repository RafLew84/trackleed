from __future__ import annotations

import csv
import hashlib
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.publication.cross_contract_supplement import (
    CrossContractSupplementError,
    freeze_cross_contract_supplement,
    validate_cross_contract_supplement,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "manuscript/cross_contract/supplement"
STATIC_PATHS = ("README.md", "contract.json", "supplement.md")


def _copy_static_package(destination: Path) -> None:
    destination.mkdir()
    for relative in STATIC_PATHS:
        shutil.copyfile(PACKAGE / relative, destination / relative)


def _csv_rows(relative: str) -> list[dict[str, str]]:
    with (PACKAGE / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CrossContractSupplementTests(unittest.TestCase):
    def test_validation_fails_closed_when_supplement_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractSupplementError,
                "cross-contract supplement contract is unavailable",
            ),
        ):
            validate_cross_contract_supplement(Path(temporary))

    def test_supplement_respects_prose_display_and_evidence_boundaries(self) -> None:
        report = validate_cross_contract_supplement(REPO_ROOT)

        self.assertEqual(report["status"], "validated-cross-contract-supplement")
        self.assertGreaterEqual(report["prose_word_count"], 1800)
        self.assertLessEqual(report["prose_word_count"], 2200)
        self.assertEqual(report["figure_count"], 2)
        self.assertEqual(report["table_count"], 2)
        self.assertEqual(
            report["supporting_validation_items"],
            ["Figure S1", "Figure S2", "Table S1", "Table S2"],
        )
        self.assertEqual(
            report["fixed_denominators"],
            {
                "opportunities_per_method": 324,
                "reference_visible_per_method": 221,
                "accepted_out_of_branch_events": 124,
                "sensitivity_method_variant_rows": 45,
                "pilot_branch_frame_records": 60,
            },
        )
        self.assertTrue(report["repository_only_full_records"])
        self.assertEqual(
            report["display_validation"]["Figure S1"]["size_mm"],
            [88.0, 75.0],
        )
        self.assertEqual(
            report["display_validation"]["Figure S2"]["size_mm"],
            [88.0, 100.0],
        )
        self.assertEqual(
            hashlib.sha256((PACKAGE / "manifest.json").read_bytes()).hexdigest(),
            "39a33d38b0978ff7051e7596625f4ad11a709e01b72a59f2aaa3a1b0a3024c34",
        )

    def test_frozen_sources_are_compact_supporting_views(self) -> None:
        self.assertEqual(
            len(_csv_rows("source_data/figure_s1_pilot_calibration.csv")), 22
        )
        self.assertEqual(len(_csv_rows("source_data/figure_s2_sensitivity.csv")), 15)
        self.assertEqual(
            len(_csv_rows("source_data/table_s1_measurement_contracts.csv")), 2
        )
        self.assertEqual(len(_csv_rows("source_data/table_s2_pilot_validation.csv")), 6)

    def test_freeze_refuses_to_replace_the_frozen_supplement(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "Refusing to replace frozen"):
            freeze_cross_contract_supplement(REPO_ROOT)

    def test_freeze_validates_staging_before_publishing(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            package = Path(temporary) / "supplement"
            _copy_static_package(package)
            with (package / "supplement.md").open(encoding="utf-8", mode="a") as handle:
                handle.write(
                    "\n| unauthorized | third | table |\n"
                    "| --- | --- | --- |\n"
                    "| must | fail | closed |\n"
                )

            with self.assertRaisesRegex(
                CrossContractSupplementError,
                "table inventory changed",
            ):
                freeze_cross_contract_supplement(REPO_ROOT, package)

            self.assertFalse((package / "manifest.json").exists())
            self.assertFalse((package / "validation_report.json").exists())
            self.assertFalse((package / "output").exists())
            self.assertFalse((package / "source_data").exists())

    def test_validation_rejects_an_unmanifested_empty_directory(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            package = Path(temporary) / "supplement"
            shutil.copytree(PACKAGE, package)
            (package / "unmanifested").mkdir()

            with self.assertRaisesRegex(
                CrossContractSupplementError,
                "package file inventory changed",
            ):
                validate_cross_contract_supplement(REPO_ROOT, package)


if __name__ == "__main__":
    unittest.main()
