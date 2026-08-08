from __future__ import annotations

import hashlib
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.publication.cross_contract_manuscript import (
    CrossContractManuscriptError,
    freeze_cross_contract_manuscript,
    validate_cross_contract_manuscript,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "manuscript/cross_contract/main_manuscript"
STATIC_PATHS = ("README.md", "contract.json", "manuscript.md")
GENERATED_PATHS = ("manifest.json", "validation_report.json")


def _copy_static_package(destination: Path) -> None:
    destination.mkdir()
    for relative in STATIC_PATHS:
        shutil.copyfile(PACKAGE / relative, destination / relative)


def _copy_frozen_package(destination: Path) -> None:
    _copy_static_package(destination)
    for relative in GENERATED_PATHS:
        shutil.copyfile(PACKAGE / relative, destination / relative)


class CrossContractManuscriptTests(unittest.TestCase):
    def test_validation_fails_closed_when_manuscript_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractManuscriptError,
                "cross-contract manuscript contract is unavailable",
            ),
        ):
            validate_cross_contract_manuscript(Path(temporary))

    def test_focused_manuscript_respects_structure_budget_and_display_contract(
        self,
    ) -> None:
        report = validate_cross_contract_manuscript(REPO_ROOT)

        self.assertEqual(report["status"], "validated-cross-contract-manuscript")
        self.assertLessEqual(report["abstract_word_count"], 200)
        self.assertGreaterEqual(report["main_text_word_count"], 4200)
        self.assertLessEqual(report["main_text_word_count"], 4600)
        self.assertEqual(
            report["main_sections"],
            ["Introduction", "Methods", "Results", "Discussion", "Conclusions"],
        )
        self.assertEqual(report["figure_count"], 3)
        self.assertEqual(report["table_count"], 0)
        self.assertEqual(
            report["authorized_deferred_inputs"],
            ["EXPERIMENT", "SUBMISSION"],
        )
        self.assertEqual(report["claim_ids"], [f"XC{index}" for index in range(1, 8)])

        manifest_sha256 = hashlib.sha256(
            (PACKAGE / "manifest.json").read_bytes()
        ).hexdigest()
        self.assertEqual(
            manifest_sha256,
            "4d269530a760a2debcd84f588ed1367f0ab1d7998d26724b9a2b7a9bce6bbafe",
        )

    def test_freeze_refuses_to_replace_the_frozen_package(self) -> None:
        with self.assertRaisesRegex(FileExistsError, "Refusing to replace frozen"):
            freeze_cross_contract_manuscript(REPO_ROOT)

    def test_freeze_validates_staging_before_publishing(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            package = Path(temporary) / "main_manuscript"
            _copy_static_package(package)
            with (package / "manuscript.md").open(encoding="utf-8", mode="a") as handle:
                handle.write("\n| unauthorized | table |\n| --- | --- |\n")

            with self.assertRaisesRegex(
                CrossContractManuscriptError,
                "contains a table",
            ):
                freeze_cross_contract_manuscript(REPO_ROOT, package)

            self.assertFalse((package / "manifest.json").exists())
            self.assertFalse((package / "validation_report.json").exists())

    def test_validation_rejects_unmanifested_nested_content(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            package = Path(temporary) / "main_manuscript"
            _copy_frozen_package(package)
            extra = package / "extra"
            extra.mkdir()
            (extra / "unauthorized.md").write_text("not manifested\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CrossContractManuscriptError,
                "package file inventory changed",
            ):
                validate_cross_contract_manuscript(REPO_ROOT, package)

    def test_freeze_rejects_claim_outside_approved_sections(self) -> None:
        with TemporaryDirectory(dir=REPO_ROOT) as temporary:
            package = Path(temporary) / "main_manuscript"
            _copy_static_package(package)
            manuscript = (package / "manuscript.md").read_text(encoding="utf-8")
            manuscript = manuscript.replace(
                "These results show",
                "The sensitivity grid was stable. These results show",
                1,
            )
            (package / "manuscript.md").write_text(manuscript, encoding="utf-8")

            with self.assertRaisesRegex(
                CrossContractManuscriptError,
                "claim XC7 appears outside",
            ):
                freeze_cross_contract_manuscript(REPO_ROOT, package)

            self.assertFalse((package / "manifest.json").exists())
            self.assertFalse((package / "validation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
