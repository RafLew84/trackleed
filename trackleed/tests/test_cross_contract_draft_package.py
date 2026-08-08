from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.publication.cross_contract_draft_package import (
    CrossContractDraftPackageError,
    build_cross_contract_draft_package,
    validate_cross_contract_draft_package,
)


class CrossContractDraftPackageTests(unittest.TestCase):
    def test_assembled_documents_record_completed_status_and_cross_references(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        package = repository_root / "manuscript/cross_contract/draft_package"
        main = (package / "assembled/main_manuscript.md").read_text(encoding="utf-8")
        supplement = (package / "assembled/supplementary_information.md").read_text(
            encoding="utf-8"
        )

        for reference in (
            "Figure 1 summarizes",
            "Figure 2 shows",
            "Figure 3 shows",
            "Figure S1, Figure S2, Table S1, and Table S2",
        ):
            self.assertIn(reference, main)
        self.assertNotIn("Ticket 09 will", supplement)
        self.assertNotIn("Until that integration", supplement)
        self.assertIn("Page-level render acceptance is complete", supplement)

    def test_canonical_package_is_a_validated_non_submission_ready_draft(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]

        report = validate_cross_contract_draft_package(repository_root)

        self.assertEqual(
            report["status"],
            "validated-cross-contract-scientific-draft-package",
        )
        self.assertEqual(report["abstract_word_count"], 152)
        self.assertEqual(report["main_text_word_count"], 4598)
        self.assertEqual(report["main_figure_count"], 3)
        self.assertEqual(report["main_table_count"], 0)
        self.assertEqual(report["supplement_figure_count"], 2)
        self.assertEqual(report["supplement_table_count"], 2)
        self.assertEqual(
            report["authorized_deferred_inputs"], ["EXPERIMENT", "SUBMISSION"]
        )
        self.assertFalse(report["submission_ready"])
        self.assertEqual(
            report["page_level_render_acceptance"],
            {
                "reviewed_page_count": 18,
                "reviewer_identity": "Codex technical page reviewer",
                "status": "page-level-render-accepted",
                "unique_page_count": 9,
            },
        )

    def test_validation_fails_closed_when_draft_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "cross-contract draft package contract is unavailable",
            ),
        ):
            validate_cross_contract_draft_package(Path(temporary))

    def test_validation_rejects_a_stale_frozen_source_binding(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        canonical = repository_root / "manuscript/cross_contract/draft_package"
        contract = json.loads(
            (canonical / "contract.json").read_text(encoding="utf-8")
        )
        with (
            TemporaryDirectory(dir=repository_root) as temporary,
            self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "source binding hash mismatch",
            ),
        ):
            temporary_root = Path(temporary)
            package = temporary_root / "manuscript/cross_contract/draft_package"
            package.parent.mkdir(parents=True)
            shutil.copytree(canonical, package)
            for record in contract["source_bindings"].values():
                relative = Path(record["path"])
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repository_root / relative, destination)
            stale = temporary_root / Path(
                contract["source_bindings"]["main_figure_manifest"]["path"]
            )
            stale.write_bytes(stale.read_bytes() + b"\n")

            validate_cross_contract_draft_package(temporary_root)

    def test_validation_rejects_unmanifested_nested_content(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        canonical = repository_root / "manuscript/cross_contract/draft_package"
        with TemporaryDirectory(dir=repository_root) as temporary:
            package = Path(temporary) / "draft_package"
            shutil.copytree(canonical, package)
            unauthorized = package / "extra/unauthorized.md"
            unauthorized.parent.mkdir()
            unauthorized.write_text("not manifested\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "draft package file inventory changed",
            ):
                validate_cross_contract_draft_package(repository_root, package)

    def test_publish_requires_explicit_complete_page_acceptance(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        canonical = repository_root / "manuscript/cross_contract/draft_package"
        with TemporaryDirectory(dir=repository_root) as temporary:
            package = Path(temporary) / "draft_package"
            package.mkdir()
            shutil.copyfile(canonical / "README.md", package / "README.md")
            shutil.copyfile(canonical / "contract.json", package / "contract.json")

            with self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "explicit complete Page-level render acceptance",
            ):
                build_cross_contract_draft_package(
                    repository_root,
                    Path(temporary) / "missing-stage-a",
                    Path(temporary) / "missing-stage-b",
                    reviewer_identity="",
                    review_date="",
                    confirm_page_acceptance=False,
                    package_path=package,
                )

    def test_publish_requires_distinct_independent_staging_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        canonical = repository_root / "manuscript/cross_contract/draft_package"
        with TemporaryDirectory(dir=repository_root) as temporary:
            package = Path(temporary) / "draft_package"
            package.mkdir()
            shutil.copyfile(canonical / "README.md", package / "README.md")
            shutil.copyfile(canonical / "contract.json", package / "contract.json")
            stage = Path(temporary) / "same-stage"

            with self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "distinct independent staging paths",
            ):
                build_cross_contract_draft_package(
                    repository_root,
                    stage,
                    stage,
                    reviewer_identity="Codex technical page reviewer",
                    review_date="2026-08-08",
                    confirm_page_acceptance=True,
                    package_path=package,
                )

    def test_validation_rejects_drift_hidden_behind_an_unchanged_manifest(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        with TemporaryDirectory(dir=repository_root) as temporary:
            temporary_root = Path(temporary)
            manuscript = temporary_root / "manuscript"
            cross_contract = manuscript / "cross_contract"
            cross_contract.mkdir(parents=True)
            for package_name in (
                "draft_package",
                "figures",
                "main_manuscript",
                "supplement",
            ):
                shutil.copytree(
                    repository_root / "manuscript/cross_contract" / package_name,
                    cross_contract / package_name,
                )
            claim_boundary = cross_contract / "claim_boundary"
            claim_boundary.mkdir()
            shutil.copyfile(
                repository_root
                / "manuscript/cross_contract/claim_boundary/decision_record.json",
                claim_boundary / "decision_record.json",
            )
            shutil.copyfile(
                repository_root / "manuscript/references.bib",
                manuscript / "references.bib",
            )
            publication_contract = temporary_root / "docs/publication-status-contract.md"
            publication_contract.parent.mkdir()
            shutil.copyfile(
                repository_root / "docs/publication-status-contract.md",
                publication_contract,
            )
            stale = cross_contract / "main_manuscript/manuscript.md"
            stale.write_bytes(stale.read_bytes() + b"\n")

            with self.assertRaisesRegex(
                CrossContractDraftPackageError,
                "upstream main manuscript package failed validation",
            ):
                validate_cross_contract_draft_package(temporary_root)

    def test_publish_refuses_to_replace_the_frozen_package(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(
            FileExistsError,
            "Refusing to replace frozen draft package",
        ):
            build_cross_contract_draft_package(
                repository_root,
                repository_root / "missing-stage-a",
                repository_root / "missing-stage-b",
                reviewer_identity="Codex technical page reviewer",
                review_date="2026-08-08",
                confirm_page_acceptance=True,
            )


if __name__ == "__main__":
    unittest.main()
