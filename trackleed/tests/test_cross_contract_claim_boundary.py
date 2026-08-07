from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trackleed.publication.cross_contract_claim_boundary import (
    CrossContractClaimBoundaryError,
    freeze_cross_contract_claim_boundary,
    preview_cross_contract_claim_boundary,
    validate_cross_contract_claim_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = REPO_ROOT / "manuscript/cross_contract/claim_boundary"


def copy_proposal(destination: Path) -> None:
    destination.mkdir()
    shutil.copyfile(
        PROPOSAL / "decision_record.json", destination / "decision_record.json"
    )
    shutil.copyfile(PROPOSAL / "README.md", destination / "README.md")


class CrossContractClaimBoundaryTests(unittest.TestCase):
    def test_committed_author_approved_package_validates(self) -> None:
        report = validate_cross_contract_claim_boundary(REPO_ROOT, PROPOSAL)

        self.assertEqual(report["status"], "validated-author-approved-boundary")
        self.assertEqual(
            report["decision_record_sha256"],
            "0f3e4d727fdf66721fa76924728fcf67ae1443c6c09654395093b20434ed6c18",
        )
        self.assertEqual(
            report["author_checkpoint"],
            {
                "confirmed_by": "publication author via repository HITL checkpoint",
                "confirmed_date": "2026-08-08",
            },
        )

    def test_validation_fails_closed_when_decision_package_is_absent(self) -> None:
        with (
            TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                CrossContractClaimBoundaryError,
                "cross-contract claim-boundary decision is unavailable",
            ),
        ):
            validate_cross_contract_claim_boundary(
                Path(temporary), Path(temporary) / "claim-boundary"
            )

    def test_preview_authenticates_sources_and_exposes_the_single_decision(
        self,
    ) -> None:
        preview = preview_cross_contract_claim_boundary(REPO_ROOT, PROPOSAL)

        self.assertEqual(
            preview["working_title"],
            "Continuity is not consistency: Cross-contract observability in LEED "
            "spot measurement chains",
        )
        self.assertEqual(
            preview["selected_thesis"],
            "combined_continuity_consistency_and_contract_dependent_observability",
        )
        self.assertEqual(preview["claim_count"], 7)
        self.assertEqual(
            preview["language_rule_concepts"],
            [
                "gaps",
                "recovery",
                "Cross-contract target inconsistency",
                "abstention",
                "possible false persistence",
            ],
        )
        self.assertEqual(preview["status"], "awaiting-author-checkpoint")

    def test_freeze_requires_an_explicit_author_identity_and_date(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "claim-boundary"
            copy_proposal(package)

            with self.assertRaisesRegex(
                CrossContractClaimBoundaryError,
                "explicit author checkpoint requires confirmed-by and confirmed-date",
            ):
                freeze_cross_contract_claim_boundary(
                    REPO_ROOT,
                    package,
                    confirmed_by="",
                    confirmed_date="",
                )

            self.assertFalse((package / "author_checkpoint.json").exists())

    def test_explicit_checkpoint_freezes_an_authenticated_decision(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "claim-boundary"
            copy_proposal(package)

            report = freeze_cross_contract_claim_boundary(
                REPO_ROOT,
                package,
                confirmed_by="test publication author",
                confirmed_date="2026-08-08",
            )
            checkpoint = json.loads(
                (package / "author_checkpoint.json").read_text(encoding="utf-8")
            )

            self.assertEqual(report["status"], "validated-author-approved-boundary")
            self.assertEqual(report["claim_count"], 7)
            self.assertEqual(checkpoint["status"], "confirmed")
            self.assertEqual(checkpoint["confirmed_by"], "test publication author")
            self.assertEqual(checkpoint["confirmed_date"], "2026-08-08")
            self.assertEqual(
                checkpoint["decision_record_sha256"],
                report["decision_record_sha256"],
            )
            self.assertEqual(
                validate_cross_contract_claim_boundary(REPO_ROOT, package), report
            )

    def test_frozen_package_rejects_a_second_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "claim-boundary"
            copy_proposal(package)
            freeze_cross_contract_claim_boundary(
                REPO_ROOT,
                package,
                confirmed_by="test publication author",
                confirmed_date="2026-08-08",
            )

            with self.assertRaisesRegex(
                CrossContractClaimBoundaryError,
                "cross-contract claim-boundary package is already frozen",
            ):
                freeze_cross_contract_claim_boundary(
                    REPO_ROOT,
                    package,
                    confirmed_by="different author",
                    confirmed_date="2026-08-09",
                )

    def test_validation_rejects_a_decision_changed_after_author_approval(self) -> None:
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "claim-boundary"
            copy_proposal(package)
            freeze_cross_contract_claim_boundary(
                REPO_ROOT,
                package,
                confirmed_by="test publication author",
                confirmed_date="2026-08-08",
            )
            with (package / "decision_record.json").open("ab") as handle:
                handle.write(b" ")

            with self.assertRaisesRegex(
                CrossContractClaimBoundaryError,
                "proposed cross-contract decision record drifted",
            ):
                validate_cross_contract_claim_boundary(REPO_ROOT, package)

    def test_cli_previews_the_exact_decision_digest(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/freeze_cross_contract_claim_boundary.py"),
                "--workspace-root",
                str(REPO_ROOT),
                "--package",
                str(PROPOSAL),
                "--preview",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        preview = json.loads(completed.stdout)
        self.assertEqual(
            preview["decision_record_sha256"],
            "0f3e4d727fdf66721fa76924728fcf67ae1443c6c09654395093b20434ed6c18",
        )
        self.assertEqual(preview["status"], "awaiting-author-checkpoint")


if __name__ == "__main__":
    unittest.main()
