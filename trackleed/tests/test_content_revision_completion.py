from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator
import unittest

from trackleed.publication.content_revision import (
    ContentRevisionValidationError,
    inventory_placeholders,
    validate_content_revision,
    validate_page_level_acceptance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "manuscript/content_revision/contract.json"


@contextmanager
def changed_contract(
    mutation: Callable[[dict[str, Any]], None],
) -> Iterator[Path]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutation(contract)
    with TemporaryDirectory() as temporary:
        changed = Path(temporary) / "contract.json"
        changed.write_text(json.dumps(contract), encoding="utf-8", newline="\n")
        yield changed


class ContentRevisionCompletionTests(unittest.TestCase):
    def test_real_workspace_asserts_status_without_submission_readiness(self) -> None:
        report = validate_content_revision(REPO_ROOT)

        self.assertEqual(report["status"], "content-revision-complete")
        self.assertFalse(report["submission_ready"])
        self.assertEqual(len(report["gates"]), 7)
        self.assertTrue(all(gate["status"] == "passed" for gate in report["gates"]))
        self.assertEqual(
            report["successor_fingerprint"]["fingerprint_sha256"],
            "66b7c59994fe5369ed02f0c8c04e4f924823a52cc423ab02a66a84302b151a3e",
        )
        self.assertEqual(report["page_level_audit"]["rendition_count"], 4)
        self.assertEqual(report["page_level_audit"]["reviewed_page_count"], 200)
        self.assertEqual(
            report["successor_fingerprint"]["predecessor_fingerprint_sha256"],
            "859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff",
        )
        self.assertEqual(
            report["placeholder_inventory_scope"]["authoritative_sources"],
            [
                "manuscript/manuscript.md",
                "manuscript/claim-evidence-map.md",
                "manuscript/supplementary-material.md",
            ],
        )
        self.assertEqual(
            report["placeholder_summary"],
            {
                "AUTHOR": 4,
                "EXPERIMENT": 1,
                "RELEASE-AUTHORIZATION": 1,
                "SUBMISSION": 1,
                "total": 7,
            },
        )

    def test_placeholder_inventory_records_every_location_and_rejects_unknown(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "manuscript.md"
            source.write_text(
                "TODO[AUTHOR]: names\nTODO[EXPERIMENT]: description\n",
                encoding="utf-8",
                newline="\n",
            )
            records = inventory_placeholders(
                root,
                ["manuscript.md"],
                ["AUTHOR", "EXPERIMENT"],
            )
            self.assertEqual(
                [(item["category"], item["line"]) for item in records],
                [("AUTHOR", 1), ("EXPERIMENT", 2)],
            )

            source.write_text(
                "TODO[AUTHOR]: names\nTODO[TEXT]: unresolved science\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "Uncategorized placeholder",
            ):
                inventory_placeholders(root, ["manuscript.md"], ["AUTHOR"])

    def test_malformed_placeholder_is_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manuscript.md").write_text(
                "TODO[AUTHOR: names\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "Malformed placeholder",
            ):
                inventory_placeholders(root, ["manuscript.md"], ["AUTHOR"])

    def test_gate_evidence_hash_drift_is_fail_closed(self) -> None:
        def corrupt_evidence(contract: dict[str, Any]) -> None:
            contract["gates"][0]["evidence"][0]["sha256"] = "0" * 64

        with changed_contract(corrupt_evidence) as changed:
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "Evidence hash drift",
            ):
                validate_content_revision(REPO_ROOT, contract_path=changed)

    def test_all_seven_gates_are_required(self) -> None:
        with changed_contract(lambda contract: contract["gates"].pop()) as changed:
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "Normative gate set changed",
            ):
                validate_content_revision(REPO_ROOT, contract_path=changed)

    def test_submission_ready_cannot_be_asserted(self) -> None:
        def assert_submission_ready(contract: dict[str, Any]) -> None:
            contract["submission_ready"] = True

        with changed_contract(assert_submission_ready) as changed:
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "must not assert submission-ready",
            ):
                validate_content_revision(REPO_ROOT, contract_path=changed)

    def test_unaccepted_page_audit_is_rejected_before_page_reads(self) -> None:
        source = REPO_ROOT / "manuscript/publication_package/page_audit/manifest.json"
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["final_decision"] = "pending-hitl"
        with TemporaryDirectory() as temporary:
            changed = Path(temporary) / "page-audit.json"
            changed.write_text(
                json.dumps(manifest), encoding="utf-8", newline="\n"
            )
            with self.assertRaisesRegex(
                ContentRevisionValidationError,
                "Page-level render acceptance failed",
            ):
                validate_page_level_acceptance(REPO_ROOT, changed)

    def test_cli_writes_a_deterministic_report(self) -> None:
        script = REPO_ROOT / "scripts/assert_content_revision_complete.py"
        with TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            for output in (first, second):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--workspace-root",
                        str(REPO_ROOT),
                        "--output",
                        str(output),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            report = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "content-revision-complete")
            self.assertFalse(report["submission_ready"])


if __name__ == "__main__":
    unittest.main()
