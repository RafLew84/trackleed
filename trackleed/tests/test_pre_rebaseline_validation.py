from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.pre_rebaseline import validate_pre_rebaseline


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CHANGED_COMPONENTS = {
    "claim-boundary-manifest",
    "cpu-environment",
    "display-index",
    "display-inventory-builder",
    "display-inventory-contract",
    "display-semantic-inventory",
    "fingerprint-validator",
    "manuscript",
    "manuscript-validator",
    "reconstruction-entrypoint",
    "sensitivity-source-provenance",
    "supplementary-material",
    "tracer-authentication",
}


class PreRebaselineValidationTests(unittest.TestCase):
    def test_real_revision_reports_exact_diff_and_preserves_predecessor(self) -> None:
        fingerprint = REPO_ROOT / "manuscript/scientific_core/fingerprint.json"
        before = fingerprint.read_bytes()

        report = validate_pre_rebaseline(REPO_ROOT)

        self.assertEqual(report["status"], "expected-predecessor-rejection")
        self.assertEqual(report["predecessor"]["fingerprint_sha256"], (
            "859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff"
        ))
        self.assertTrue(report["predecessor"]["bytes_preserved"])
        self.assertEqual(
            report["predecessor"]["path"],
            "manuscript/scientific_core/history/fingerprints/"
            "859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff.json",
        )
        self.assertEqual(
            report["mutation_policy"],
            {
                "fingerprint_replaced": False,
                "manual_fingerprint_edit_allowed": False,
                "validation_only": True,
            },
        )
        self.assertEqual(fingerprint.read_bytes(), before)
        self.assertEqual(report["component_summary"], {
            "changed": 13,
            "total": 23,
            "unchanged": 10,
        })
        self.assertEqual(
            {
                component["id"]
                for component in report["components"]
                if component["status"] == "changed"
            },
            EXPECTED_CHANGED_COMPONENTS,
        )
        self.assertEqual(len(report["components"]), 23)
        self.assertEqual(
            set(report["validations"]),
            {
                "canonical_document_manifest",
                "manuscript_contract",
                "reader_facing_inventory",
            },
        )
        self.assertTrue(
            all(item["status"] == "passed" for item in report["validations"].values())
        )

    def test_expected_component_diff_is_fail_closed(self) -> None:
        source = REPO_ROOT / "manuscript/scientific_core/pre_rebaseline_contract.json"
        expectation = json.loads(source.read_text(encoding="utf-8"))
        expectation["expected_changed_component_ids"].remove("manuscript")
        with TemporaryDirectory() as temporary:
            changed_contract = Path(temporary) / "pre_rebaseline_contract.json"
            changed_contract.write_text(
                json.dumps(expectation), encoding="utf-8", newline="\n"
            )

            with self.assertRaisesRegex(ValueError, "component diff changed"):
                validate_pre_rebaseline(
                    REPO_ROOT,
                    expectation_path=changed_contract,
                )

    def test_predecessor_body_must_match_its_recorded_fingerprint(self) -> None:
        source = REPO_ROOT / "manuscript/scientific_core/fingerprint.json"
        predecessor = json.loads(source.read_text(encoding="utf-8"))
        predecessor["components"][0]["roles"].append("tampered-role")
        with TemporaryDirectory() as temporary:
            changed_fingerprint = Path(temporary) / "fingerprint.json"
            changed_fingerprint.write_text(
                json.dumps(predecessor), encoding="utf-8", newline="\n"
            )

            with self.assertRaisesRegex(ValueError, "integrity changed"):
                validate_pre_rebaseline(
                    REPO_ROOT,
                    fingerprint_path=changed_fingerprint,
                )

    def test_cli_writes_a_deterministic_review_scope_report(self) -> None:
        script = REPO_ROOT / "scripts/validate_pre_rebaseline.py"
        fingerprint = REPO_ROOT / "manuscript/scientific_core/fingerprint.json"
        predecessor = fingerprint.read_bytes()
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
            self.assertEqual(
                set(report["review_scope"]),
                {
                    "claim_evidence_map",
                    "denominators",
                    "displays",
                    "final_prose",
                    "limitations",
                },
            )
            self.assertTrue(all(report["review_scope"].values()))
            self.assertEqual(fingerprint.read_bytes(), predecessor)


if __name__ == "__main__":
    unittest.main()
