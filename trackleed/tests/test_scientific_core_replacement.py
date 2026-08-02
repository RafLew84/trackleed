from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from trackleed.publication.scientific_core import (
    freeze_fingerprint,
    validate_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/replace_scientific_core.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


def _commit_all(workspace: Path, message: str) -> str:
    _git(workspace, "add", "--all")
    _git(workspace, "commit", "-m", message)
    return _git(workspace, "rev-parse", "HEAD")


def _build_authorized_fixture(workspace: Path) -> dict[str, object]:
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "fixture@example.invalid")
    _git(workspace, "config", "user.name", "Replacement Fixture")
    package = workspace / "manuscript/scientific_core"
    components = []
    for index in range(23):
        relative_path = f"manuscript/components/component-{index:02d}.txt"
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen component {index}\n", encoding="utf-8")
        components.append(
            {
                "id": f"component-{index:02d}",
                "path": relative_path,
                "roles": ["scientific_core"],
            }
        )
    contract_path = package / "fingerprint_contract.json"
    _write_json(
        contract_path,
        {
            "components": components,
            "required_roles": ["scientific_core"],
            "schema_version": 1,
            "status": "frozen",
        },
    )
    fingerprint_path = package / "fingerprint.json"
    frozen = freeze_fingerprint(workspace, contract_path, fingerprint_path)
    predecessor_bytes = fingerprint_path.read_bytes()
    _commit_all(workspace, "Freeze predecessor")

    changed_ids = [f"component-{index:02d}" for index in range(13)]
    for index in range(13):
        (workspace / components[index]["path"]).write_text(
            f"approved component {index}\n",
            encoding="utf-8",
        )
    predecessor = json.loads(predecessor_bytes)
    frozen_by_id = {item["id"]: item for item in predecessor["components"]}
    report_components = []
    for component in components:
        current_sha256 = _sha256(workspace / component["path"])
        frozen_sha256 = frozen_by_id[component["id"]]["sha256"]
        report_components.append(
            {
                **component,
                "current_sha256": current_sha256,
                "frozen_sha256": frozen_sha256,
                "status": (
                    "unchanged" if current_sha256 == frozen_sha256 else "changed"
                ),
            }
        )
    report_path = package / "pre_rebaseline_report.json"
    _write_json(
        report_path,
        {
            "component_summary": {"changed": 13, "total": 23, "unchanged": 10},
            "components": report_components,
            "predecessor": {
                "fingerprint_sha256": frozen.fingerprint_sha256,
                "validation_result": "expected-rejection",
            },
            "status": "expected-predecessor-rejection",
        },
    )
    implementation_commit = _commit_all(workspace, "Implement approved revision")

    decision_path = workspace / ".scratch/decision.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        "\n".join(
            (
                "# Accepted scientific-core revision",
                "",
                "Status: `accepted`",
                "Decision: `accepted`",
                f"Implementation commit: `{implementation_commit}`",
                f"Predecessor: `{frozen.fingerprint_sha256}`",
                f"Reviewed report: `{_sha256(report_path)}`",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    _commit_all(workspace, "Record decision")

    authorization_path = package / "replacement_authorization.json"
    _write_json(
        authorization_path,
        {
            "decision_record": {
                "path": ".scratch/decision.md",
                "sha256": _sha256(decision_path),
            },
            "expected_changed_component_ids": changed_ids,
            "expected_component_count": 23,
            "expected_predecessor_fingerprint_sha256": (
                frozen.fingerprint_sha256
            ),
            "implementation_commit": implementation_commit,
            "reason": "Approved scientific content revision",
            "reviewed_component_diff": {
                "path": "manuscript/scientific_core/pre_rebaseline_report.json",
                "sha256": _sha256(report_path),
            },
            "schema_version": "1.0",
            "status": "authorized",
        },
    )
    _commit_all(workspace, "Authorize replacement")
    return {
        "authorization_path": authorization_path,
        "changed_ids": changed_ids,
        "contract_path": contract_path,
        "decision_path": decision_path,
        "fingerprint_path": fingerprint_path,
        "implementation_commit": implementation_commit,
        "predecessor_bytes": predecessor_bytes,
        "predecessor_sha256": frozen.fingerprint_sha256,
    }


def _run_replacement(
    workspace: Path,
    fixture: dict[str, object],
    *,
    expected_predecessor: str | None = None,
    implementation_commit: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace-root",
            str(workspace),
            "--expected-predecessor",
            expected_predecessor or str(fixture["predecessor_sha256"]),
            "--decision-record",
            str(fixture["decision_path"]),
            "--implementation-commit",
            implementation_commit or str(fixture["implementation_commit"]),
            "--authorization",
            str(fixture["authorization_path"]),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class ScientificCoreReplacementCliTests(unittest.TestCase):
    def test_repository_authorization_pins_i10_and_expected_predecessor(self) -> None:
        authorization_path = (
            REPO_ROOT / "manuscript/scientific_core/replacement_authorization.json"
        )
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        decision = REPO_ROOT / authorization["decision_record"]["path"]
        reviewed = REPO_ROOT / authorization["reviewed_component_diff"]["path"]

        self.assertEqual(authorization["status"], "authorized")
        self.assertEqual(authorization["expected_component_count"], 23)
        self.assertEqual(
            authorization["expected_predecessor_fingerprint_sha256"],
            "859cd188d9fce23f062c9037ee5b231741c80c6df162d34014363bc2fa216eff",
        )
        self.assertEqual(
            authorization["implementation_commit"],
            "38a0e6aa668c4a7751fb6661e94af6d58bea92bc",
        )
        self.assertEqual(_sha256(decision), authorization["decision_record"]["sha256"])
        self.assertEqual(
            _sha256(reviewed),
            authorization["reviewed_component_diff"]["sha256"],
        )
        self.assertEqual(len(authorization["expected_changed_component_ids"]), 13)

    def test_command_requires_every_authorization_input(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--workspace-root", str(REPO_ROOT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--expected-predecessor", completed.stderr)
        self.assertIn("--decision-record", completed.stderr)
        self.assertIn("--implementation-commit", completed.stderr)
        self.assertIn("--authorization", completed.stderr)

    def test_authorized_command_replaces_atomically_and_preserves_history(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--workspace-root",
                    str(workspace),
                    "--expected-predecessor",
                    str(fixture["predecessor_sha256"]),
                    "--decision-record",
                    str(fixture["decision_path"]),
                    "--implementation-commit",
                    str(fixture["implementation_commit"]),
                    "--authorization",
                    str(fixture["authorization_path"]),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["status"], "replaced")
            self.assertEqual(output["component_count"], 23)
            self.assertNotEqual(
                output["successor_fingerprint_sha256"],
                fixture["predecessor_sha256"],
            )
            history = (
                workspace
                / "manuscript/scientific_core/history/fingerprints"
                / f"{fixture['predecessor_sha256']}.json"
            )
            self.assertEqual(history.read_bytes(), fixture["predecessor_bytes"])
            replacement = (
                workspace
                / "manuscript/scientific_core/history/replacements"
                / f"{output['successor_fingerprint_sha256']}.json"
            )
            record = json.loads(replacement.read_text(encoding="utf-8"))
            self.assertEqual(record["predecessor_fingerprint_sha256"], (
                fixture["predecessor_sha256"]
            ))
            self.assertEqual(record["successor_fingerprint_sha256"], (
                output["successor_fingerprint_sha256"]
            ))
            self.assertEqual(len(record["components"]), 23)
            self.assertEqual(
                {
                    item["id"]
                    for item in record["components"]
                    if item["status"] == "changed"
                },
                set(fixture["changed_ids"]),
            )
            successor = validate_fingerprint(
                workspace,
                fixture["contract_path"],
                fixture["fingerprint_path"],
            )
            self.assertEqual(
                successor.fingerprint_sha256,
                output["successor_fingerprint_sha256"],
            )

    def test_command_rejects_untracked_scientific_input_without_mutation(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)
            (workspace / "manuscript/unapproved-input.txt").write_text(
                "not reviewed\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--workspace-root",
                    str(workspace),
                    "--expected-predecessor",
                    str(fixture["predecessor_sha256"]),
                    "--decision-record",
                    str(fixture["decision_path"]),
                    "--implementation-commit",
                    str(fixture["implementation_commit"]),
                    "--authorization",
                    str(fixture["authorization_path"]),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("dirty or untracked", completed.stderr)
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )
            self.assertFalse(
                (workspace / "manuscript/scientific_core/history").exists()
            )

    def test_command_rejects_noncanonical_authorization(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)
            alternate = workspace / "alternate-authorization.json"
            alternate.write_bytes(fixture["authorization_path"].read_bytes())
            _commit_all(workspace, "Add unauthorized alternate")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--workspace-root",
                    str(workspace),
                    "--expected-predecessor",
                    str(fixture["predecessor_sha256"]),
                    "--decision-record",
                    str(fixture["decision_path"]),
                    "--implementation-commit",
                    str(fixture["implementation_commit"]),
                    "--authorization",
                    str(alternate),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("canonical replacement authorization", completed.stderr)
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )

    def test_command_rejects_wrong_predecessor_and_commit_without_mutation(
        self,
    ) -> None:
        for override in (
            {"expected_predecessor": "0" * 64},
            {"implementation_commit": "0" * 40},
        ):
            with self.subTest(override=override), TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                fixture = _build_authorized_fixture(workspace)

                completed = _run_replacement(workspace, fixture, **override)

                self.assertEqual(completed.returncode, 2)
                self.assertEqual(
                    fixture["fingerprint_path"].read_bytes(),
                    fixture["predecessor_bytes"],
                )
                self.assertFalse(
                    (workspace / "manuscript/scientific_core/history").exists()
                )

    def test_command_rejects_changed_decision_record_without_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)
            fixture["decision_path"].write_text(
                "Status: `rejected`\nDecision: `rejected`\n",
                encoding="utf-8",
            )

            completed = _run_replacement(workspace, fixture)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("Decision record", completed.stderr)
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )

    def test_command_rejects_incomplete_component_coverage(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)
            authorization_path = fixture["authorization_path"]
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            report_path = workspace / authorization["reviewed_component_diff"]["path"]
            old_report_sha256 = _sha256(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["components"].pop()
            _write_json(report_path, report)
            new_report_sha256 = _sha256(report_path)
            decision_path = fixture["decision_path"]
            decision_path.write_text(
                decision_path.read_text(encoding="utf-8").replace(
                    old_report_sha256,
                    new_report_sha256,
                ),
                encoding="utf-8",
                newline="\n",
            )
            authorization["reviewed_component_diff"]["sha256"] = new_report_sha256
            authorization["decision_record"]["sha256"] = _sha256(decision_path)
            _write_json(authorization_path, authorization)
            _commit_all(workspace, "Tamper with reviewed coverage")

            completed = _run_replacement(workspace, fixture)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("complete coverage of 23 components", completed.stderr)
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )

    def test_history_conflict_cannot_replace_predecessor(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)
            history = (
                workspace
                / "manuscript/scientific_core/history/fingerprints"
                / f"{fixture['predecessor_sha256']}.json"
            )
            history.parent.mkdir(parents=True)
            history.write_text("conflicting history\n", encoding="utf-8")

            completed = _run_replacement(workspace, fixture)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("history differs", completed.stderr)
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )

    def test_check_only_validates_without_creating_rebaseline_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            fixture = _build_authorized_fixture(workspace)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--workspace-root",
                    str(workspace),
                    "--expected-predecessor",
                    str(fixture["predecessor_sha256"]),
                    "--decision-record",
                    str(fixture["decision_path"]),
                    "--implementation-commit",
                    str(fixture["implementation_commit"]),
                    "--authorization",
                    str(fixture["authorization_path"]),
                    "--check-only",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "authorized")
            self.assertEqual(
                fixture["fingerprint_path"].read_bytes(),
                fixture["predecessor_bytes"],
            )
            self.assertFalse(
                (workspace / "manuscript/scientific_core/history").exists()
            )


if __name__ == "__main__":
    unittest.main()
