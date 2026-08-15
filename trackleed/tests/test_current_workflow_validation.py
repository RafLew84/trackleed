from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from trackleed.tests.validation_test_support import (
    initialize_repository,
    run_git,
    write_manifest,
)
from trackleed.validation.current_workflow import validate_current_workflow


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CurrentWorkflowValidationTests(unittest.TestCase):
    def test_public_validator_runs_code_size_and_synthetic_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_source = (
                "import unittest\n\n"
                "class SyntheticTest(unittest.TestCase):\n"
                "    def test_known_value(self):\n"
                "        self.assertEqual(2 + 2, 4)\n"
            )
            initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            report = validate_current_workflow(root)

        self.assertTrue(report.ok, report.format_errors())
        self.assertEqual(report.tests_run, 1)

    def test_public_validator_runs_every_existing_test_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initialize_repository(
                root,
                {
                    "trackleed/tests/test_synthetic.py": (
                        "import unittest\n\n"
                        "class SyntheticTest(unittest.TestCase):\n"
                        "    def test_first(self):\n"
                        "        self.assertTrue(True)\n"
                    ),
                    "trackleed/tests/test_scientific_synthetic.py": (
                        "import unittest\n\n"
                        "class ScientificSyntheticTest(unittest.TestCase):\n"
                        "    def test_second(self):\n"
                        "        self.assertTrue(True)\n"
                    ),
                },
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            report = validate_current_workflow(root)

        self.assertTrue(report.ok, report.format_errors())
        self.assertEqual(report.tests_run, 2)

    def test_public_validator_keeps_local_reference_modules_out_of_ci_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initialize_repository(
                root,
                {
                    "trackleed/tests/test_synthetic.py": (
                        "import unittest\n\n"
                        "class SyntheticTest(unittest.TestCase):\n"
                        "    def test_ci_behavior(self):\n"
                        "        self.assertTrue(True)\n"
                    ),
                    "trackleed/tests/test_scientific_panel.py": (
                        "import unittest\n\n"
                        "class LocalReferenceTest(unittest.TestCase):\n"
                        "    def test_requires_local_data(self):\n"
                        "        self.fail('local reference data is unavailable')\n"
                    ),
                },
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            report = validate_current_workflow(root)

        self.assertTrue(report.ok, report.format_errors())
        self.assertEqual(report.tests_run, 1)

    def test_expected_failure_is_rejected_as_a_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_source = (
                "import unittest\n\n"
                "class SyntheticTest(unittest.TestCase):\n"
                "    @unittest.expectedFailure\n"
                "    def test_known_failure(self):\n"
                "        self.fail('known red test')\n"
            )
            initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            report = validate_current_workflow(root)

        self.assertFalse(report.ok)
        self.assertIn("expected failures are not allowed", report.format_errors())

    def test_skipped_test_is_rejected_as_an_unavailable_mandatory_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_source = (
                "import unittest\n\n"
                "class SyntheticTest(unittest.TestCase):\n"
                "    @unittest.skip('dependency unavailable')\n"
                "    def test_required_dependency(self):\n"
                "        pass\n"
            )
            initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            report = validate_current_workflow(root)

        self.assertFalse(report.ok)
        self.assertIn("skipped mandatory tests", report.format_errors())

    def test_exact_cli_entry_point_validates_a_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            copied_paths = (
                "scripts/validate_current_workflow.py",
                "trackleed/__init__.py",
                "trackleed/validation/__init__.py",
                "trackleed/validation/code_size.py",
                "trackleed/validation/current_workflow.py",
            )
            for relative_path in copied_paths:
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPOSITORY_ROOT / relative_path, destination)
            initialize_repository(
                root,
                {
                    "trackleed/tests/test_synthetic.py": (
                        "import unittest\n\n"
                        "class SyntheticTest(unittest.TestCase):\n"
                        "    def test_passes(self):\n"
                        "        self.assertTrue(True)\n"
                    )
                },
            )
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            write_manifest(root, fixed_point, {})

            completed = subprocess.run(
                [sys.executable, "scripts/validate_current_workflow.py"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Current workflow validation passed", completed.stdout)

    def test_reference_cli_fails_closed_for_missing_explicit_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _initialize_validator_fixture(root)
            missing = root / "local-reference.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_current_workflow.py",
                    "--reference-manifest",
                    str(missing),
                    "--reference-report",
                    str(root / "report.json"),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("reference manifest does not exist", completed.stderr)

    def test_reference_cli_requires_an_explicit_local_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _initialize_validator_fixture(root)
            manifest = root / "local-reference.json"
            manifest.write_text("{}", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_current_workflow.py",
                    "--reference-manifest",
                    str(manifest),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("--reference-report is required", completed.stderr)

    def test_reference_cli_propagates_a_reference_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _initialize_validator_fixture(root, include_reference=True)
            manifest = root / "local-reference.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reference_root": ".",
                        "golden_approval": {"status": "pending"},
                        "benchmark": {},
                        "series": [],
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_current_workflow.py",
                    "--reference-manifest",
                    str(manifest),
                    "--reference-report",
                    str(root / ".scratch" / "report.json"),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("manually_approved", completed.stderr)


def _initialize_validator_fixture(
    root: Path,
    *,
    include_reference: bool = False,
) -> None:
    copied_paths = [
        "scripts/validate_current_workflow.py",
        "trackleed/__init__.py",
        "trackleed/validation/__init__.py",
        "trackleed/validation/code_size.py",
        "trackleed/validation/current_workflow.py",
    ]
    if include_reference:
        copied_paths.append("trackleed/validation/reference_workflow.py")
    for relative_path in copied_paths:
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative_path, destination)
    initialize_repository(
        root,
        {
            "trackleed/tests/test_synthetic.py": (
                "import unittest\n\n"
                "class SyntheticTest(unittest.TestCase):\n"
                "    def test_passes(self):\n"
                "        self.assertTrue(True)\n"
            )
        },
    )
    fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
    write_manifest(root, fixed_point, {})

if __name__ == "__main__":
    unittest.main()
