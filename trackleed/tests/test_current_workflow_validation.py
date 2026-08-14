from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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
            self._initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            (root / "code_size_baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixed_point": fixed_point,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )

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
            self._initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            (root / "code_size_baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixed_point": fixed_point,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )

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
            self._initialize_repository(
                root,
                {"trackleed/tests/test_synthetic.py": test_source},
            )
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            (root / "code_size_baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixed_point": fixed_point,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )

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
            self._initialize_repository(
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
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            (root / "code_size_baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixed_point": fixed_point,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "scripts/validate_current_workflow.py"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Current workflow validation passed", completed.stdout)

    def _initialize_repository(self, root: Path, files: dict[str, str]) -> None:
        self._git(root, "init")
        self._git(root, "config", "user.email", "validation@example.invalid")
        self._git(root, "config", "user.name", "Validation Test")
        for relative_path, source in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        self._git(root, "add", ".")
        self._git(root, "commit", "-m", "baseline")

    def _git(self, root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
