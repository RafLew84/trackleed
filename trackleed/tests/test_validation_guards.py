"""Public command checks with deliberately broken repository examples."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trackleed.tests.test_current_workflow_validation import _initialize_validator_fixture
from trackleed.tests.validation_test_support import run_git


class ValidationCommandGuardTests(unittest.TestCase):
    def test_missing_utf8_startup_mode_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            completed = subprocess.run(
                [sys.executable, "scripts/validate_current_workflow.py"],
                cwd=root, env={**os.environ, "PYTHONUTF8": "0"},
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("python -X utf8", completed.stderr)

    def test_same_command_sets_utf8_before_python_and_headless_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import os, sys, unittest\n"
                "class Startup(unittest.TestCase):\n"
                "    def test_startup(self):\n"
                "        self.assertEqual(sys.flags.utf8_mode, 1)\n"
                "        self.assertEqual(os.environ['QT_QPA_PLATFORM'], 'offscreen')\n"
                "        self.assertEqual(os.environ['MPLBACKEND'], 'Agg')\n",
                encoding="utf-8", newline="\n",
            )
            environment = {**os.environ, "PYTHONUTF8": "0"}
            environment.pop("TRACKLEED_VALIDATION_BASE", None)
            environment.pop("QT_QPA_PLATFORM", None)
            environment.pop("MPLBACKEND", None)
            completed = self.run_validator(root, environment)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_corrupt_examples_fail_with_filename_and_reason(self):
        examples = (
            ("notes.md", bytes([255]), "UTF-8"),
            ("notes.md", b"first\r\r\nsecond\r\r\n", "CRCRLF"),
            ("notes.md", b"first\rsecond\r", "bare CR"),
            ("notes.md", b"first\r\nsecond\n", "mixed"),
            ("trackleed/synthetic/scene.py", b'count: int = "wrong"\n', "Incompatible types"),
        )
        for filename, content, reason in examples:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _initialize_validator_fixture(root)
                (root / filename).write_bytes(content)
                completed = self.run_validator(root)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(filename, completed.stderr.replace("\\", "/"))
                self.assertIn(reason, completed.stderr)
                self.assertEqual((root / filename).read_bytes(), content)

    def test_ci_base_includes_bad_file_from_an_earlier_commit_in_the_push(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            base = run_git(root, "rev-parse", "HEAD").stdout.strip()
            (root / "earlier.md").write_bytes(bytes([255]))
            run_git(root, "add", "earlier.md")
            run_git(root, "commit", "-m", "bad text")
            (root / "latest.md").write_text("valid\n", encoding="utf-8")
            run_git(root, "add", "latest.md")
            run_git(root, "commit", "-m", "later change")
            completed = self.run_validator(
                root, {**os.environ, "TRACKLEED_VALIDATION_BASE": base},
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("earlier.md", completed.stderr)
        self.assertIn("UTF-8", completed.stderr)

    def run_validator(self, root, environment=None):
        if environment is None:
            environment = {k: v for k, v in os.environ.items()
                           if k != "TRACKLEED_VALIDATION_BASE"}
        return subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/validate_current_workflow.py"],
            cwd=root, env=environment, capture_output=True, text=True,
            encoding="utf-8", check=False,
        )
