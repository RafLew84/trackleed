"""Console summaries and complete logs at the public validation command."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trackleed.tests.test_current_workflow_validation import _initialize_validator_fixture


class ValidationLoggingTests(unittest.TestCase):
    def test_success_keeps_noisy_output_and_test_transcript_in_unique_utf8_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import os, subprocess, sys, unittest\n"
                "class Noisy(unittest.TestCase):\n"
                "    def test_output(self):\n"
                "        print('pełny wynik żółć\\n' * 200)\n"
                "        print('stderr detail', file=sys.stderr)\n"
                "        os.write(2, b'native diagnostic\\n')\n"
                "        subprocess.run([sys.executable, '-c', "
                "'print(123456789)'], check=True)\n",
                encoding="utf-8", newline="\n",
            )
            paths = []
            for _ in range(2):
                completed = self.run_validator(root)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Current workflow validation passed", completed.stdout)
                self.assertEqual(completed.stderr, "")
                log_path = self.log_path(completed.stdout)
                self.assertTrue(log_path.is_absolute())
                detail = log_path.read_text(encoding="utf-8")
                for expected in ("pełny wynik żółć", "stderr detail",
                                 "native diagnostic", "123456789",
                                 "test_output", "Ran 1 test", "OK"):
                    self.assertIn(expected, detail)
                    if expected != "OK":
                        self.assertNotIn(expected, completed.stdout)
                self.assertLessEqual(len(completed.stdout.splitlines()), 3)
                paths.append(log_path)
            self.assertNotEqual(*paths)
            self.assertTrue(all(path.is_file() for path in paths))


    def test_failed_test_has_a_short_error_and_complete_traceback_in_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import unittest\n"
                "class Broken(unittest.TestCase):\n"
                "    def test_failure(self):\n"
                "        print('diagnostic detail\\n' * 200)\n"
                "        self.fail('deliberate regression żółć')\n",
                encoding="utf-8", newline="\n",
            )
            completed = self.run_validator(root)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout, "")
            self.assertIn("Full test suite failed", completed.stderr)
            self.assertLessEqual(len(completed.stderr.splitlines()), 3)
            self.assertNotIn("diagnostic detail", completed.stderr)
            detail = self.log_path(completed.stderr).read_text(encoding="utf-8")
            for expected in ("test_failure", "Traceback", "deliberate regression żółć",
                             "diagnostic detail", "FAILED (failures=1)"):
                self.assertIn(expected, detail)

    def test_unwritable_log_location_fails_before_running_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / ".scratch").mkdir()
            (root / ".scratch/validation").write_text("occupied", encoding="utf-8")
            completed = self.run_validator(root)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("Cannot create full validation log", completed.stderr)
            self.assertNotIn("validation passed", completed.stdout + completed.stderr)
            self.assertNotIn("Full log:", completed.stdout + completed.stderr)

    def run_validator(self, root):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/validate_current_workflow.py"],
            cwd=root,
            env={k: v for k, v in os.environ.items()
                 if k != "TRACKLEED_VALIDATION_BASE"},
            capture_output=True, text=True, encoding="utf-8", check=False,
        )

    def log_path(self, output):
        lines = [line for line in output.splitlines() if line.startswith("Full log: ")]
        self.assertEqual(len(lines), 1, output)
        return Path(lines[0].removeprefix("Full log: "))
