"""Public CLI status is readable without starting another validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from trackleed.tests.test_current_workflow_validation import _initialize_validator_fixture


class ValidationProgressTests(unittest.TestCase):
    def test_completed_run_has_status_and_read_only_status_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            result = self.run_cli(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            statuses = list((root / ".scratch/validation").glob("*.status.json"))
            self.assertEqual(len(statuses), 1)
            state = json.loads(statuses[0].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "passed")
            self.assertEqual(state["last_completed_step"], "tests")
            self.assertEqual(state["tests_completed"], 1)
            self.assertEqual(state["tests_total"], 1)
            self.assertTrue(state["started_at"])
            self.assertTrue(state["finished_at"])
            self.assertTrue(Path(state["log_path"]).is_file())
            before = {p.name: p.read_bytes() for p in statuses[0].parent.iterdir()}
            for args in (("--status",), ("--status", str(statuses[0]))):
                read = self.run_cli(root, *args)
                self.assertEqual(read.returncode, 0, read.stderr)
                self.assertEqual(json.loads(read.stdout)["status"], "passed")
            after = {p.name: p.read_bytes() for p in statuses[0].parent.iterdir()}
            self.assertEqual(before, after)

    def test_running_test_exposes_live_status_and_log_before_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import time, unittest\nfrom pathlib import Path\n"
                "class Held(unittest.TestCase):\n"
                "    def test_a_fast(self): self.assertTrue(True)\n"
                "    def test_b_wait(self):\n"
                "        print('live checkpoint', flush=True)\n"
                "        Path('started').touch()\n"
                "        deadline = time.monotonic() + 30\n"
                "        while not Path('release').exists():\n"
                "            if time.monotonic() > deadline: raise TimeoutError('test release')\n"
                "            time.sleep(.02)\n", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "scripts/validate_current_workflow.py"],
                cwd=root, env={k: v for k, v in os.environ.items()
                               if k != "TRACKLEED_VALIDATION_BASE"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 30
                while not (root / "started").exists() and process.poll() is None:
                    if time.monotonic() > deadline:
                        self.fail("Validation did not reach the waiting test")
                    time.sleep(.02)
                self.assertIsNone(process.poll())
                read = self.run_cli(root, "--status")
                self.assertEqual(read.returncode, 0, read.stderr)
                state = json.loads(read.stdout)
                self.assertEqual(state["pid"], process.pid)
                self.assertEqual(state["status"], "running")
                self.assertEqual(state["stage"], "tests")
                self.assertEqual(state["last_completed_step"], "test_discovery")
                self.assertEqual(state["tests_total"], 2)
                self.assertEqual(state["tests_completed"], 1)
                self.assertTrue(state["last_completed_test"].endswith("test_a_fast"))
                log = Path(state["log_path"]).read_text(encoding="utf-8")
                self.assertIn("test_a_fast", log)
                self.assertIn("test_b_wait", log)
                self.assertIn("live checkpoint", log)
                self.assertIsNone(process.poll())
            finally:
                (root / "release").touch()
                stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stdout + stderr)

    def test_interrupted_run_records_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import unittest\nclass Interrupted(unittest.TestCase):\n"
                "    def test_interrupt(self): raise KeyboardInterrupt()\n",
                encoding="utf-8")
            result = self.run_cli(root)
            state = json.loads(self.run_cli(root, "--status").stdout)
            self.assertEqual(state["status"], "interrupted")
            self.assertIsNotNone(state["finished_at"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("interrupted", result.stderr)

    def test_failed_text_check_records_cause_before_test_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "broken.md").write_bytes(bytes([255]))
            result = self.run_cli(root)
            self.assertEqual(result.returncode, 1)
            state = json.loads(self.run_cli(root, "--status").stdout)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stage"], "changed_text")
            self.assertEqual(state["tests_completed"], 0)
            self.assertIsNone(state["tests_total"])
            self.assertIsNotNone(state["finished_at"])
            self.assertIn("broken.md", state["summary"])

    def test_status_without_history_does_not_start_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            result = self.run_cli(root, "--status")
            self.assertEqual(result.returncode, 1)
            self.assertIn("No validation status", result.stderr)
            self.assertFalse((root / ".scratch").exists())

    def test_status_write_failure_keeps_short_error_and_full_log_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import unittest\nfrom pathlib import Path\n"
                "class Obstructed(unittest.TestCase):\n"
                "    def test_obstruct_status(self):\n"
                "        path = next(Path('.scratch/validation').glob('*.status.json'))\n"
                "        path.unlink()\n        path.mkdir()\n", encoding="utf-8")
            result = self.run_cli(root)
            self.assertEqual(result.returncode, 1)
            self.assertLessEqual(len(result.stderr.splitlines()), 3)
            self.assertIn("Full log:", result.stderr)
            self.assertIn("progress", result.stderr)
            self.assertNotIn("validation passed", result.stdout + result.stderr)

    def test_failed_test_keeps_status_small_and_full_details_in_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _initialize_validator_fixture(root)
            (root / "trackleed/tests/test_synthetic.py").write_text(
                "import unittest\nclass Failed(unittest.TestCase):\n"
                "    def test_failure(self): self.fail('failure-detail-' * 1000)\n",
                encoding="utf-8")
            result = self.run_cli(root)
            self.assertEqual(result.returncode, 1)
            read = self.run_cli(root, "--status")
            self.assertLess(len(read.stdout), 2500)
            state = json.loads(read.stdout)
            self.assertEqual(state["status"], "failed")
            self.assertIn("Full test suite failed", state["summary"])
            self.assertIn("failure-detail-" * 1000,
                          Path(state["log_path"]).read_text(encoding="utf-8"))

    def run_cli(self, root, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/validate_current_workflow.py", *args],
            cwd=root, env={k: v for k, v in os.environ.items()
                           if k != "TRACKLEED_VALIDATION_BASE"},
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
