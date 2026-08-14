from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from trackleed.tests.validation_test_support import (
    initialize_repository,
    run_git,
    write_manifest,
)
from trackleed.validation.code_size import validate_code_size


class CodeSizeValidationTests(unittest.TestCase):
    _initialize_repository = staticmethod(initialize_repository)
    _git = staticmethod(run_git)
    _write_manifest = staticmethod(write_manifest)

    def test_small_repository_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initialize_repository(root, {"module.py": "def answer():\n    return 42\n"})
            fixed_point = run_git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = write_manifest(root, fixed_point, {})

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())
        self.assertEqual(report.checked_files, 1)

    def test_parse_error_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"broken.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            (root / "broken.py").write_text("def unfinished(\n", encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("broken.py", report.format_errors())
        self.assertIn("cannot be parsed", report.format_errors())

    def test_new_production_file_counts_nested_and_async_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            nested = "".join(
                f"    def nested_{index}():\n        return {index}\n" for index in range(50)
            )
            (root / "too_many.py").write_text(
                f"async def outer():\n{nested}",
                encoding="utf-8",
            )

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("too_many.py has 51 definitions; limit is 50", report.format_errors())

    def test_new_production_file_rejects_more_than_1000_physical_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            (root / "too_long.py").write_text("# line\n" * 1001, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("too_long.py has 1001 lines; limit is 1000", report.format_errors())

    def test_test_files_use_their_separate_line_and_definition_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            definitions = "".join(
                f"def test_case_{index}():\n    return {index}\n" for index in range(60)
            )
            padding = "# test padding\n" * (1200 - len(definitions.splitlines()))
            test_path = root / "tests" / "test_large_suite.py"
            test_path.parent.mkdir()
            test_path.write_text(definitions + padding, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())

    def test_test_file_rejects_more_than_1500_lines_or_75_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            definitions = "".join(
                f"def test_case_{index}():\n    return {index}\n" for index in range(76)
            )
            padding = "# test padding\n" * (1501 - len(definitions.splitlines()))
            test_path = root / "tests" / "test_oversized_suite.py"
            test_path.parent.mkdir()
            test_path.write_text(definitions + padding, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("has 1501 lines; limit is 1500", report.format_errors())
        self.assertIn("has 76 definitions; limit is 75", report.format_errors())

    def test_new_async_method_over_80_lines_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            body = "".join(f"        value_{index} = {index}\n" for index in range(80))
            (root / "worker.py").write_text(
                f"class Worker:\n    async def run(self):\n{body}",
                encoding="utf-8",
            )

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("worker.py:Worker.run has 81 lines; limit is 80", report.format_errors())

    def test_only_declared_code_categories_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            for directory in ("generated", "vendored", "frozen", "third_party"):
                path = root / directory / "invalid.py"
                path.parent.mkdir()
                path.write_text("def unfinished(\n", encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())
        self.assertEqual(report.checked_files, 1)

    def test_oversized_fixed_point_file_requires_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# legacy\n" * 1001})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("legacy.py is oversized at the fixed point but has no manifest entry", report.format_errors())

    def test_unchanged_legacy_file_is_allowed_by_its_manifest_ratchet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# legacy\n" * 1001})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1001,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())

    def test_legacy_file_cannot_grow_in_lines_or_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = "".join(
                f"def legacy_{index}():\n    return {index}\n" for index in range(51)
            )
            self._initialize_repository(root, {"legacy.py": source})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 102,
                        "definitions": 51,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text(
                source + "def added():\n    return None\n",
                encoding="utf-8",
            )

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("legacy.py grew from 102 to 104 lines", report.format_errors())
        self.assertIn("legacy.py grew from 51 to 52 definitions", report.format_errors())

    def test_manifest_entry_may_be_removed_after_file_falls_below_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# legacy\n" * 1001})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            (root / "legacy.py").write_text("# reduced\n" * 1000, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())

    def test_unchanged_oversized_legacy_function_uses_its_own_ratchet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = "def legacy():\n" + "    value = 1\n" * 80
            self._initialize_repository(root, {"legacy.py": source})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 81,
                        "definitions": 1,
                        "oversized_definitions": {"legacy": 81},
                    }
                },
            )

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())

    def test_function_size_includes_decorators(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            body = "".join(f"    value_{index} = {index}\n" for index in range(79))
            (root / "decorated.py").write_text(
                f"decorator = lambda function: function\n@decorator\ndef long():\n{body}",
                encoding="utf-8",
            )

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("decorated.py:long has 81 lines; limit is 80", report.format_errors())

    def test_manifest_cannot_raise_a_ratchet_above_the_fixed_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# legacy\n" * 1001})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1002,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text("# grown\n" * 1002, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn(
            "legacy.py manifest line ratchet 1002 exceeds fixed-point value 1001",
            report.format_errors(),
        )

    def test_manifest_ratchet_can_be_lowered_while_file_remains_oversized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# baseline\n" * 1005})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1002,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text("# reduced\n" * 1002, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertTrue(report.ok, report.format_errors())

    def test_recorded_fixed_point_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "record ratchet")
            self._git(root, "commit", "--allow-empty", "-m", "later change")
            replacement = self._git(root, "rev-parse", "HEAD").stdout.strip()
            self._write_manifest(root, replacement, {})

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("fixed point changed", report.format_errors())

    def test_manifest_ratchet_cannot_be_raised_after_prior_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# baseline\n" * 1005})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1002,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text("# reduced\n" * 1002, encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "lower ratchet")
            self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1003,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text("# regrown\n" * 1003, encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("raised from 1002 to 1003", report.format_errors())

    def test_malformed_committed_manifest_history_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = root / "code_size_baseline.json"
            manifest_path.write_text("[]", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "malformed ratchet")
            self._write_manifest(root, fixed_point, {})

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("Code-size manifest at", report.format_errors())
        self.assertIn("root must be an object", report.format_errors())

    def test_manifest_cannot_be_reset_after_two_commits_without_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"legacy.py": "# baseline\n" * 1005})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(
                root,
                fixed_point,
                {
                    "legacy.py": {
                        "lines": 1002,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            (root / "legacy.py").write_text("# reduced\n" * 1002, encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "record lower ratchet")
            manifest_path.unlink()
            self._git(root, "add", "--update")
            self._git(root, "commit", "-m", "remove ratchet")
            self._git(root, "commit", "--allow-empty", "-m", "unrelated change")
            replacement = self._git(root, "rev-parse", "HEAD").stdout.strip()
            self._write_manifest(
                root,
                replacement,
                {
                    "legacy.py": {
                        "lines": 1002,
                        "definitions": 0,
                        "oversized_definitions": {},
                    }
                },
            )
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "reset ratchet")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("fixed point changed", report.format_errors())

    def test_invalid_fixed_point_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            manifest_path = self._write_manifest(root, "not-a-commit", {})

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("Cannot read fixed point not-a-commit", report.format_errors())

    def test_fixed_point_parse_error_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"broken.py": "def unfinished(\n"})
            fixed_point = self._git(root, "rev-parse", "HEAD").stdout.strip()
            manifest_path = self._write_manifest(root, fixed_point, {})
            (root / "broken.py").write_text("value = 1\n", encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertIn("broken.py cannot be parsed at fixed point", report.format_errors())

    def test_malformed_manifest_fails_closed_without_running_the_checker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._initialize_repository(root, {"module.py": "value = 1\n"})
            manifest_path = root / "code_size_baseline.json"
            manifest_path.write_text("[]", encoding="utf-8")

            report = validate_code_size(root, manifest_path)

        self.assertFalse(report.ok)
        self.assertEqual(report.checked_files, 0)
        self.assertIn("manifest root must be an object", report.format_errors())

if __name__ == "__main__":
    unittest.main()
