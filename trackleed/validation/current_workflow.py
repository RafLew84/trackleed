from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import sys
import unittest

from trackleed.validation.code_size import validate_code_size


REFERENCE_ONLY_TEST_PREFIXES = (
    "test_content_revision",
    "test_controlled_translation",
    "test_cross_contract",
    "test_experimental_metadata",
    "test_pre_rebaseline",
    "test_publication",
    "test_scientific",
    "test_viperleed",
)


@dataclass(frozen=True)
class CurrentWorkflowValidationReport:
    checked_python_files: int
    tests_run: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(self.errors)


def validate_current_workflow(repo_root: Path) -> CurrentWorkflowValidationReport:
    root = repo_root.resolve()
    code_size_report = validate_code_size(root, root / "code_size_baseline.json")
    errors = list(code_size_report.errors)
    if errors:
        return CurrentWorkflowValidationReport(
            checked_python_files=code_size_report.checked_files,
            tests_run=0,
            errors=tuple(errors),
        )

    stream = io.StringIO()
    tests_root = root / "trackleed" / "tests"
    try:
        suite = _load_automatic_test_suite(tests_root)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    except Exception as exc:
        _purge_test_modules(tests_root)
        errors.append(f"Full test suite is unavailable: {exc}")
        return CurrentWorkflowValidationReport(
            checked_python_files=code_size_report.checked_files,
            tests_run=0,
            errors=tuple(errors),
        )
    _purge_test_modules(tests_root)

    if result.testsRun == 0:
        errors.append("Full test suite is unavailable: no tests were discovered.")
    if not result.wasSuccessful():
        errors.append(f"Full test suite failed.\n{stream.getvalue().rstrip()}")
    if result.skipped:
        skipped_names = ", ".join(str(test) for test, _reason in result.skipped)
        errors.append(f"Full test suite skipped mandatory tests: {skipped_names}.")
    if result.expectedFailures:
        expected_failure_names = ", ".join(
            str(test) for test, _traceback in result.expectedFailures
        )
        errors.append(
            "Full test suite expected failures are not allowed: "
            f"{expected_failure_names}."
        )
    return CurrentWorkflowValidationReport(
        checked_python_files=code_size_report.checked_files,
        tests_run=result.testsRun,
        errors=tuple(errors),
    )


def _load_automatic_test_suite(tests_root: Path) -> unittest.TestSuite:
    if not tests_root.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {tests_root}")
    module_names = [
        path.stem
        for path in sorted(tests_root.glob("test_*.py"))
        if not path.stem.startswith(REFERENCE_ONLY_TEST_PREFIXES)
    ]
    if not module_names:
        return unittest.TestSuite()
    sys.path.insert(0, str(tests_root))
    try:
        return unittest.defaultTestLoader.loadTestsFromNames(module_names)
    finally:
        sys.path.remove(str(tests_root))


def _purge_test_modules(tests_root: Path) -> None:
    resolved_tests_root = tests_root.resolve()
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            Path(module_file).resolve().relative_to(resolved_tests_root)
        except (OSError, ValueError):
            continue
        sys.modules.pop(module_name, None)
