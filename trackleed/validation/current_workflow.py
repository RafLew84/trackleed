from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import sys
import unittest

from trackleed.validation import ValidationReport
from trackleed.validation.code_size import validate_code_size
from trackleed.validation.text_files import validate_changed_text
from trackleed.validation.generator_types import validate_generator_types


LOCAL_REFERENCE_TEST_MODULES = frozenset(
    {
        "test_content_revision_completion",
        "test_controlled_translation_evidence",
        "test_cross_contract_claim_boundary",
        "test_cross_contract_draft_package",
        "test_cross_contract_figures",
        "test_cross_contract_manuscript",
        "test_cross_contract_outcomes",
        "test_cross_contract_supplement",
        "test_experimental_metadata_contract",
        "test_pre_rebaseline_validation",
        "test_publication_exporter",
        "test_publication_line_endings",
        "test_publication_status_contract",
        "test_scientific_baseline_contract",
        "test_scientific_core_fingerprint",
        "test_scientific_core_replacement",
        "test_scientific_panel",
        "test_scientific_sensitivity",
        "test_scientific_sensitivity_evidence",
        "test_scientific_tracer",
        "test_viperleed_pilot_contract",
        "test_viperleed_principal_contract",
    }
)


@dataclass(frozen=True)
class CurrentWorkflowValidationReport(ValidationReport):
    checked_python_files: int
    tests_run: int
    test_output: str = ""


def validate_current_workflow(
    repo_root: Path, *, text_base: str | None = None,
) -> CurrentWorkflowValidationReport:
    root = repo_root.resolve()
    text_report = validate_changed_text(root, base=text_base)
    if not text_report.ok:
        return CurrentWorkflowValidationReport(
            checked_python_files=0, tests_run=0, errors=text_report.errors,
        )
    type_report = validate_generator_types(root)
    if not type_report.ok:
        return CurrentWorkflowValidationReport(
            checked_python_files=0, tests_run=0, errors=type_report.errors,
        )
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
        suite = _load_ci_test_suite(tests_root)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    except Exception as exc:
        _purge_test_modules(tests_root)
        errors.append(f"Full test suite is unavailable: {exc}")
        return CurrentWorkflowValidationReport(
            checked_python_files=code_size_report.checked_files,
            tests_run=0,
            errors=tuple(errors),
            test_output=stream.getvalue(),
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
        test_output=stream.getvalue(),
    )


def _load_ci_test_suite(tests_root: Path) -> unittest.TestSuite:
    if not tests_root.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {tests_root}")
    module_names = [
        path.stem
        for path in sorted(tests_root.glob("test_*.py"))
        if path.stem not in LOCAL_REFERENCE_TEST_MODULES
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
