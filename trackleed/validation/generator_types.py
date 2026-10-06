"""Required, isolated mypy check for the benchmark generator and its CLIs."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from trackleed.validation import ValidationReport


GENERATOR_TARGETS = (
    "trackleed/synthetic",
    "scripts/generate_tracking_benchmark.py",
    "scripts/export_segmentation_to_png.py",
)


def validate_generator_types(root: Path) -> ValidationReport:
    required = (*GENERATOR_TARGETS, "mypy-generator.ini")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return ValidationReport(errors=(
            "Generator type checking unavailable; missing: " + ", ".join(missing),
        ))
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "mypy",
             "--config-file", "mypy-generator.ini", "--no-pretty",
             "--show-error-codes", *GENERATOR_TARGETS],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ValidationReport(errors=(f"Generator type checking unavailable: {exc}",))
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()
        return ValidationReport(errors=(
            "Generator type checking failed (mypy). "
            "Install requirements-validation.txt if mypy is unavailable.\n" + detail,
        ))
    return ValidationReport(errors=())
