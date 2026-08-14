"""Fail-closed validation support for the current workflow."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationReport:
    """Common success and diagnostic behavior for validation results."""

    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(self.errors)
