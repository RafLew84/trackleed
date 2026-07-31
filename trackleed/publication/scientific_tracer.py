"""Publication-facing facade for the Scan A / pair 1 tracer."""

from trackleed.analysis.scientific_tracer import (
    TracerReport,
    TracerValidationError,
    evaluate_candidates,
    run_tracer,
    validate_tracer_output,
)

__all__ = [
    "TracerReport",
    "TracerValidationError",
    "evaluate_candidates",
    "run_tracer",
    "validate_tracer_output",
]
