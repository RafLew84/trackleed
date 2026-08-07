"""Publication facade for principal-scan cross-contract outcomes."""

from trackleed.analysis.cross_contract_outcomes import (
    CrossContractValidationError,
    build_cross_contract_outcomes,
    validate_cross_contract_outcomes,
)

__all__ = [
    "CrossContractValidationError",
    "build_cross_contract_outcomes",
    "validate_cross_contract_outcomes",
]
