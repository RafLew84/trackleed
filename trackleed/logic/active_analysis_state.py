"""Consistent lifecycle state for reusable Track Analysis publications."""

from __future__ import annotations

from dataclasses import dataclass

from trackleed.logic.workflows.track_analysis import FitSet
from trackleed.logic.workflows.track_geometry import GeometryResult


@dataclass(frozen=True)
class ActiveAnalysisState:
    fit_set: FitSet | None = None
    geometry_result: GeometryResult | None = None
    requires_fit_validation: bool = False

    def __post_init__(self) -> None:
        if self.fit_set is None and (
            self.geometry_result is not None or self.requires_fit_validation
        ):
            raise ValueError("Analysis geometry or validation requires a FitSet")

    def invalidate_source(self) -> ActiveAnalysisState:
        """Retain reusable fits but require fingerprint validation before geometry."""
        if self.fit_set is None:
            return ActiveAnalysisState()
        return ActiveAnalysisState(self.fit_set, None, True)

    def invalidate_geometry(self) -> ActiveAnalysisState:
        """Retain fit validity while discarding only dependent geometry."""
        return ActiveAnalysisState(
            self.fit_set,
            None,
            self.requires_fit_validation,
        )

    def publish(
        self,
        fit_set: FitSet,
        geometry_result: GeometryResult,
    ) -> ActiveAnalysisState:
        """Activate one complete compatible fit-and-geometry publication."""
        return ActiveAnalysisState(fit_set, geometry_result, False)

    def clear(self) -> ActiveAnalysisState:
        """Discard the complete publication when no analysis input remains."""
        return ActiveAnalysisState()
