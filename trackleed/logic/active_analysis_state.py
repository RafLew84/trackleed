"""Consistent lifecycle state for reusable Track Analysis publications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from trackleed.logic.workflows.track_analysis import FitSet
from trackleed.logic.workflows.track_geometry import GeometryResult


class AnalysisResultStatus(str, Enum):
    EMPTY = "empty"
    CURRENT = "current"
    LEGACY_UNVERIFIED = "legacy_unverified"


class AnalysisSourceStatus(str, Enum):
    AVAILABLE = "available"
    UNVERIFIED = "unverified"
    SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass(frozen=True)
class AnalysisSourceState:
    frame_id: int | None
    path: Path
    relative_path: Path
    pixel_sha256: str | None
    status: AnalysisSourceStatus
    is_background: bool = False
    pixel_dtype: str | None = None
    pixel_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ActiveAnalysisState:
    fit_set: FitSet | None = None
    geometry_result: GeometryResult | None = None
    requires_fit_validation: bool = False
    result_status: AnalysisResultStatus = AnalysisResultStatus.EMPTY
    sources: tuple[AnalysisSourceState, ...] = ()

    def __post_init__(self) -> None:
        if self.fit_set is not None and self.result_status is AnalysisResultStatus.EMPTY:
            object.__setattr__(self, "result_status", AnalysisResultStatus.CURRENT)
        if self.fit_set is None and (
            self.geometry_result is not None or self.requires_fit_validation
        ):
            raise ValueError("Analysis geometry or validation requires a FitSet")
        if (
            self.result_status is AnalysisResultStatus.CURRENT
            and self.fit_set is None
        ):
            raise ValueError("Current analysis state requires a FitSet")

    def invalidate_source(self) -> ActiveAnalysisState:
        """Retain reusable fits but require fingerprint validation before geometry."""
        if self.fit_set is None:
            return self
        return ActiveAnalysisState(
            self.fit_set,
            None,
            True,
            self.result_status,
            self.sources,
        )

    def invalidate_geometry(self) -> ActiveAnalysisState:
        """Retain fit validity while discarding only dependent geometry."""
        return ActiveAnalysisState(
            self.fit_set,
            None,
            self.requires_fit_validation,
            self.result_status,
            self.sources,
        )

    def publish(
        self,
        fit_set: FitSet,
        geometry_result: GeometryResult,
    ) -> ActiveAnalysisState:
        """Activate one complete compatible fit-and-geometry publication."""
        sources = _sources_from_fit_set(fit_set) or self.sources
        return ActiveAnalysisState(
            fit_set,
            geometry_result,
            False,
            AnalysisResultStatus.CURRENT,
            sources,
        )

    @property
    def can_recompute(self) -> bool:
        return all(
            source.status is not AnalysisSourceStatus.SOURCE_UNAVAILABLE
            for source in self.sources
        )

    def retain_observations(
        self,
        keys: set[tuple[int, int]],
    ) -> ActiveAnalysisState:
        """Drop fits outside the active track view and invalidate its geometry."""
        if self.fit_set is None:
            return self.invalidate_geometry()
        retained = tuple(
            item
            for item in self.fit_set.observations
            if (item.track_id, item.frame_id) in keys
        )
        if len(retained) == len(self.fit_set.observations):
            return self
        return ActiveAnalysisState(
            FitSet(retained),
            None,
            self.requires_fit_validation,
            self.result_status,
            self.sources,
        )

    def clear(self) -> ActiveAnalysisState:
        """Discard the complete publication when no analysis input remains."""
        return ActiveAnalysisState()


def _sources_from_fit_set(fit_set: FitSet) -> tuple[AnalysisSourceState, ...]:
    sources: dict[tuple[bool, int | None], AnalysisSourceState] = {}
    for observation in fit_set.observations:
        provenance = observation.provenance
        frame_source = AnalysisSourceState(
            observation.frame_id,
            provenance.source_path,
            Path(provenance.source_path.name),
            provenance.source_pixel_sha256,
            AnalysisSourceStatus.AVAILABLE,
            pixel_dtype=provenance.source_dtype,
            pixel_shape=provenance.source_shape,
        )
        sources[(False, observation.frame_id)] = frame_source
        if provenance.background_used and provenance.background_path is not None:
            sources[(True, None)] = AnalysisSourceState(
                None,
                provenance.background_path,
                Path(provenance.background_path.name),
                provenance.background_pixel_sha256,
                AnalysisSourceStatus.AVAILABLE,
                True,
            )
    return tuple(sources[key] for key in sorted(sources, key=lambda item: str(item)))
