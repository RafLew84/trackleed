"""Versioned JSON codec for immutable Track Analysis lifecycle state."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisResultStatus,
    AnalysisSourceState,
    AnalysisSourceStatus,
)
from trackleed.logic.workflows.gaussian_fit_recovery import (
    FitRecoveryConfig,
    FitValidation,
    GaussianFitterConfig,
    RecoveryAttempt,
    RecoveryAttemptStatus,
    RecoveryRecord,
    RecoverySelectionReason,
    RecoveryStatus,
    Rect,
)
from trackleed.logic.workflows.track_analysis import (
    FitFailure,
    FitFingerprint,
    FitObservation,
    FitProvenance,
    FitQuality,
    FitSet,
    FitStatus,
    GaussianErrors,
    GaussianFitResult,
    GaussianIntensity,
    GaussianParameters,
    GaussianWidths,
    PreprocessingConfig,
)
from trackleed.logic.workflows.track_geometry import (
    FrameGeometry,
    GeometryFailure,
    GeometryFingerprint,
    GeometryMethod,
    GeometryResult,
    GeometryStatus,
    GeometryUnavailableReason,
    ObservationGeometry,
    PatternCenter,
)


ANALYSIS_STATE_SCHEMA_VERSION = 1

_DATACLASS_TYPES = (
    ActiveAnalysisState,
    AnalysisSourceState,
    FitRecoveryConfig,
    FitValidation,
    GaussianFitterConfig,
    RecoveryAttempt,
    RecoveryRecord,
    Rect,
    FitFailure,
    FitFingerprint,
    FitObservation,
    FitProvenance,
    FitQuality,
    FitSet,
    GaussianErrors,
    GaussianFitResult,
    GaussianIntensity,
    GaussianParameters,
    GaussianWidths,
    PreprocessingConfig,
    FrameGeometry,
    GeometryFailure,
    GeometryFingerprint,
    GeometryResult,
    ObservationGeometry,
    PatternCenter,
)
_ENUM_TYPES = (
    AnalysisResultStatus,
    AnalysisSourceStatus,
    RecoveryAttemptStatus,
    RecoverySelectionReason,
    RecoveryStatus,
    FitStatus,
    GeometryMethod,
    GeometryStatus,
    GeometryUnavailableReason,
)
_DATACLASS_REGISTRY: dict[str, type[Any]] = {
    item.__name__: item for item in _DATACLASS_TYPES
}
_ENUM_REGISTRY = {item.__name__: item for item in _ENUM_TYPES}


def encode_analysis_state(state: ActiveAnalysisState) -> dict[str, object]:
    """Encode exactly one active fit/geometry publication into JSON values."""
    return {
        "schema_version": ANALYSIS_STATE_SCHEMA_VERSION,
        "state": _encode_value(state),
    }


def decode_analysis_state(payload: object) -> ActiveAnalysisState:
    """Decode a current analysis publication, rejecting unknown typed values."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid session: analysis_state must be an object")
    if payload.get("schema_version") != ANALYSIS_STATE_SCHEMA_VERSION:
        raise ValueError("Invalid session: unsupported analysis_state schema")
    state = _decode_value(payload.get("state"))
    if not isinstance(state, ActiveAnalysisState):
        raise ValueError("Invalid session: analysis_state root has the wrong type")
    return state


def _encode_value(value: object) -> object:
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "$value": value.value}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"$path": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": type(value).__name__,
            "$fields": {
                item.name: _encode_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode_value(item) for item in value]}
    raise TypeError(f"Analysis state contains unsupported value: {type(value)}")


def _decode_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict):
        raise ValueError("Invalid session: unsupported analysis_state value")
    if set(value) == {"$path"} and isinstance(value["$path"], str):
        return Path(value["$path"])
    if set(value) == {"$tuple"} and isinstance(value["$tuple"], list):
        return tuple(_decode_value(item) for item in value["$tuple"])
    if set(value) == {"$enum", "$value"}:
        enum_type = _ENUM_REGISTRY.get(value["$enum"])
        if enum_type is None:
            raise ValueError("Invalid session: unknown analysis_state enum")
        return enum_type(value["$value"])
    if set(value) == {"$type", "$fields"} and isinstance(value["$fields"], dict):
        data_type = _DATACLASS_REGISTRY.get(value["$type"])
        if data_type is None:
            raise ValueError("Invalid session: unknown analysis_state type")
        decoded_fields: dict[str, Any] = {
            key: _decode_value(item) for key, item in value["$fields"].items()
        }
        try:
            return data_type(**decoded_fields)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid session: malformed analysis_state value") from exc
    raise ValueError("Invalid session: malformed analysis_state value")
