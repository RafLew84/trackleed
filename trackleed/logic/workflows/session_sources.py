"""Headless verification of external image sources restored from a session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path

import numpy as np

from trackleed.logic.active_analysis_state import (
    ActiveAnalysisState,
    AnalysisSourceState,
    AnalysisSourceStatus,
)


class SourceVerificationFailureReason(str, Enum):
    SOURCE_NOT_FOUND = "source_not_found"
    FINGERPRINT_UNAVAILABLE = "fingerprint_unavailable"
    PIXEL_FINGERPRINT_MISMATCH = "pixel_fingerprint_mismatch"


@dataclass(frozen=True)
class SourceVerificationFailure:
    frame_id: int | None
    is_background: bool
    reason: SourceVerificationFailureReason
    path: Path


@dataclass(frozen=True)
class SourceVerificationResult:
    state: ActiveAnalysisState
    failures: tuple[SourceVerificationFailure, ...]


def _never_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class SourceVerificationContext:
    image_reader: Callable[[Path], np.ndarray | None]
    relocation_root: Path | None = None
    is_cancelled: Callable[[], bool] = field(
        default=_never_cancelled,
        compare=False,
        repr=False,
    )


class SourceVerificationCancelled(RuntimeError):
    """Raised instead of returning partially verified source state."""


def verify_analysis_sources(
    state: ActiveAnalysisState,
    context: SourceVerificationContext,
) -> SourceVerificationResult:
    """Resolve and verify every non-available source without mutating input state."""
    verified_sources = []
    failures = []
    for source in state.sources:
        if context.is_cancelled():
            raise SourceVerificationCancelled("Analysis source verification cancelled")
        if source.status is AnalysisSourceStatus.AVAILABLE:
            verified_sources.append(source)
            continue
        verified, failure = _verify_source(source, context)
        verified_sources.append(verified)
        if failure is not None:
            failures.append(failure)
    if context.is_cancelled():
        raise SourceVerificationCancelled("Analysis source verification cancelled")
    return SourceVerificationResult(
        replace(state, sources=tuple(verified_sources)),
        tuple(failures),
    )


def _verify_source(
    source: AnalysisSourceState,
    context: SourceVerificationContext,
) -> tuple[AnalysisSourceState, SourceVerificationFailure | None]:
    candidate = _resolve_candidate(source, context.relocation_root)
    if candidate is None:
        return _failed_source(source, SourceVerificationFailureReason.SOURCE_NOT_FOUND)
    if source.pixel_sha256 is None:
        return _failed_source(
            replace(source, path=candidate),
            SourceVerificationFailureReason.FINGERPRINT_UNAVAILABLE,
        )
    pixels = context.image_reader(candidate)
    if pixels is None or not _pixels_match(source, pixels):
        return _failed_source(
            replace(source, path=candidate),
            SourceVerificationFailureReason.PIXEL_FINGERPRINT_MISMATCH,
        )
    return replace(
        source,
        path=candidate,
        status=AnalysisSourceStatus.AVAILABLE,
    ), None


def _resolve_candidate(
    source: AnalysisSourceState,
    relocation_root: Path | None,
) -> Path | None:
    candidates = [source.path]
    if relocation_root is not None:
        candidates.extend(
            (relocation_root / source.relative_path, relocation_root / source.path.name)
        )
    direct = next((candidate for candidate in candidates if candidate.is_file()), None)
    if direct is not None or relocation_root is None or not relocation_root.is_dir():
        return direct
    return next(iter(sorted(relocation_root.rglob(source.path.name))), None)


def _pixels_match(source: AnalysisSourceState, pixels: np.ndarray) -> bool:
    array = np.ascontiguousarray(np.asarray(pixels))
    if source.pixel_dtype is not None and str(array.dtype) != source.pixel_dtype:
        return False
    if source.pixel_shape is not None and tuple(array.shape) != source.pixel_shape:
        return False
    return sha256(array.tobytes(order="C")).hexdigest() == source.pixel_sha256


def _failed_source(
    source: AnalysisSourceState,
    reason: SourceVerificationFailureReason,
) -> tuple[AnalysisSourceState, SourceVerificationFailure]:
    unavailable = replace(source, status=AnalysisSourceStatus.SOURCE_UNAVAILABLE)
    return unavailable, SourceVerificationFailure(
        source.frame_id,
        source.is_background,
        reason,
        source.path,
    )
