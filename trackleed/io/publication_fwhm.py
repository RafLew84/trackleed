"""FWHM domain values and adapters for stored TrackLEED fit results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Iterator, Mapping


FWHM_SIGMA_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


class FwhmSource(str, Enum):
    """Provenance of a published FWHM value."""

    STORED = "stored"
    CALCULATED_FROM_SIGMA = "calculated_from_sigma"
    MISSING = "missing"


@dataclass(frozen=True)
class FwhmMetrics:
    """FWHM values recovered exclusively from an existing fit record."""

    sigma_x: float | None = None
    sigma_y: float | None = None
    fwhm_x: float | None = None
    fwhm_x_error: float | None = None
    fwhm_y: float | None = None
    fwhm_y_error: float | None = None
    source: FwhmSource = FwhmSource.MISSING

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "FwhmMetrics":
        values = values or {}
        if "fit_ok" in values and not is_true(values.get("fit_ok")):
            return cls()
        sigma_x = _finite_float(values.get("sigma_x"))
        sigma_y = _finite_float(values.get("sigma_y"))
        stored_fwhm_x = _finite_float(values.get("fwhm_x"))
        stored_fwhm_y = _finite_float(values.get("fwhm_y"))
        fwhm_x = stored_fwhm_x
        fwhm_y = stored_fwhm_y
        if fwhm_x is None and sigma_x is not None:
            fwhm_x = FWHM_SIGMA_FACTOR * sigma_x
        if fwhm_y is None and sigma_y is not None:
            fwhm_y = FWHM_SIGMA_FACTOR * sigma_y

        fwhm_x_error = _finite_float(values.get("fwhm_x_error"))
        fwhm_y_error = _finite_float(values.get("fwhm_y_error"))
        if fwhm_x_error is None:
            sigma_x_error = _finite_float(values.get("sigma_x_error"))
            if sigma_x_error is not None:
                fwhm_x_error = FWHM_SIGMA_FACTOR * sigma_x_error
        if fwhm_y_error is None:
            sigma_y_error = _finite_float(values.get("sigma_y_error"))
            if sigma_y_error is not None:
                fwhm_y_error = FWHM_SIGMA_FACTOR * sigma_y_error

        if stored_fwhm_x is not None and stored_fwhm_y is not None:
            source = FwhmSource.STORED
        elif fwhm_x is not None and fwhm_y is not None:
            source = FwhmSource.CALCULATED_FROM_SIGMA
        else:
            source = FwhmSource.MISSING
            fwhm_x = None
            fwhm_x_error = None
            fwhm_y = None
            fwhm_y_error = None
        return cls(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            fwhm_x=fwhm_x,
            fwhm_x_error=fwhm_x_error,
            fwhm_y=fwhm_y,
            fwhm_y_error=fwhm_y_error,
            source=source,
        )

    @property
    def mean(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        return (self.fwhm_x + self.fwhm_y) / 2.0

    @property
    def geometric(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        if self.fwhm_x < 0 or self.fwhm_y < 0:
            return None
        return math.sqrt(self.fwhm_x * self.fwhm_y)

    @property
    def minor(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        return min(self.fwhm_x, self.fwhm_y)

    @property
    def major(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        return max(self.fwhm_x, self.fwhm_y)

    @property
    def anisotropy(self) -> float | None:
        if self.minor is None or self.major is None or self.minor <= 0:
            return None
        return self.major / self.minor

    @property
    def minor_error(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        if self.fwhm_x <= self.fwhm_y:
            return self.fwhm_x_error
        return self.fwhm_y_error

    @property
    def major_error(self) -> float | None:
        if self.fwhm_x is None or self.fwhm_y is None:
            return None
        if self.fwhm_x <= self.fwhm_y:
            return self.fwhm_y_error
        return self.fwhm_x_error

    @property
    def effective_error(self) -> float | None:
        if (
            self.geometric is None
            or self.fwhm_x is None
            or self.fwhm_y is None
            or self.fwhm_x <= 0
            or self.fwhm_y <= 0
            or self.fwhm_x_error is None
            or self.fwhm_y_error is None
        ):
            return None
        return (
            self.geometric
            * 0.5
            * math.hypot(
                self.fwhm_x_error / self.fwhm_x,
                self.fwhm_y_error / self.fwhm_y,
            )
        )

    @property
    def anisotropy_error(self) -> float | None:
        if (
            self.anisotropy is None
            or self.minor is None
            or self.major is None
            or self.minor <= 0
            or self.major <= 0
            or self.minor_error is None
            or self.major_error is None
        ):
            return None
        return self.anisotropy * math.hypot(
            self.major_error / self.major,
            self.minor_error / self.minor,
        )

    def as_csv_fields(self) -> dict[str, str]:
        return {
            "fwhm_x": _format_number(self.fwhm_x),
            "fwhm_x_error": _format_number(self.fwhm_x_error),
            "fwhm_y": _format_number(self.fwhm_y),
            "fwhm_y_error": _format_number(self.fwhm_y_error),
            "fwhm_mean": _format_number(self.mean),
            "fwhm_geometric": _format_number(self.geometric),
            "fwhm_minor": _format_number(self.minor),
            "fwhm_minor_error": _format_number(self.minor_error),
            "fwhm_major": _format_number(self.major),
            "fwhm_major_error": _format_number(self.major_error),
            "fwhm_effective": _format_number(self.geometric),
            "fwhm_effective_error": _format_number(self.effective_error),
            "fwhm_anisotropy": _format_number(self.anisotropy),
            "fwhm_anisotropy_error": _format_number(self.anisotropy_error),
            "fwhm_source": self.source.value,
        }

    def as_json_fields(self) -> dict[str, float | str | None]:
        return {
            "fwhm_x": self.fwhm_x,
            "fwhm_x_error": self.fwhm_x_error,
            "fwhm_y": self.fwhm_y,
            "fwhm_y_error": self.fwhm_y_error,
            "fwhm_mean": self.mean,
            "fwhm_geometric": self.geometric,
            "fwhm_minor": self.minor,
            "fwhm_minor_error": self.minor_error,
            "fwhm_major": self.major,
            "fwhm_major_error": self.major_error,
            "fwhm_effective": self.geometric,
            "fwhm_effective_error": self.effective_error,
            "fwhm_anisotropy": self.anisotropy,
            "fwhm_anisotropy_error": self.anisotropy_error,
            "fwhm_source": self.source.value,
        }


@dataclass(frozen=True)
class StoredFitRecord:
    """One mutable stored fit mapping plus its optional frame/track key."""

    key: tuple[int, int] | None
    fit_results: dict[str, object] | None
    accepted: bool


def iter_stored_fit_records(payload: dict[str, object]) -> Iterator[StoredFitRecord]:
    """Yield both positions-session and legacy tracking fit records."""

    positions_results = payload.get("positions_analysis_results")
    if isinstance(positions_results, dict):
        frames = positions_results.get("fit_points_by_frame")
        if isinstance(frames, dict):
            for frame_id, frame_fits in frames.items():
                if not isinstance(frame_fits, dict):
                    continue
                for track_id, fit_payload in frame_fits.items():
                    if not isinstance(fit_payload, dict):
                        continue
                    fit_results = fit_payload.get("fit_results")
                    yield StoredFitRecord(
                        key=fit_key(frame_id, track_id),
                        fit_results=fit_results if isinstance(fit_results, dict) else None,
                        accepted=is_true(fit_payload.get("fit_ok")),
                    )

    tracks = payload.get("tracks")
    if isinstance(tracks, list):
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_id = track.get("track_id")
            analysis_results = track.get("analysis_results")
            if not isinstance(analysis_results, dict):
                continue
            for frame_id, fit_results in analysis_results.items():
                is_mapping = isinstance(fit_results, dict)
                yield StoredFitRecord(
                    key=fit_key(frame_id, track_id),
                    fit_results=fit_results if is_mapping else None,
                    accepted=is_mapping
                    and (
                        "fit_ok" not in fit_results
                        or is_true(fit_results.get("fit_ok"))
                    ),
                )


def provenance_counts(sources: Iterable[FwhmSource]) -> dict[str, int]:
    """Return the stable manifest schema for FWHM provenance counts."""

    counts = Counter(sources)
    return {source.value: counts.get(source, 0) for source in FwhmSource}


def fit_key(frame_id: object, track_id: object) -> tuple[int, int] | None:
    try:
        return int(frame_id), int(track_id)
    except (TypeError, ValueError):
        return None


def is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float | None) -> str:
    return "" if value is None else str(value)
