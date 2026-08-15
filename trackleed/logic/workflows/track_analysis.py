"""Headless, fingerprinted Gaussian fitting for Track Analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.util import img_as_float

from trackleed.analysis.peak_analyzer import fit_spot_to_gaussian


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PreprocessingConfig:
    grayscale_mode: str = "skimage_rgb2gray_v1"
    float_conversion: str = "skimage_img_as_float_v1"
    clamp_negative: bool = True


@dataclass(frozen=True)
class GaussianFitterConfig:
    fitter_id: str = "trackleed_gaussian_curve_fit_v1"
    center_preference_factor: float | None = None
    theta_bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class FitValidation:
    min_r_squared: float = 0.85
    min_amplitude: float = 0.05
    center_margin: float = 2.0
    min_sigma: float = 0.5


@dataclass(frozen=True)
class FitObservationInput:
    track_id: int
    frame_id: int
    source_path: Path
    roi: Rect
    subtract_background: bool = False
    background_path: Path | None = None
    energy: float | None = field(default=None, compare=False)
    temperature: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class FitProgress:
    completed: int
    total: int
    track_id: int
    frame_id: int


def _ignore_progress(_progress: FitProgress) -> None:
    return None


def _never_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class FitRequest:
    observations: tuple[FitObservationInput, ...]
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    fitter: GaussianFitterConfig = field(default_factory=GaussianFitterConfig)
    validation: FitValidation = field(default_factory=FitValidation)
    recovery_policy_version: str = "disabled-v1"
    previous: FitSet | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class FitRunContext:
    image_reader: Callable[[Path], np.ndarray | None]
    report_progress: Callable[[FitProgress], None] = field(
        default=_ignore_progress,
        compare=False,
        repr=False,
    )
    is_cancelled: Callable[[], bool] = field(
        default=_never_cancelled,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class FitFingerprint:
    sha256: str

    def __post_init__(self) -> None:
        is_hex = all(
            character in "0123456789abcdef"
            for character in self.sha256.lower()
        )
        if len(self.sha256) != 64 or not is_hex:
            raise ValueError("FitFingerprint must contain a SHA-256 hex digest")


class FitStatus(Enum):
    FITTED = "fitted"
    FAILED = "failed"


@dataclass(frozen=True)
class GaussianParameters:
    amplitude: float
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    theta: float
    offset: float


@dataclass(frozen=True)
class GaussianErrors:
    amplitude: float
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    theta: float
    offset: float


@dataclass(frozen=True)
class GaussianIntensity:
    volume: float
    volume_error: float


@dataclass(frozen=True)
class GaussianWidths:
    fwhm_x: float
    fwhm_y: float
    fwhm_x_error: float
    fwhm_y_error: float
    fwhm_profile_x: float
    fwhm_profile_y: float
    fwhm_profile_x_error: float
    fwhm_profile_y_error: float
    fwhm_minor: float
    fwhm_major: float
    fwhm_minor_error: float
    fwhm_major_error: float
    fwhm_effective: float
    fwhm_effective_error: float
    fwhm_anisotropy: float
    fwhm_anisotropy_error: float
    theta_major: float
    theta_major_error: float
    major_axis_angle_deg: float
    major_axis_angle_deg_error: float
    asymmetry: float


@dataclass(frozen=True)
class FitQuality:
    r_squared: float
    chi_squared_reduced: float


@dataclass(frozen=True)
class GaussianFitResult:
    parameters: GaussianParameters
    errors: GaussianErrors
    intensity: GaussianIntensity
    widths: GaussianWidths
    quality: FitQuality
    covariance_matrix: tuple[tuple[float, ...], ...]
    global_center_x: float
    global_center_y: float


@dataclass(frozen=True)
class FitFailure:
    code: str
    message: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FitProvenance:
    source_path: Path
    energy: float | None
    temperature: float | None
    source_dtype: str
    source_shape: tuple[int, ...]
    source_pixel_sha256: str
    original_roi: Rect
    background_used: bool
    background_path: Path | None
    background_pixel_sha256: str | None
    preprocessing: PreprocessingConfig
    fitter: GaussianFitterConfig
    validation: FitValidation
    recovery_policy_version: str


@dataclass(frozen=True)
class FitObservation:
    track_id: int
    frame_id: int
    fingerprint: FitFingerprint
    status: FitStatus
    provenance: FitProvenance
    result: GaussianFitResult | None = None
    failure: FitFailure | None = None

    def __post_init__(self) -> None:
        has_result = self.result is not None
        has_failure = self.failure is not None
        if has_result == has_failure:
            raise ValueError("A fit observation needs exactly one result or failure")


@dataclass(frozen=True)
class FitSet:
    observations: tuple[FitObservation, ...]

    def __post_init__(self) -> None:
        keys = tuple((item.track_id, item.frame_id) for item in self.observations)
        if len(set(keys)) != len(keys):
            raise ValueError("FitSet observation keys must be unique")

    def observation(self, track_id: int, frame_id: int) -> FitObservation | None:
        for item in self.observations:
            if item.track_id == track_id and item.frame_id == frame_id:
                return item
        return None


class FitError(RuntimeError):
    """Base error for a headless Track Analysis fit run."""


class FitPreflightError(FitError):
    """Raised before fitting when the complete request is invalid."""


class FitCancelled(FitError):
    """Raised instead of exposing an incomplete FitSet."""


class FitExecutionError(FitError):
    def __init__(self, track_id: int, frame_id: int, cause: Exception):
        self.track_id = track_id
        self.frame_id = frame_id
        super().__init__(f"Gaussian fit failed for track {track_id}, frame {frame_id}: {cause}")


def fit(
    request: FitRequest,
    context: FitRunContext | None = None,
) -> FitSet:
    """Validate, fit, and return a complete immutable result or raise."""
    if context is None:
        from trackleed.io.image_reader import load_tiff

        context = FitRunContext(image_reader=load_tiff)
    _validate_request(request)
    cached_by_fingerprint = {
        item.fingerprint: item
        for item in (() if request.previous is None else request.previous.observations)
    }
    completed: list[FitObservation] = []
    total = len(request.observations)
    loaded_source: tuple[Path, np.ndarray] | None = None
    loaded_backgrounds: dict[Path, np.ndarray] = {}
    for index, fit_input in enumerate(request.observations):
        if context.is_cancelled():
            raise FitCancelled("Track Analysis was cancelled")
        if loaded_source is None or loaded_source[0] != fit_input.source_path:
            loaded_source = (
                fit_input.source_path,
                _read_pixels(fit_input, fit_input.source_path, context),
            )
        pixels = loaded_source[1]
        background = None
        if fit_input.subtract_background and fit_input.background_path is not None:
            if fit_input.background_path not in loaded_backgrounds:
                loaded_backgrounds[fit_input.background_path] = _read_pixels(
                    fit_input,
                    fit_input.background_path,
                    context,
                )
            background = loaded_backgrounds[fit_input.background_path]
        _validate_decoded_input(fit_input, pixels, background)
        fingerprint = _fingerprint(fit_input, request, pixels, background)
        provenance = _provenance(fit_input, request, pixels, background)
        cached = cached_by_fingerprint.get(fingerprint)
        if cached is not None:
            observation = _reuse_cached_observation(
                cached, fit_input, provenance
            )
        else:
            observation = _fit_observation(
                fit_input,
                request,
                fingerprint,
                provenance,
                pixels,
                background,
            )
        completed.append(observation)
        context.report_progress(
            FitProgress(index + 1, total, fit_input.track_id, fit_input.frame_id)
        )
    if context.is_cancelled():
        raise FitCancelled("Track Analysis was cancelled")
    return FitSet(tuple(completed))


def _validate_request(request: FitRequest) -> None:
    if not request.recovery_policy_version.strip():
        raise FitPreflightError("recovery_policy_version must not be empty")
    validation_values = (
        request.validation.min_r_squared,
        request.validation.min_amplitude,
        request.validation.center_margin,
        request.validation.min_sigma,
    )
    if not all(np.isfinite(value) for value in validation_values):
        raise FitPreflightError("Fit validation thresholds must be finite")
    if request.validation.center_margin < 0 or request.validation.min_sigma < 0:
        raise FitPreflightError("Fit validation margins must be non-negative")
    preference = request.fitter.center_preference_factor
    if preference is not None and not np.isfinite(preference):
        raise FitPreflightError("center_preference_factor must be finite")
    if request.fitter.theta_bounds is not None:
        lower, upper = request.fitter.theta_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise FitPreflightError("theta_bounds must be strictly increasing")
    keys: set[tuple[int, int]] = set()
    for item in request.observations:
        key = (item.track_id, item.frame_id)
        if key in keys:
            raise FitPreflightError(f"Duplicate fit observation: {key}")
        keys.add(key)
        if not str(item.source_path):
            raise FitPreflightError(f"Source path required for observation {key}")
        if item.roi.width <= 0 or item.roi.height <= 0:
            raise FitPreflightError(f"ROI must be non-empty for observation {key}")
        if item.subtract_background and item.background_path is None:
            raise FitPreflightError(f"Background required for observation {key}")
    if request.preprocessing.grayscale_mode != "skimage_rgb2gray_v1":
        raise FitPreflightError("Unsupported grayscale preprocessing mode")
    if request.preprocessing.float_conversion != "skimage_img_as_float_v1":
        raise FitPreflightError("Unsupported float conversion")
    if request.fitter.fitter_id != "trackleed_gaussian_curve_fit_v1":
        raise FitPreflightError("Unsupported Gaussian fitter")


def _read_pixels(
    fit_input: FitObservationInput,
    path: Path,
    context: FitRunContext,
) -> np.ndarray:
    try:
        decoded = context.image_reader(path)
        pixels = np.asarray(decoded)
        if decoded is None or pixels.size == 0 or pixels.ndim not in (2, 3):
            raise ValueError(f"Could not decode source pixels from {path}")
        if pixels.dtype.kind not in "buif":
            raise ValueError(f"Decoded pixels from {path} are not numeric")
        return pixels
    except FitExecutionError:
        raise
    except Exception as cause:
        raise FitExecutionError(
            fit_input.track_id,
            fit_input.frame_id,
            cause,
        ) from cause


def _validate_decoded_input(
    fit_input: FitObservationInput,
    pixels: np.ndarray,
    background: np.ndarray | None,
) -> None:
    key = (fit_input.track_id, fit_input.frame_id)
    _validate_roi(fit_input.roi, pixels.shape, key)
    if background is not None and background.shape[:2] != pixels.shape[:2]:
        raise FitPreflightError(f"Background shape mismatch for observation {key}")


def _validate_roi(
    roi: Rect,
    image_shape: tuple[int, ...],
    key: tuple[int, int],
) -> None:
    if roi.width <= 0 or roi.height <= 0:
        raise FitPreflightError(f"ROI must be non-empty for observation {key}")
    height, width = image_shape[:2]
    if roi.x < 0 or roi.y < 0 or roi.x + roi.width > width or roi.y + roi.height > height:
        raise FitPreflightError(f"ROI is outside decoded pixels for observation {key}")


def _fingerprint(
    fit_input: FitObservationInput,
    request: FitRequest,
    pixels: np.ndarray,
    background: np.ndarray | None,
) -> FitFingerprint:
    digest = sha256()
    _update_array_digest(digest, "frame", pixels)
    canonical = {
        "roi": asdict(fit_input.roi),
        "subtract_background": fit_input.subtract_background,
        "preprocessing": asdict(request.preprocessing),
        "fitter": asdict(request.fitter),
        "validation": asdict(request.validation),
        "recovery_policy_version": request.recovery_policy_version,
    }
    digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())
    if fit_input.subtract_background:
        _update_array_digest(digest, "background", background)
    return FitFingerprint(digest.hexdigest())


def _update_array_digest(digest, label: str, values: np.ndarray | None) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"label": label, "dtype": array.dtype.str, "shape": array.shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _reuse_cached_observation(
    cached: FitObservation,
    fit_input: FitObservationInput,
    provenance: FitProvenance,
) -> FitObservation:
    if (
        cached.track_id == fit_input.track_id
        and cached.frame_id == fit_input.frame_id
        and cached.provenance == provenance
    ):
        return cached
    return replace(
        cached,
        track_id=fit_input.track_id,
        frame_id=fit_input.frame_id,
        provenance=provenance,
    )


def _fit_observation(
    fit_input: FitObservationInput,
    request: FitRequest,
    fingerprint: FitFingerprint,
    provenance: FitProvenance,
    pixels: np.ndarray,
    background: np.ndarray | None,
) -> FitObservation:
    roi_pixels = _prepare_roi(
        fit_input,
        request.preprocessing,
        pixels,
        background,
    )
    try:
        raw_result = fit_spot_to_gaussian(
            roi_pixels,
            center_preference_factor=request.fitter.center_preference_factor,
            theta_bounds=request.fitter.theta_bounds,
        )
    except Exception as cause:
        raise FitExecutionError(fit_input.track_id, fit_input.frame_id, cause) from cause
    if raw_result is None:
        return FitObservation(
            fit_input.track_id,
            fit_input.frame_id,
            fingerprint,
            FitStatus.FAILED,
            provenance,
            failure=FitFailure("fit_failed", "Gaussian optimizer returned no fit"),
        )
    reasons = _validation_failures(raw_result, roi_pixels.shape, request.validation)
    if reasons:
        return FitObservation(
            fit_input.track_id,
            fit_input.frame_id,
            fingerprint,
            FitStatus.FAILED,
            provenance,
            failure=FitFailure(
                "validation_failed",
                "Gaussian fit did not pass physical validation",
                reasons,
            ),
        )
    return FitObservation(
        fit_input.track_id,
        fit_input.frame_id,
        fingerprint,
        FitStatus.FITTED,
        provenance,
        result=_freeze_result(raw_result, fit_input.roi),
    )


def _prepare_roi(
    fit_input: FitObservationInput,
    config: PreprocessingConfig,
    pixels: np.ndarray,
    background_pixels: np.ndarray | None,
) -> np.ndarray:
    image = _grayscale_float(pixels, config)
    roi = fit_input.roi
    prepared = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()
    if fit_input.subtract_background:
        background = _grayscale_float(background_pixels, config)
        background_roi = background[
            roi.y : roi.y + roi.height,
            roi.x : roi.x + roi.width,
        ]
        prepared -= background_roi
        if config.clamp_negative:
            np.maximum(prepared, 0.0, out=prepared)
    return prepared


def _grayscale_float(values: np.ndarray | None, config: PreprocessingConfig) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        if config.grayscale_mode != "skimage_rgb2gray_v1":
            raise FitPreflightError("Unsupported grayscale preprocessing mode")
        array = rgb2gray(array[..., :3]) if array.shape[-1] >= 3 else array[..., 0]
    if config.float_conversion == "skimage_img_as_float_v1":
        return np.asarray(img_as_float(array), dtype=np.float64)
    raise FitPreflightError(f"Unsupported float conversion: {config.float_conversion}")


def _validation_failures(
    result: Mapping[str, Any],
    roi_shape: tuple[int, ...],
    config: FitValidation,
) -> tuple[str, ...]:
    height, width = roi_shape[:2]
    checks = (
        (float(result.get("r_squared", 0.0)) < config.min_r_squared, "r_squared"),
        (float(result.get("amplitude", 0.0)) < config.min_amplitude, "amplitude"),
        (not -config.center_margin < float(result.get("center_x", 0.0)) < width + config.center_margin, "center_x"),
        (not -config.center_margin < float(result.get("center_y", 0.0)) < height + config.center_margin, "center_y"),
        (not config.min_sigma < float(result.get("sigma_x", 0.0)) < width, "sigma_x"),
        (not config.min_sigma < float(result.get("sigma_y", 0.0)) < height, "sigma_y"),
    )
    return tuple(name for failed, name in checks if failed)


def _provenance(
    fit_input: FitObservationInput,
    request: FitRequest,
    pixels: np.ndarray,
    background: np.ndarray | None,
) -> FitProvenance:
    pixels = np.ascontiguousarray(pixels)
    pixel_digest = sha256(pixels.tobytes(order="C")).hexdigest()
    background_digest = (
        sha256(np.ascontiguousarray(background).tobytes(order="C")).hexdigest()
        if background is not None
        else None
    )
    return FitProvenance(
        source_path=fit_input.source_path,
        energy=fit_input.energy,
        temperature=fit_input.temperature,
        source_dtype=pixels.dtype.str,
        source_shape=tuple(int(value) for value in pixels.shape),
        source_pixel_sha256=pixel_digest,
        original_roi=fit_input.roi,
        background_used=fit_input.subtract_background,
        background_path=fit_input.background_path,
        background_pixel_sha256=background_digest,
        preprocessing=request.preprocessing,
        fitter=request.fitter,
        validation=request.validation,
        recovery_policy_version=request.recovery_policy_version,
    )


def _float_value(result: Mapping[str, Any], name: str) -> float:
    return float(result[name])


def _freeze_result(result: Mapping[str, Any], roi: Rect) -> GaussianFitResult:
    value = partial(_float_value, result)
    parameters = GaussianParameters(*(value(name) for name in (
        "amplitude", "center_x", "center_y", "sigma_x", "sigma_y", "theta", "offset"
    )))
    errors = GaussianErrors(*(value(f"{name}_error") for name in (
        "amplitude", "center_x", "center_y", "sigma_x", "sigma_y", "theta", "offset"
    )))
    width_names = tuple(GaussianWidths.__dataclass_fields__)
    widths = GaussianWidths(*(value(name) for name in width_names))
    covariance = np.asarray(result["covariance_matrix"], dtype=float)
    return GaussianFitResult(
        parameters=parameters,
        errors=errors,
        intensity=GaussianIntensity(value("volume"), value("volume_error")),
        widths=widths,
        quality=FitQuality(value("r_squared"), value("chi_squared_reduced")),
        covariance_matrix=tuple(tuple(float(cell) for cell in row) for row in covariance),
        global_center_x=float(roi.x) + parameters.center_x,
        global_center_y=float(roi.y) + parameters.center_y,
    )
