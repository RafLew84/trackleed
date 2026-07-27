"""Unambiguous width metrics for rotated 2D Gaussian fits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def canonicalize_axial_angle(angle: float) -> float:
    """Return an equivalent pi-periodic angle in [-pi/2, pi/2)."""

    return float((float(angle) + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _standard_error_from_variance(variance: object) -> float:
    """Return a standard error without turning invalid covariance into zero."""
    try:
        numeric_variance = float(variance)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(numeric_variance) or numeric_variance < 0.0:
        return float("nan")
    return float(np.sqrt(numeric_variance))


def _propagated_standard_error(
    gradient: np.ndarray,
    covariance: np.ndarray,
) -> float:
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(covariance)):
        return float("nan")
    variance = float(gradient @ covariance @ gradient)
    return _standard_error_from_variance(variance)


def calculate_gaussian_width_properties(
    fit_params: Mapping[str, object],
) -> dict[str, float]:
    """Derive profile-axis and canonical widths from a rotated Gaussian fit.

    ``sigma_x`` and ``sigma_y`` are the historical principal-axis parameters
    of :func:`trackleed.analysis.peak_analyzer.gaussian_2d`. Their labels can
    swap together with a 90-degree change in ``theta`` without changing the
    fitted image.

    ``fwhm_profile_x`` and ``fwhm_profile_y`` are center-crossing profile
    widths parallel to the fixed image axes. ``fwhm_minor`` and
    ``fwhm_major`` are ordered principal widths, while ``fwhm_effective`` is
    their geometric mean.
    """
    sigma_x = abs(float(fit_params["sigma_x"]))
    sigma_y = abs(float(fit_params["sigma_y"]))
    theta = float(fit_params["theta"])
    if (
        not np.isfinite(sigma_x)
        or not np.isfinite(sigma_y)
        or sigma_x <= 0.0
        or sigma_y <= 0.0
        or not np.isfinite(theta)
    ):
        raise ValueError("Gaussian widths and theta must be finite and positive")

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sigma_profile_x = float(
        1.0
        / np.sqrt(
            cos_theta**2 / sigma_x**2
            + sin_theta**2 / sigma_y**2
        )
    )
    sigma_profile_y = float(
        1.0
        / np.sqrt(
            sin_theta**2 / sigma_x**2
            + cos_theta**2 / sigma_y**2
        )
    )
    sigma_minor = min(sigma_x, sigma_y)
    sigma_major = max(sigma_x, sigma_y)
    effective_sigma = float(np.sqrt(sigma_x * sigma_y))

    circular = bool(np.isclose(sigma_x, sigma_y, rtol=1e-6, atol=1e-9))
    if circular:
        theta_major = float("nan")
    else:
        # gaussian_2d's mixed-term sign makes the sigma_x axis equal to -theta.
        theta_major = -theta if sigma_x > sigma_y else np.pi / 2.0 - theta
        theta_major = canonicalize_axial_angle(theta_major)

    covariance_matrix = np.asarray(
        fit_params.get("covariance_matrix"),
        dtype=float,
    )
    if covariance_matrix.shape != (7, 7):
        covariance_matrix = np.full((7, 7), np.nan, dtype=float)
    width_covariance = covariance_matrix[np.ix_([3, 4, 5], [3, 4, 5])]

    profile_x_gradient = FWHM_FACTOR * np.array(
        [
            cos_theta**2 * sigma_profile_x**3 / sigma_x**3,
            sin_theta**2 * sigma_profile_x**3 / sigma_y**3,
            (1.0 / sigma_x**2 - 1.0 / sigma_y**2)
            * sin_theta
            * cos_theta
            * sigma_profile_x**3,
        ],
        dtype=float,
    )
    profile_y_gradient = FWHM_FACTOR * np.array(
        [
            sin_theta**2 * sigma_profile_y**3 / sigma_x**3,
            cos_theta**2 * sigma_profile_y**3 / sigma_y**3,
            (1.0 / sigma_y**2 - 1.0 / sigma_x**2)
            * sin_theta
            * cos_theta
            * sigma_profile_y**3,
        ],
        dtype=float,
    )
    effective_gradient = FWHM_FACTOR * np.array(
        [
            0.5 * effective_sigma / sigma_x,
            0.5 * effective_sigma / sigma_y,
            0.0,
        ],
        dtype=float,
    )

    sigma_x_error = _standard_error_from_variance(width_covariance[0, 0])
    sigma_y_error = _standard_error_from_variance(width_covariance[1, 1])
    if sigma_x <= sigma_y:
        minor_error = FWHM_FACTOR * sigma_x_error
        major_error = FWHM_FACTOR * sigma_y_error
        anisotropy_gradient = np.array(
            [-sigma_y / sigma_x**2, 1.0 / sigma_x, 0.0],
            dtype=float,
        )
    else:
        minor_error = FWHM_FACTOR * sigma_y_error
        major_error = FWHM_FACTOR * sigma_x_error
        anisotropy_gradient = np.array(
            [1.0 / sigma_y, -sigma_x / sigma_y**2, 0.0],
            dtype=float,
        )

    if circular:
        minor_error = float("nan")
        major_error = float("nan")
        anisotropy_error = float("nan")
        theta_major_error = float("nan")
    else:
        anisotropy_error = _propagated_standard_error(
            anisotropy_gradient,
            width_covariance,
        )
        theta_major_error = _standard_error_from_variance(
            width_covariance[2, 2]
        )

    return {
        "fwhm_profile_x": float(FWHM_FACTOR * sigma_profile_x),
        "fwhm_profile_y": float(FWHM_FACTOR * sigma_profile_y),
        "fwhm_minor": float(FWHM_FACTOR * sigma_minor),
        "fwhm_major": float(FWHM_FACTOR * sigma_major),
        "fwhm_effective": float(FWHM_FACTOR * effective_sigma),
        "fwhm_anisotropy": float(sigma_major / sigma_minor),
        "fwhm_anisotropy_error": anisotropy_error,
        "fwhm_profile_x_error": _propagated_standard_error(
            profile_x_gradient,
            width_covariance,
        ),
        "fwhm_profile_y_error": _propagated_standard_error(
            profile_y_gradient,
            width_covariance,
        ),
        "fwhm_minor_error": float(minor_error),
        "fwhm_major_error": float(major_error),
        "fwhm_effective_error": _propagated_standard_error(
            effective_gradient,
            width_covariance,
        ),
        "theta_major": theta_major,
        "major_axis_angle_deg": float(np.degrees(theta_major)),
        "theta_major_error": theta_major_error,
        "major_axis_angle_deg_error": float(
            np.degrees(theta_major_error)
        ),
    }
