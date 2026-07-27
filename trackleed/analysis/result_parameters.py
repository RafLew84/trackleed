"""Shared result-field definitions for analysis tables and plots."""

from __future__ import annotations


GAUSSIAN_DERIVED_RESULT_PARAMETERS: dict[str, tuple[str, str | None]] = {
    "FWHM Profile X (image axis)": (
        "fwhm_profile_x",
        "fwhm_profile_x_error",
    ),
    "FWHM Profile Y (image axis)": (
        "fwhm_profile_y",
        "fwhm_profile_y_error",
    ),
    "FWHM Minor": ("fwhm_minor", "fwhm_minor_error"),
    "FWHM Major": ("fwhm_major", "fwhm_major_error"),
    "FWHM Effective": ("fwhm_effective", "fwhm_effective_error"),
    "FWHM Anisotropy": (
        "fwhm_anisotropy",
        "fwhm_anisotropy_error",
    ),
    "Major Axis Angle (deg)": (
        "major_axis_angle_deg",
        "major_axis_angle_deg_error",
    ),
}


GAUSSIAN_DERIVED_EXPORT_FIELDS: tuple[str, ...] = tuple(
    field_name
    for value_field, error_field in GAUSSIAN_DERIVED_RESULT_PARAMETERS.values()
    for field_name in (value_field, error_field)
    if field_name is not None
)


ANALYSIS_RESULT_PARAMETERS: dict[str, tuple[str, str | None]] = {
    "Radius": ("radius", "radius_error"),
    "Global Center X": ("global_center_x", "center_x_error"),
    "Global Center Y": ("global_center_y", "center_y_error"),
    "Volume": ("volume", "volume_error"),
    "Amplitude": ("amplitude", "amplitude_error"),
    **GAUSSIAN_DERIVED_RESULT_PARAMETERS,
    "Legacy Principal-Axis Ratio sigma_x/sigma_y": ("asymmetry", None),
    "R-Squared": ("r_squared", None),
}
