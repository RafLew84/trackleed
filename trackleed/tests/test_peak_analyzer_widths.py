import unittest

import numpy as np

from trackleed.analysis.peak_analyzer import (
    calculate_spot_properties,
    fit_spot_to_gaussian,
    gaussian_2d,
)


class GaussianWidthTests(unittest.TestCase):
    def _asymmetric_spot_roi(self) -> np.ndarray:
        y, x = np.mgrid[0:38, 0:38]
        roi_data = gaussian_2d(
            (x, y),
            1.0,
            18.0,
            18.0,
            4.792246244317086,
            4.423022483822282,
            1.884678103127568,
            0.08,
        ).reshape(x.shape)
        roi_data += gaussian_2d(
            (x, y),
            0.5640144145369694,
            14.250886166295894,
            10.335906762771708,
            2.172092470808118,
            6.811169545312385,
            -1.2365114508417054,
            0.0,
        ).reshape(x.shape)
        return roi_data

    def _properties(self, *, sigma_x: float, sigma_y: float, theta: float) -> dict:
        y, x = np.mgrid[0:41, 0:41]
        fit_params = {
            "amplitude": 1.0,
            "center_x": 20.0,
            "center_y": 20.0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta": theta,
            "offset": 0.0,
            "covariance_matrix": np.eye(7, dtype=float) * 1e-4,
        }
        roi_data = gaussian_2d(
            (x, y),
            **{
                key: value
                for key, value in fit_params.items()
                if key != "covariance_matrix"
            },
        ).reshape(x.shape)
        return calculate_spot_properties(fit_params, roi_data, (x, y))

    def test_image_axis_widths_are_invariant_to_equivalent_axis_swap(self) -> None:
        original = self._properties(sigma_x=2.0, sigma_y=4.0, theta=0.3)
        swapped = self._properties(
            sigma_x=4.0,
            sigma_y=2.0,
            theta=0.3 + np.pi / 2.0,
        )

        invariant_fields = (
            "fwhm_profile_x",
            "fwhm_profile_y",
            "fwhm_minor",
            "fwhm_major",
            "fwhm_effective",
            "fwhm_profile_x_error",
            "fwhm_profile_y_error",
            "fwhm_minor_error",
            "fwhm_major_error",
            "fwhm_effective_error",
            "fwhm_anisotropy_error",
            "theta_major",
            "theta_major_error",
            "major_axis_angle_deg_error",
        )
        for field in invariant_fields:
            with self.subTest(field=field):
                self.assertAlmostEqual(original[field], swapped[field], places=10)
        self.assertAlmostEqual(original["asymmetry"], 0.5)
        self.assertAlmostEqual(swapped["asymmetry"], 2.0)

    def test_image_axis_widths_are_center_crossing_profile_fwhm(self) -> None:
        properties = self._properties(
            sigma_x=2.0,
            sigma_y=4.0,
            theta=np.pi / 4.0,
        )
        expected_sigma = 1.0 / np.sqrt(0.5 / 2.0**2 + 0.5 / 4.0**2)
        factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

        self.assertAlmostEqual(
            properties["fwhm_profile_x"],
            factor * expected_sigma,
        )
        self.assertAlmostEqual(
            properties["fwhm_profile_y"],
            factor * expected_sigma,
        )

    def test_profile_error_matches_numerical_covariance_propagation(
        self,
    ) -> None:
        from trackleed.analysis.fwhm_metrics import (
            calculate_gaussian_width_properties,
        )

        covariance = np.zeros((7, 7), dtype=float)
        width_covariance = np.array(
            [
                [0.0100, 0.0010, 0.0002],
                [0.0010, 0.0225, -0.0001],
                [0.0002, -0.0001, 0.0004],
            ]
        )
        covariance[np.ix_([3, 4, 5], [3, 4, 5])] = width_covariance
        base = {
            "sigma_x": 2.5,
            "sigma_y": 4.0,
            "theta": 0.37,
            "covariance_matrix": covariance,
        }

        result = calculate_gaussian_width_properties(base)
        gradient = []
        for parameter in ("sigma_x", "sigma_y", "theta"):
            step = 1e-6
            lower = dict(base)
            upper = dict(base)
            lower[parameter] -= step
            upper[parameter] += step
            lower_value = calculate_gaussian_width_properties(lower)[
                "fwhm_profile_x"
            ]
            upper_value = calculate_gaussian_width_properties(upper)[
                "fwhm_profile_x"
            ]
            gradient.append((upper_value - lower_value) / (2.0 * step))

        gradient_array = np.asarray(gradient)
        expected_error = np.sqrt(
            gradient_array @ width_covariance @ gradient_array
        )
        self.assertAlmostEqual(
            result["fwhm_profile_x_error"],
            expected_error,
            places=8,
        )

    def test_image_axis_widths_cross_when_spot_shape_changes_direction(self) -> None:
        x_elongated = self._properties(
            sigma_x=4.0,
            sigma_y=2.0,
            theta=0.0,
        )
        y_elongated = self._properties(
            sigma_x=2.0,
            sigma_y=4.0,
            theta=0.0,
        )

        self.assertGreater(
            x_elongated["fwhm_profile_x"],
            x_elongated["fwhm_profile_y"],
        )
        self.assertLess(
            y_elongated["fwhm_profile_x"],
            y_elongated["fwhm_profile_y"],
        )
        self.assertAlmostEqual(
            x_elongated["fwhm_effective"],
            y_elongated["fwhm_effective"],
            places=10,
        )

    def test_canonical_widths_keep_shape_and_orientation_semantics(self) -> None:
        properties = self._properties(
            sigma_x=2.0,
            sigma_y=4.0,
            theta=np.deg2rad(20.0),
        )
        factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

        self.assertAlmostEqual(properties["fwhm_minor"], factor * 2.0)
        self.assertAlmostEqual(properties["fwhm_major"], factor * 4.0)
        self.assertAlmostEqual(
            properties["fwhm_effective"],
            factor * np.sqrt(8.0),
        )
        self.assertAlmostEqual(properties["fwhm_anisotropy"], 2.0)
        self.assertAlmostEqual(properties["major_axis_angle_deg"], 70.0)
        self.assertTrue(np.isfinite(properties["fwhm_anisotropy_error"]))

        circular = self._properties(
            sigma_x=3.0,
            sigma_y=3.0,
            theta=np.deg2rad(37.0),
        )
        self.assertTrue(np.isnan(circular["major_axis_angle_deg"]))
        self.assertTrue(np.isnan(circular["major_axis_angle_deg_error"]))
        self.assertTrue(np.isnan(circular["fwhm_anisotropy_error"]))
        self.assertTrue(np.isnan(circular["fwhm_minor_error"]))
        self.assertTrue(np.isnan(circular["fwhm_major_error"]))

    def test_invalid_covariance_produces_unknown_not_zero_errors(self) -> None:
        y, x = np.mgrid[0:41, 0:41]
        fit_params = {
            "amplitude": 1.0,
            "center_x": 20.0,
            "center_y": 20.0,
            "sigma_x": 2.0,
            "sigma_y": 4.0,
            "theta": 0.2,
            "offset": 0.0,
            "covariance_matrix": np.full((7, 7), np.nan),
        }
        roi_data = gaussian_2d(
            (x, y),
            **{
                key: value
                for key, value in fit_params.items()
                if key != "covariance_matrix"
            },
        ).reshape(x.shape)

        properties = calculate_spot_properties(
            fit_params,
            roi_data,
            (x, y),
        )

        for field in (
            "fwhm_profile_x_error",
            "fwhm_profile_y_error",
            "fwhm_minor_error",
            "fwhm_major_error",
            "fwhm_effective_error",
            "fwhm_anisotropy_error",
            "major_axis_angle_deg_error",
        ):
            with self.subTest(field=field):
                self.assertTrue(np.isnan(properties[field]))

    def test_fit_escapes_periodic_angle_boundary_for_asymmetric_spot(
        self,
    ) -> None:
        result = fit_spot_to_gaussian(self._asymmetric_spot_roi())

        self.assertIsNotNone(result)
        self.assertGreater(result["r_squared"], 0.94)

    def test_fit_returns_theta_in_canonical_period(self) -> None:
        result = fit_spot_to_gaussian(self._asymmetric_spot_roi())

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["theta"], -np.pi / 2.0)
        self.assertLess(result["theta"], np.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
