import unittest

from trackleed.analysis.result_parameters import ANALYSIS_RESULT_PARAMETERS


class AnalysisDialogFwhmMetricTests(unittest.TestCase):
    def test_plot_metrics_use_unambiguous_fwhm_definitions(self) -> None:
        expected = {
            "FWHM Effective": "fwhm_effective",
            "FWHM Profile X (image axis)": "fwhm_profile_x",
            "FWHM Profile Y (image axis)": "fwhm_profile_y",
            "FWHM Minor": "fwhm_minor",
            "FWHM Major": "fwhm_major",
            "FWHM Anisotropy": "fwhm_anisotropy",
        }

        for label, field in expected.items():
            with self.subTest(label=label):
                self.assertEqual(ANALYSIS_RESULT_PARAMETERS[label][0], field)

        self.assertNotIn("FWHM X", ANALYSIS_RESULT_PARAMETERS)
        self.assertNotIn("FWHM Y", ANALYSIS_RESULT_PARAMETERS)
        self.assertEqual(
            ANALYSIS_RESULT_PARAMETERS[
                "Legacy Principal-Axis Ratio sigma_x/sigma_y"
            ][0],
            "asymmetry",
        )
        self.assertEqual(
            ANALYSIS_RESULT_PARAMETERS["FWHM Effective"][1],
            "fwhm_effective_error",
        )
        self.assertEqual(
            ANALYSIS_RESULT_PARAMETERS["FWHM Anisotropy"][1],
            "fwhm_anisotropy_error",
        )


if __name__ == "__main__":
    unittest.main()
