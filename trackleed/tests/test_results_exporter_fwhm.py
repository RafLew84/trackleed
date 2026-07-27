import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from trackleed.analysis.peak_analyzer import (
    calculate_spot_properties,
    gaussian_2d,
)
from trackleed.io.results_exporter import export_full_results


class ResultsExporterFwhmTests(unittest.TestCase):
    def test_full_export_contains_unambiguous_gaussian_widths(self) -> None:
        y, x = np.mgrid[0:41, 0:41]
        fit_params = {
            "amplitude": 1.0,
            "center_x": 20.0,
            "center_y": 20.0,
            "sigma_x": 2.0,
            "sigma_y": 4.0,
            "theta": 0.3,
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
        analysis_result = {
            **fit_params,
            **calculate_spot_properties(fit_params, roi_data, (x, y)),
            "global_center_x": 100.0,
            "global_center_y": 120.0,
            "radius": 25.0,
            "radius_error": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            image_path = output_dir / "frame_0001.png"
            Image.fromarray(
                np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
            ).save(image_path)
            track = SimpleNamespace(
                track_id=7,
                analysis_results={1: analysis_result},
                spots_by_image_id={},
            )
            leed_file = SimpleNamespace(
                filepath=image_path,
                filename=image_path.name,
                temperature=300.0,
                energy=150.0,
                coverage=0.0,
                pattern_center=None,
                post_analysis_results={},
            )

            export_full_results([track], {1: leed_file}, str(output_dir))

            with (output_dir / "full_analysis_per_spot.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                row = next(csv.DictReader(handle))

        expected_columns = {
            "sigma_x",
            "sigma_y",
            "theta",
            "fwhm_profile_x",
            "fwhm_profile_y",
            "fwhm_minor",
            "fwhm_major",
            "fwhm_effective",
            "fwhm_anisotropy",
            "fwhm_anisotropy_error",
            "major_axis_angle_deg",
        }
        self.assertTrue(expected_columns.issubset(row))


if __name__ == "__main__":
    unittest.main()
